import copy
import datetime
import json
import logging
import os

import torch as tc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.config import INPUT_CHANNELS, CHANNEL_SIZES, IMG_SIZE, EPOCHS, BATCH_SIZE, PATH_MODEL, PATH_HISTORY


class XrayAgent(tc.nn.Module):
    def __init__(self, lr: float = 1e-3, patience: int = 5, min_delta: float = 1e-3):
        super().__init__()
        self.input_channels = INPUT_CHANNELS
        self.channel_sizes = CHANNEL_SIZES
        self.size = IMG_SIZE
        self.epochs = EPOCHS
        self.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")  # we check that gpu is available
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()
        self.batch_size = BATCH_SIZE
        self.path = PATH_MODEL
        self.to(self.device)
        self.path_history = PATH_HISTORY

        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta

        self.hparams = {
            "lr": self.lr,
            "optimizer": "Adam",
            "batch_size": self.batch_size,
            "epochs_limit": self.epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "channel_sizes": str(self.channel_sizes),
            "dropout": 0.3,
            "img_size": self.size,
        }
        self.tb_dir = None


    def _build_feature_extractor(self):
        layers = []
        current_in_c = self.input_channels

        for out_c in self.channel_sizes:
            layers.append(tc.nn.Conv2d(in_channels=current_in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1))
            layers.append(tc.nn.ReLU())
            layers.append(tc.nn.Dropout2d(p=0.3))
            current_in_c = out_c
        return tc.nn.Sequential(*layers)

    def _build_classifier(self):
        flattened_size = self._calculate_flattened_size()

        return tc.nn.Sequential(
            tc.nn.Flatten(),
            tc.nn.Linear(in_features=flattened_size, out_features=1),
            # tc.nn.Sigmoid()
        )

    def _calculate_flattened_size(self):
        dummy_input = tc.zeros(1, self.input_channels, self.size, self.size).to(self.device)
        with tc.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


    def fit(self, X_train, y_train, X_val, y_val, counts) -> dict:
        criterion, optimizer = self._build_training_tools(counts)
        train_loader, val_loader = self._build_loaders(X_train, y_train, X_val, y_val)
        self._init_hparams(counts, X_train, X_val)
        writer = self._create_writer()

        history = self._empty_history()
        best = {'loss': float('inf'), 'recall': 0.0, 'epoch': 0, 'state': None}
        patience_counter = 0

        try:
            for epoch in range(self.epochs):
                self.train()
                epoch_loss, t_targets, t_preds = self._train_one_epoch(train_loader, criterion, optimizer)
                metrics = self._calculate_metrics(epoch_loss, len(X_train), t_targets, t_preds)

                val_loss_sum, v_targets, v_preds = self._validate(val_loader, criterion)
                val_metrics = self._calculate_metrics(val_loss_sum, len(X_val), v_targets, v_preds)

                self._record_epoch(history, writer, metrics, val_metrics, epoch)

                if best['loss'] - val_metrics['loss'] > self.min_delta:
                    self._update_best(best, val_metrics, epoch)
                    patience_counter = 0
                else:
                    patience_counter += 1

                self._log_epoch(epoch, metrics, val_metrics)

                if patience_counter >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break

            self._finish_training(best, history, writer)
        finally:
            writer.close()

        return history

    def _build_training_tools(self, counts):
        pos_weight_value = counts['PNEUMONIA'] / counts['NORMAL']
        pos_weight = tc.tensor([pos_weight_value], dtype=tc.float32).to(self.device)
        criterion = tc.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = tc.optim.Adam(self.parameters(), lr=self.lr)
        return criterion, optimizer

    def _build_loaders(self, X_train, y_train, X_val, y_val):
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _init_hparams(self, counts, X_train, X_val):
        self.hparams.update({
            "pos_weight": float(counts['PNEUMONIA'] / counts['NORMAL']),
            "train_size": len(X_train),
            "val_size": len(X_val),
        })
        logging.info(f"Hyperparameters: {self.hparams}")

    def _create_writer(self):
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_dir = os.path.join(self.path_history, "tb", run_name)
        return SummaryWriter(log_dir=self.tb_dir)

    @staticmethod
    def _empty_history() -> dict:
        keys = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        return {**{k: [] for k in keys}, **{f'val_{k}': [] for k in keys}}

    @staticmethod
    def _record_epoch(history, writer, metrics, val_metrics, epoch):
        for key in metrics:
            history[key].append(metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        for name in ("loss", "accuracy", "recall", "f1"):
            writer.add_scalars(name, {"train": metrics[name], "val": val_metrics[name]}, epoch)

    def _update_best(self, best, val_metrics, epoch):
        best['loss'] = val_metrics['loss']
        best['recall'] = val_metrics['recall']
        best['epoch'] = epoch + 1
        best['state'] = copy.deepcopy(self.state_dict())

    def _log_epoch(self, epoch, metrics, val_metrics):
        logging.info(
            f"Epoch [{epoch + 1}/{self.epochs}] - "
            f"Loss: {metrics['loss']:.4f} | Val loss: {val_metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.2f}% | Val acc: {val_metrics['accuracy']:.2f}% | "
            f"Val recall: {val_metrics['recall']:.2f}"
        )

    def _finish_training(self, best, history, writer):
        if best['state'] is not None:
            self.load_state_dict(best['state'])

        self.hparams.update({
            "best_epoch": best['epoch'],
            "best_val_loss": float(best['loss']),
            "stopped_at_epoch": len(history['loss']),
        })
        logging.info(f"Best epoch: {best['epoch']}, best val loss: {best['loss']:.4f}")

        result_keys = ("best_epoch", "best_val_loss", "stopped_at_epoch")
        settings = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for k, v in self.hparams.items() if k not in result_keys}
        writer.add_hparams(
            settings,
            {"hparam/val_loss": float(best['loss']),
             "hparam/val_recall": float(best['recall']),
             "hparam/best_epoch": best['epoch']},
        )

    def _train_one_epoch(self, dataloader, criterion, optimizer) -> tuple:
        epoch_loss = 0.0
        all_preds, all_targets = [], []

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device=self.device), batch_y.to(device=self.device)
            batch_y = batch_y.view(-1, 1)
            optimizer.zero_grad()
            prediction = self.forward(batch_X)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            probabilities = tc.sigmoid(prediction)
            binary_preds = (probabilities > 0.5).float()
            all_preds.extend(binary_preds.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

        return epoch_loss, all_targets, all_preds

    def _validate(self, dataloader, criterion) -> tuple:
        self.eval()  # wylacza dropout
        total_loss = 0.0
        all_preds, all_targets = [], []
        with tc.no_grad():  # bez gradientow
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).view(-1, 1)
                logits = self.forward(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                preds = (tc.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        return total_loss, all_targets, all_preds


    @staticmethod
    def _score(targets, preds) -> dict:
        return {
            'accuracy': float(accuracy_score(targets, preds) * 100),
            'precision': float(precision_score(targets, preds, zero_division=0)),
            'recall': float(recall_score(targets, preds, zero_division=0)),
            'f1': float(f1_score(targets, preds, zero_division=0))
        }

    def _calculate_metrics(self, epoch_loss, total_samples, targets, preds) -> dict:
        metrics = self._score(targets, preds)
        metrics['loss'] = epoch_loss / total_samples
        return metrics

    def predict(self, X_test):
        self.eval()
        if not tc.is_tensor(X_test):
            X_test = tc.tensor(X_test, dtype=tc.float32)
        X_test = X_test.to(self.device)
        with tc.no_grad():
            predictions = self.forward(X_test)
            probabilities = tc.sigmoid(predictions)
        binary_predictions = (probabilities > 0.5).float()

        return binary_predictions.cpu().numpy()

    def evaluate(self, X_test, y_test) -> dict:
        predictions = self.predict(X_test)
        if tc.is_tensor(y_test):
            y_test = y_test.cpu().numpy()
        return self._score(y_test, predictions)


    def save_model(self):
        try:
            directory = os.path.dirname(self.path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            tc.save(self.state_dict(), self.path)
            logging.info(f"Saved model to {self.path}")
        except Exception as e:
            logging.error(f"We cannot save the model (xray_agent) {e}")

    def load_model(self):
        try:
            state_dict = tc.load(self.path, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
            logging.info(f"Loaded model from {self.path}")
            self.eval()
            return True
        except Exception as e:
            logging.warning(f"We cannot load the model (xray_agent) {e}")
            return False

    def save_history_and_results(self, history, results):
        try:
            if not os.path.exists(self.path_history):
                os.makedirs(self.path_history)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path_history = os.path.join(self.path_history, f"history_{timestamp}.json")
            path_results = os.path.join(self.path_history, f"results_agent_{timestamp}.json")
            with open(path_history, "w") as f:
                json.dump(history, f, indent=4)
            logging.info(f"Saved history to {path_history}")
            with open(path_results, "w") as f:
                json.dump({"hyperparameters": self.hparams, "test_results": results}, f, indent=4)
            logging.info(f"Saved results to {path_results}")
            return True
        except Exception as e:
            logging.error(f"We cannot save the history ({e}) to {self.path_history}")
            return False

    def run(self, X_train, X_val, X_test, y_train, y_val, y_test, counts):
        try:
            history = self.fit(X_train, y_train, X_val, y_val, counts)
            results = self.evaluate(X_test, y_test)
            logging.info(f"Results: {results}")
            return history, results
        except Exception as e:
            logging.error(f"We have a problem with our neural network (method run in XrayAgent) {e}")
            return None, None