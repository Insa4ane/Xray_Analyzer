import torch as tc
from config.config import INPUT_CHANNELS, CHANNEL_SIZES, IMG_SIZE, EPOCHS, BATCH_SIZE, PATH_MODEL
from torch.utils.data import TensorDataset, DataLoader
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

class XrayAgent(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = INPUT_CHANNELS
        self.channel_sizes = CHANNEL_SIZES
        self.size=IMG_SIZE
        self.epochs=EPOCHS
        self.device = tc.device("cuda" if tc.cuda.is_available() else "cpu")  # we check that gpu is available
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()
        self.batch_size=BATCH_SIZE
        self.path=PATH_MODEL
        self.to(self.device)

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
            #tc.nn.Sigmoid()
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

    def fit(self, X_train, y_train, counts, lr: float = 1e-3, min_delta=1e-3, patience=5) -> dict:
        self.train()
        pos_weight_value=counts['PNEUMONIA'] / counts['NORMAL']
        pos_weight = tc.tensor([pos_weight_value], dtype=tc.float32).to(self.device)
        criterion = tc.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = tc.optim.Adam(self.parameters(), lr=lr)
        dataset=TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        best_loss=float('inf')
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss, all_targets, all_preds = self._train_one_epoch(dataloader, criterion, optimizer)

            metrics = self._calculate_metrics(epoch_loss, len(dataset), all_targets, all_preds)

            for key in history.keys():
                history[key].append(metrics[key])

            if best_loss - metrics['loss'] > min_delta:
                best_loss = metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >=patience:
                logging.info(f"Early stopping! {patience_counter} patience")
                break

            logging.info(
                f"Epoch [{epoch + 1}/{self.epochs}] - Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | P: {metrics['precision']:.2f} | "
                f"R: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}"
            )

        return history

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

    @staticmethod
    def _score(targets, preds) -> dict:
        return {
            'accuracy': accuracy_score(targets, preds) * 100,
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0)
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
            directory=os.path.dirname(self.path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            tc.save(self.state_dict(), self.path)
            logging.info(f"Saved model to {self.path}")
        except Exception as e:
            logging.error(f"We cannot save the model (xray_agent) {e}")

    def load_model(self):
        try:
            state_dict=tc.load(self.path, map_location=self.device, weights_only=True)
            self.load_state_dict(state_dict)
            logging.info(f"Loaded model from {self.path}")
            self.eval()
            return True
        except Exception as e:
            logging.warning(f"We cannot load the model (xray_agent) {e}")
            return False


    def run(self, X_train, X_test, y_train, y_test, counts):
        try:
            history=self.fit(X_train, y_train, counts, lr=1e-3, min_delta=1e-3, patience=5)
            results=self.evaluate(X_test, y_test)
            logging.info(f"Results: {results}")
            return history, results
        except Exception as e:
            logging.error(f"We have a problem with our neural network (method run in XrayAgent) {e}")
            return None, None














