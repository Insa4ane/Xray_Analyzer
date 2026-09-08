import torch as tc
from config.config import INPUT_CHANNELS, CHANNEL_SIZES, IMG_SIZE, EPOCHS
from torch.utils.data import TensorDataset, DataLoader
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class XrayAgent(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = INPUT_CHANNELS
        self.channel_sizes = CHANNEL_SIZES
        self.size=IMG_SIZE
        self.epochs=EPOCHS
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = self._build_classifier()

    def _build_feature_extractor(self):
        layers = []
        current_in_c = self.input_channels

        for out_c in self.channel_sizes:
            layers.append(tc.nn.Conv2d(in_channels=current_in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1))
            layers.append(tc.nn.ReLU())
            current_in_c = out_c
        return tc.nn.Sequential(*layers)

    def _build_classifier(self):
        flattened_size = self._calculate_flattened_size()

        return tc.nn.Sequential(
            tc.nn.Flatten(),
            tc.nn.Linear(in_features=flattened_size, out_features=1),
            tc.nn.Sigmoid()
        )

    def _calculate_flattened_size(self):
        dummy_input = tc.zeros(1, self.input_channels, self.size, self.size)
        with tc.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        return flattened_size

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def fit(self, X_train, y_train, lr: float = 1e-3, batch_size: int = 32) -> dict:
        self.train()
        criterion = tc.nn.BCELoss()
        optimizer = tc.optim.Adam(self.parameters(), lr=lr)
        dataset=TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for epoch in range(self.epochs):
            epoch_loss, all_targets, all_preds = self._train_one_epoch(dataloader, criterion, optimizer)

            metrics = self._calculate_metrics(epoch_loss, len(dataset), all_targets, all_preds)

            for key in history.keys():
                history[key].append(metrics[key])

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
            optimizer.zero_grad()
            prediction = self.forward(batch_X)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

            binary_preds = (prediction > 0.5).float()
            all_preds.extend(binary_preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        return epoch_loss, all_targets, all_preds

    def _score(self, targets, preds) -> dict:
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
        with tc.no_grad():
            predictions = self.forward(X_test)
        binary_predictions = (predictions > 0.5).float()

        return binary_predictions.numpy()

    def evaluate(self, X_test, y_test) -> dict:
        predictions = self.predict(X_test)
        if tc.is_tensor(y_test):
            y_test = y_test.numpy()
        return self._score(y_test, predictions)





