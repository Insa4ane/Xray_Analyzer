import logging

import joblib
import xgboost as xgb
import torch as tc
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config.config import BATCH_SIZE, PATH_XGB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class XGBoostAgent:

    def __init__(self, agent, **xgb_params):
        self.agent = agent
        self.classifier = xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            **xgb_params
        )
        self.batch_size=BATCH_SIZE
        self.path=PATH_XGB
    def extract_features(self, X) -> np.ndarray:  #for xgboost
        self.agent.eval()
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_features = []
        with tc.no_grad():
            for batch_X, batch_Y in dataloader:
                features = self.agent.feature_extractor(batch_X)
                features_flattened = features.view(features.size(0), -1)
                all_features.append(features_flattened.cpu().numpy())
            return np.vstack(all_features)

    def train(self, X_train, y_train):
        try:
            X_features = self.extract_features(X_train)
            if tc.is_tensor(y_train):
                y_train = y_train.cpu().numpy().ravel()
            else:
                y_train=y_train.ravel()
            self.classifier.fit(X_features, y_train)
            logging.info(f"Trained XGBoost model")
        except Exception as e:
            logging.error(f"Something's gone wrong with train model. {e}")
            return None

    def predict(self, X_test):
        try:
            X_test_features = self.extract_features(X_test)
            predictions=self.classifier.predict(X_test_features)
            logging.info(f"Predicted XGBoost model")
            return predictions
        except Exception as e:
            logging.error(f"Something's gone wrong with predict. {e}")
            return None

    def evaluate(self, X_test, y_test):
        X_test_features = self.extract_features(X_test)
        predictions = self.classifier.predict(X_test_features)
        if tc.is_tensor(y_test):
            y_test = y_test.cpu().numpy().ravel()
        else:
            y_test=y_test.ravel()
        return {
            'accuracy': accuracy_score(y_test, predictions) * 100,
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        }
    def save_model(self):
        try:
            joblib.dump(self.classifier, self.path)
            logging.info(f"Saved XGBoost model")
        except Exception as e:
            logging.error(f"Something's gone wrong with save model (XGBOOST) {e}")

    def load_model(self):
        try:
            self.classifier = joblib.load(self.path)
        except Exception as e:
            logging.error(f"Something's gone wrong with load model XGB. {e}")

    def run(self, X_train, X_test, y_train, y_test):
        try:
            self.train(X_train, y_train)
            results=self.evaluate(X_test, y_test)
            logging.info(f"Results XGB: {results}")
            return results
        except Exception as e:
            logging.error(f"Something's gone wrong with train xgb. {e}")
            return None

