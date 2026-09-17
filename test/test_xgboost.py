from unittest.mock import patch, MagicMock
import torch as tc
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from agent.XGBoostAgent import XGBoostAgent
import xgboost as xgb
import numpy as np
from agent.XrayAgent import XrayAgent

@pytest.fixture
def xgb_agent():
    xgb_agent= XGBoostAgent(XrayAgent())
    xgb_agent.agent=XrayAgent()
    xgb_agent.classifier=xgb.XGBClassifier()
    xgb_agent.batch_size=32
    xgb_agent.path_history = "fake/tmp/bubka"
    return xgb_agent

def test_xgboost_initialization():
    agent=XGBoostAgent(XrayAgent())
    assert agent is not None
    assert hasattr(agent, 'batch_size')
    assert hasattr(agent, 'classifier')
    assert hasattr(agent, 'agent')
    assert hasattr(agent, 'path')


def test_extract_features(xgb_agent):
    fake_X = tc.rand(2, 1, 64, 64)
    fake_extracted_features = tc.rand(2, 16, 4, 4)
    xgb_agent.agent.eval = MagicMock()
    with patch.object(xgb_agent.agent.feature_extractor, 'forward', return_value=fake_extracted_features):
        result = xgb_agent.extract_features(fake_X)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 256)


@patch.object(XGBoostAgent, 'extract_features')
def test_train(mock_extract_features, xgb_agent):
    fake_X_train = tc.rand(2, 1, 64, 64)
    fake_y_train = np.array([0, 1])

    fake_extracted_matrix = np.array([[0.1, 0.5], [0.8, 0.2]])
    mock_extract_features.return_value = fake_extracted_matrix

    xgb_agent.classifier.fit = MagicMock()
    xgb_agent.agent.eval = MagicMock()

    xgb_agent.train(fake_X_train, fake_y_train)

    mock_extract_features.assert_called_once_with(fake_X_train)

    xgb_agent.classifier.fit.assert_called_once()

    args, kwargs = xgb_agent.classifier.fit.call_args

    np.testing.assert_array_equal(args[0], fake_extracted_matrix)
    np.testing.assert_array_equal(args[1], fake_y_train)


@patch.object(XGBoostAgent, 'extract_features')
def test_predict_success(mock_extract_features, xgb_agent):
    fake_X_test = tc.rand(2, 1, 64, 64)
    fake_features = np.array([[0.1, 0.5], [0.8, 0.2]])
    fake_predictions = np.array([0, 1])

    mock_extract_features.return_value = fake_features
    xgb_agent.classifier.predict = MagicMock(return_value=fake_predictions)

    result = xgb_agent.predict(fake_X_test)

    mock_extract_features.assert_called_once_with(fake_X_test)
    xgb_agent.classifier.predict.assert_called_once()
    np.testing.assert_array_equal(result, fake_predictions)


@patch.object(XGBoostAgent, 'extract_features')
@patch('logging.error')
def test_predict_exception(mock_log, mock_extract_features, xgb_agent):
    mock_extract_features.side_effect = Exception("Awaria sieci!")

    result = xgb_agent.predict(tc.rand(2, 1, 64, 64))

    assert result is None
    mock_log.assert_called_once()
    assert "Something's gone wrong with predict" in mock_log.call_args[0][0]



@patch.object(XGBoostAgent, 'extract_features')
def test_evaluate(mock_extract_features, xgb_agent):
    fake_X_test = tc.rand(2, 1, 64, 64)
    fake_y_test = tc.tensor([[1], [0]])

    mock_extract_features.return_value = np.array([[0.9], [0.1]])

    xgb_agent.classifier.predict = MagicMock(return_value=np.array([1, 0]))

    result = xgb_agent.evaluate(fake_X_test, fake_y_test)

    assert result['accuracy'] == 100.0
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0

@patch('joblib.dump')
@patch("agent.XGBoostAgent.os.makedirs")
def test_save_model_success(mock_mkdir, mock_joblib_dump, xgb_agent):
    xgb_agent.path = "bubka/fake_xgb_model.joblib"
    xgb_agent.save_model()
    mock_joblib_dump.assert_called_once_with(xgb_agent.classifier, "bubka/fake_xgb_model.joblib")

@patch('joblib.load')
def test_load_model_success(mock_joblib_load, xgb_agent):
    xgb_agent.path = "bubka/fake_xgb_model.joblib"
    mock_joblib_load.return_value = "FakeLoadedModel"
    xgb_agent.load_model()
    mock_joblib_load.assert_called_once_with("bubka/fake_xgb_model.joblib")
    assert xgb_agent.classifier == "FakeLoadedModel"



@patch.object(XGBoostAgent, 'train')
@patch.object(XGBoostAgent, 'evaluate')
def test_run_success(mock_evaluate, mock_train, xgb_agent):
    mock_evaluate.return_value = {'accuracy': 95.0, 'f1': 0.9}
    result = xgb_agent.run("X_train", "X_test", "y_train", "y_test")
    mock_train.assert_called_once_with("X_train", "y_train")
    mock_evaluate.assert_called_once_with("X_test", "y_test")
    assert result['accuracy'] == 95.0


@patch.object(XGBoostAgent, 'train')
@patch('logging.error')
def test_run_exception(mock_log, mock_train, xgb_agent):
    mock_train.side_effect = Exception("Błąd pamięci")
    result = xgb_agent.run("X_train", "X_test", "y_train", "y_test")
    assert result is None
    mock_log.assert_called_once()
    assert "Something's gone wrong with train xgb" in mock_log.call_args[0][0]

@patch("agent.XGBoostAgent.json.dump")
@patch("builtins.open")
@patch("agent.XGBoostAgent.os.makedirs")
def test_save_results_success(mock_makedirs, mock_open, mock_json_dump, xgb_agent):

    fake_results = {'accuracy': 85, 'precision': 85, 'recall': 30, 'f1': 30}
    success = xgb_agent.save_results(fake_results)
    assert success is True
    mock_json_dump.assert_called_once()
    args, kwargs = mock_json_dump.call_args
    assert args[0] == fake_results

@patch("agent.XGBoostAgent.os.makedirs")
@patch("agent.XGBoostAgent.json.dump")
@patch("builtins.open")
def test_save_results_exception(mock_open, mock_json_dump,mock_mkdir, xgb_agent):

    mock_json_dump.side_effect = Exception("Awaria sieci!")
    fake_results = {'accuracy': 85, 'precision': 85, 'recall': 30, 'f1': 30}
    success = xgb_agent.save_results(fake_results)
    assert success is False



