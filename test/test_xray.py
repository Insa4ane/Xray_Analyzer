from unittest.mock import patch, MagicMock
import torch as tc
import pytest
from torch.utils.data import TensorDataset, DataLoader

from agent.XrayAgent import XrayAgent
import numpy as np


@pytest.fixture
def dummy_agent():
    agent = XrayAgent()
    agent.input_channels = 1
    agent.channel_sizes = [16, 32]
    agent.size = 64
    agent.epochs = 5
    agent.feature_extractor = agent._build_feature_extractor()
    agent.classifier = agent._build_classifier()
    return agent



@patch.object(XrayAgent, '_build_classifier')
@patch.object(XrayAgent, '_build_feature_extractor')
def test_agent_initialization(mock_build_extractor, mock_build_classifier):
    mock_build_extractor.return_value = MagicMock()
    mock_build_classifier.return_value = MagicMock()

    agent = XrayAgent()

    assert agent is not None
    assert hasattr(agent, 'feature_extractor')
    assert hasattr(agent, 'classifier')
    mock_build_extractor.assert_called_once()
    mock_build_classifier.assert_called_once()


def test_build_feature_extractor(dummy_agent):
    dummy_agent.channel_sizes = [32, 64, 128]
    feature_extractor = dummy_agent._build_feature_extractor()

    assert isinstance(feature_extractor, tc.nn.Sequential)
    assert len(feature_extractor) == 6
    assert isinstance(feature_extractor[0], tc.nn.Conv2d)
    assert feature_extractor[0].in_channels == 1
    assert feature_extractor[0].out_channels == 32
    assert feature_extractor[0].kernel_size == (3, 3)

    assert isinstance(feature_extractor[1], tc.nn.ReLU)
    assert feature_extractor[2].in_channels == 32
    assert feature_extractor[2].out_channels == 64


@patch.object(XrayAgent, '_calculate_flattened_size', return_value=1024)
def test_build_classifier(mock_size, dummy_agent):
    classifier = dummy_agent._build_classifier()

    mock_size.assert_called_once()

    assert isinstance(classifier, tc.nn.Sequential)
    assert len(classifier) == 3

    assert isinstance(classifier[0], tc.nn.Flatten)
    assert isinstance(classifier[1], tc.nn.Linear)
    assert classifier[1].in_features == 1024
    assert classifier[1].out_features == 1
    assert isinstance(classifier[2], tc.nn.Sigmoid)


def test_calculate_flattened_size(dummy_agent):
    expected_size = 32 * 16 * 16

    flattened_size = dummy_agent._calculate_flattened_size()

    assert isinstance(flattened_size, int)
    assert flattened_size == expected_size


def test_forward_pass(dummy_agent):
    dummy_image = tc.randn(1, 1, 64, 64)

    output = dummy_agent.forward(dummy_image)

    assert output.shape == (1, 1)
    assert 0.0 <= output.item() <= 1.0

@patch.object(XrayAgent, 'predict')
def test_evaluate(mock_predict, dummy_agent):
    mock_predict.return_value = np.array([[1.0], [0.0], [1.0], [1.0]])
    y_test = tc.tensor([[1.0], [0.0], [1.0], [0.0]])
    dummy_X = tc.randn(4, 1, 64, 64)
    results = dummy_agent.evaluate(dummy_X, y_test)
    assert isinstance(results, dict)
    assert results['accuracy'] == 75.0

@patch.object(XrayAgent, 'forward')
def test_predict(mock_forward,dummy_agent):
    dummy_image = tc.randn(4, 1, 64, 64)
    mock_forward.return_value =tc.tensor([[0.12], [0.6], [0.49], [0.51]])
    predictions = dummy_agent.predict(dummy_image)
    dummy_results=np.array([[0.0], [1.0], [0.0], [1.0]])
    assert isinstance(predictions, np.ndarray)
    assert np.array_equal(predictions, dummy_results)

def test_fit(dummy_agent):
    x_train=tc.randn(4, 1, 64, 64)
    y_train=tc.tensor([[1.0], [0.0], [1.0], [0.0]])
    dummy_agent.epochs=1
    initial_weights = dummy_agent.feature_extractor[0].weight.clone()
    dummy_agent.fit(x_train, y_train)
    updated_weights = dummy_agent.feature_extractor[0].weight.clone()
    assert not tc.equal(initial_weights, updated_weights)

@patch.object(XrayAgent, '_calculate_metrics')
@patch.object(XrayAgent, '_train_one_epoch')
def test_fit_history_structure(mock_train_epoch, mock_calc_metrics, dummy_agent):
    dummy_agent.epochs = 3
    mock_train_epoch.return_value = (1.0, [1, 0], [1, 0])
    mock_calc_metrics.return_value = {
        'loss': 0.5, 'accuracy': 90.0, 'precision': 0.8, 'recall': 0.7, 'f1': 0.75
    }

    x_train = tc.randn(2, 1, 64, 64)
    y_train = tc.tensor([[1.0], [0.0]])

    history = dummy_agent.fit(x_train, y_train)

    assert set(history.keys()) == {'loss', 'accuracy', 'precision', 'recall', 'f1'}
    for key in history:
        assert len(history[key]) == 3
        assert all(v == mock_calc_metrics.return_value[key] for v in history[key])

    assert mock_train_epoch.call_count == 3
    assert mock_calc_metrics.call_count == 3

def test_score(dummy_agent):
    targets = [1, 0, 1, 1]
    preds = [1, 0, 0, 1]

    result = dummy_agent._score(targets, preds)

    assert result['accuracy'] == 75.0
    assert result['precision'] == 1.0
    assert round(result['recall'], 4) == 0.6667
    assert round(result['f1'], 4) == 0.8


def test_score_zero_division(dummy_agent):
    targets = [0, 0, 0, 0]
    preds = [0, 0, 0, 0]

    result = dummy_agent._score(targets, preds)

    assert result['precision'] == 0
    assert result['recall'] == 0
    assert result['f1'] == 0
    assert result['accuracy'] == 100.0


def test_calculate_metrics(dummy_agent):
    targets = [1, 0, 1, 1]
    preds = [1, 0, 0, 1]

    result = dummy_agent._calculate_metrics(epoch_loss=2.0, total_samples=4, targets=targets, preds=preds)

    assert result['loss'] == 0.5
    assert result['accuracy'] == 75.0
    assert result['precision'] == 1.0

@patch.object(XrayAgent, 'forward')
def test_train_one_epoch(mock_forward, dummy_agent):
    mock_forward.return_value = tc.tensor([[0.9], [0.1]], requires_grad=True)

    X = tc.randn(2, 1, 64, 64)
    y = tc.tensor([[1.0], [0.0]])
    dataset=TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2)
    criterion = tc.nn.BCELoss()
    optimizer = tc.optim.Adam(dummy_agent.parameters(), lr=1e-3)

    epoch_loss, targets, preds = dummy_agent._train_one_epoch(dataloader, criterion, optimizer)

    assert epoch_loss == pytest.approx(0.2107, abs=1e-3)
    assert len(targets) == 2
    assert len(preds) == 2
    assert preds[0][0] == 1.0
    assert preds[1][0] == 0.0



