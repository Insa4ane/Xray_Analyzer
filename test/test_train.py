from unittest.mock import patch, MagicMock
import torch as tc
from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
from train import train
from DataLoader.Loader import DataLoader
import numpy as np

@patch.object(XrayAgent, "save_model")
@patch.object(XGBoostAgent, "save_model")
@patch.object(XGBoostAgent, "run")
@patch.object(XrayAgent, "run")
@patch.object(DataLoader, "count_training_images_per_class")
@patch.object(DataLoader, "split_data")
def test_train_success(mock_data, mock_count, mock_run_agent, mock_run_xgb, mock_save_agent, mock_save_xgb):
    fake_x_train=tc.rand(3,1, 64, 64)
    fake_y_train=tc.tensor([1.0, 0.0, 1.0], dtype=tc.float)
    fake_x_test = tc.rand(3, 1, 64, 64)
    fake_y_test=tc.tensor([1.0, 0.0, 1.0], dtype=tc.float)
    mock_data.return_value = [fake_x_train, fake_x_test, fake_y_train, fake_y_test]
    mock_run_agent.return_value=(MagicMock(), MagicMock())
    mock_run_xgb.return_value=MagicMock()
    mock_count.return_value = 3
    result=train.main()
    assert result
    mock_run_agent.assert_called_once()
    mock_run_xgb.assert_called_once()
    mock_save_agent.assert_called_once()
    mock_save_xgb.assert_called_once()




