from unittest.mock import patch

import torch as tc

from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
from DataLoader.Loader import DataLoader
from train import train


@patch.object(train, "Plotting")
@patch.object(DataLoader, "dataset_download", return_value="C:/fake/path")
@patch.object(XrayAgent, "save_model")
@patch.object(XGBoostAgent, "save_model")
@patch.object(XGBoostAgent, "run")
@patch.object(XrayAgent, "run")
@patch.object(DataLoader, "count_training_images_per_class")
@patch.object(DataLoader, "split_data")
@patch.object(XrayAgent, "save_history_and_results", return_value=True)
@patch.object(XGBoostAgent, "save_results", return_value=True)
def test_train_success(mock_xgb_results, mock_history_and_results, mock_split, mock_count,
                       mock_run_agent, mock_run_xgb, mock_save_xgb, mock_save_agent,
                       mock_download, mock_plotting):
    fake_x_train = tc.rand(3, 1, 64, 64)
    fake_y_train = tc.tensor([1.0, 0.0, 1.0], dtype=tc.float)
    fake_x_test = tc.rand(3, 1, 64, 64)
    fake_y_test = tc.tensor([1.0, 0.0, 1.0], dtype=tc.float)
    fake_x_val = tc.rand(2, 1, 64, 64)
    fake_y_val = tc.tensor([1.0, 0.0], dtype=tc.float)
    mock_split.return_value = (fake_x_train, fake_x_test, fake_y_train, fake_y_test, fake_x_val, fake_y_val)

    fake_history = {'loss': [0.5], 'val_loss': [0.4]}
    fake_result_agent = {'accuracy': 90.0}
    fake_result_xgb = {'accuracy': 91.0}
    mock_run_agent.return_value = (fake_history, fake_result_agent)
    mock_run_xgb.return_value = fake_result_xgb
    mock_count.return_value = {'PNEUMONIA': 3, 'NORMAL': 1}
    mock_plotting.return_value.plot_history.return_value = "plot.png"

    result = train.main()

    assert result is True

    args = mock_run_agent.call_args.args
    assert args[0] is fake_x_train
    assert args[1] is fake_x_val
    assert args[2] is fake_x_test
    assert args[3] is fake_y_train
    assert args[4] is fake_y_val
    assert args[5] is fake_y_test
    assert args[6] == {'PNEUMONIA': 3, 'NORMAL': 1}

    xgb_args = mock_run_xgb.call_args.args
    assert xgb_args[0] is fake_x_train
    assert xgb_args[1] is fake_x_test
    assert xgb_args[2] is fake_y_train
    assert xgb_args[3] is fake_y_test

    mock_history_and_results.assert_called_once_with(fake_history, fake_result_agent)
    mock_plotting.return_value.plot_history.assert_called_once_with(fake_history, True)
    mock_xgb_results.assert_called_once_with(fake_result_xgb)
    mock_save_agent.assert_called_once()
    mock_save_xgb.assert_called_once()


@patch.object(DataLoader, "dataset_download", return_value="C:/fake/path")
@patch.object(DataLoader, "split_data")
def test_train_exception(mock_split, mock_download):
    mock_split.side_effect = Exception("Error! we cannot split data")
    result = train.main()
    assert not result


@patch.object(train, "Plotting")
@patch.object(DataLoader, "dataset_download", return_value="C:/fake/path")
@patch.object(XrayAgent, "save_model")
@patch.object(XGBoostAgent, "save_model")
@patch.object(XGBoostAgent, "run")
@patch.object(XrayAgent, "run")
@patch.object(DataLoader, "count_training_images_per_class")
@patch.object(DataLoader, "split_data")
def test_train_stops_when_agent_training_fails(mock_split, mock_count, mock_run_agent, mock_run_xgb,
                                               mock_save_xgb, mock_save_agent, mock_download, mock_plotting):
    mock_split.return_value = (tc.rand(3, 1, 64, 64), tc.rand(3, 1, 64, 64),
                               tc.tensor([1.0, 0.0, 1.0]), tc.tensor([1.0, 0.0, 1.0]),
                               tc.rand(2, 1, 64, 64), tc.tensor([1.0, 0.0]))
    mock_count.return_value = {'PNEUMONIA': 3, 'NORMAL': 1}
    mock_run_agent.return_value = (None, None)

    result = train.main()

    assert not result
    mock_run_xgb.assert_not_called()
    mock_save_agent.assert_not_called()
    mock_save_xgb.assert_not_called()