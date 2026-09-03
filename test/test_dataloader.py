import os
import pytest
from unittest.mock import patch
from DataLoader.DataLoader import DataLoader
from config.config import PATH


@patch("DataLoader.DataLoader.kagglehub.dataset_download")
def test_dataloader_initialization(mock_download):
    mock_download.return_value = "C:/fake/path"

    loader = DataLoader()

    assert loader is not None
    assert loader.path == "C:/fake/path"


@patch("DataLoader.DataLoader.kagglehub.dataset_download")
def test_dataloader_dataset_download(mock_download):
    mock_download.return_value = "C:/fake/path"
    loader = DataLoader()
    result = loader.dataset_download()
    assert result == "C:/fake/path"
    mock_download.assert_called_once_with(PATH)


# @patch("DataLoader.DataLoader.kagglehub.open_file")
# def test_open_file(mock_open):
#     mock_open.return_value =
