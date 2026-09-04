import pytest
import cv2
import numpy as np
from unittest.mock import patch

from DataLoader.DataLoader import DataLoader
from config.config import PATH

@pytest.fixture
def loader_instance():
    """Zwraca gotowy obiekt DataLoader, omijając pobieranie danych z internetu."""
    with patch("DataLoader.DataLoader.kagglehub.dataset_download", return_value="C:/fake/path"):
        return DataLoader()

@patch("DataLoader.DataLoader.kagglehub.dataset_download")
def test_dataloader_initialization(mock_download):
    mock_download.return_value = "C:/fake/path"
    loader = DataLoader()

    assert loader is not None
    assert loader.path == "C:/fake/path"
    mock_download.assert_called_once_with(PATH)


@patch("DataLoader.DataLoader.kagglehub.dataset_download")
def test_dataloader_dataset_download(mock_download):
    mock_download.return_value = "C:/fake/path"
    loader = DataLoader()
    result = loader.dataset_download()

    assert result == "C:/fake/path"

@patch("DataLoader.DataLoader.cv2.imread")
@patch("DataLoader.DataLoader.cv2.resize")
def test_process_image(mock_resize, mock_imread, loader_instance):
    fake_img = np.zeros((300, 300), dtype=np.uint8)
    fake_resized_img = np.zeros((224, 224), dtype=np.uint8)
    mock_imread.return_value = fake_img
    mock_resize.return_value = fake_resized_img
    test_image_path = "fake/image.jpg"
    result = loader_instance.process_image(test_image_path)

    assert result is not None
    assert np.array_equal(result, fake_resized_img)
    mock_imread.assert_called_once_with(test_image_path, cv2.IMREAD_GRAYSCALE)


@patch("DataLoader.DataLoader.os.path.exists")
@patch("DataLoader.DataLoader.os.listdir")
@patch("DataLoader.DataLoader.CATEGORIES", {"fake_category": 1})
@patch.object(DataLoader, "process_image")
def test_build_dataset(mock_process, mock_listdir, mock_exist, loader_instance):
    mock_exist.return_value = True
    mock_listdir.return_value = ["img1.jpg", "img2.jpg"]

    fake_img = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    mock_process.return_value = fake_img
    fake_dir = "C:/fake/path"
    result = loader_instance.build_dataset(fake_dir)

    assert result is not None
    assert len(result) == 2
    assert np.array_equal(result[0][0], fake_img)
    assert result[0][1] == 1


@patch.object(DataLoader, "build_dataset")
def test_split_data(mock_build, loader_instance):
    mock_train_data = ["fake_train_item_1", "fake_train_item_2"]
    mock_test_data = ["fake_test_item"]
    mock_build.side_effect = [mock_train_data, mock_test_data]
    train, test = loader_instance.split_data()

    assert train == mock_train_data
    assert test == mock_test_data