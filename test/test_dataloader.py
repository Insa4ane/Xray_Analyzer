import pytest
import cv2
import numpy as np
from unittest.mock import patch
import torch as tc
from numpy import dtypes

from DataLoader.Loader import DataLoader
from config.config import PATH, IMG_SIZE

@pytest.fixture
def loader_instance():
    with patch("DataLoader.Loader.kagglehub.dataset_download", return_value="C:/fake/path"):
        return DataLoader()

@patch("DataLoader.Loader.kagglehub.dataset_download")
def test_dataloader_initialization(mock_download):
    mock_download.return_value = "C:/fake/path"
    loader = DataLoader()

    assert loader is not None
    assert loader.path == "C:/fake/path"
    assert loader.size == IMG_SIZE
    mock_download.assert_called_once_with(PATH)


@patch("DataLoader.Loader.kagglehub.dataset_download")
def test_dataloader_dataset_download(mock_download):
    mock_download.return_value = "C:/fake/path"
    loader = DataLoader()
    result = loader.dataset_download()

    assert result == "C:/fake/path"

@patch("DataLoader.Loader.cv2.imread")
@patch("DataLoader.Loader.cv2.resize")
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


@patch("DataLoader.Loader.os.path.exists")
@patch("DataLoader.Loader.os.listdir")
@patch("DataLoader.Loader.CATEGORIES", {"fake_category": 1})
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
@patch.object(DataLoader, "prepare_for_training")
def test_split_data(mock_prepare, mock_build, loader_instance):
    mock_train_data = ["fake_train_item"]
    mock_test_data = ["fake_test_item"]
    mock_build.side_effect = [mock_train_data, mock_test_data]
    size=loader_instance.size
    X_train_mock = np.full((size, size), 1, dtype=np.uint8)
    y_train_mock = np.full((size,), 0, dtype=np.uint8)

    X_test_mock = np.full((size, size), 0.25, dtype=np.uint8)
    y_test_mock = np.full((size,), 1, dtype=np.uint8)

    mock_prepare.side_effect = [
        (X_test_mock, y_test_mock),
        (X_train_mock, y_train_mock)
    ]

    X_train, X_test, y_train, y_test = loader_instance.split_data()

    assert X_train is not None
    assert X_test is not None

    assert np.array_equal(X_train, X_train_mock)
    assert np.array_equal(X_test, X_test_mock)
    assert np.array_equal(y_train, y_train_mock)
    assert np.array_equal(y_test, y_test_mock)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_prepare_for_training_success(loader_instance):
    img_size=loader_instance.size
    fake_img=np.full((img_size,img_size), 255, dtype=np.uint8)
    fake_dataset = [[fake_img, 1], [fake_img, 0]]
    expected_x = tc.ones((2, 1, img_size, img_size), dtype=tc.float32)
    expected_y = tc.tensor([1, 0], dtype=tc.long)
    X, y = loader_instance.prepare_for_training(fake_dataset)
    assert X is not None
    assert y is not None

    assert tc.equal(X, expected_x)
    assert tc.equal(y, expected_y)

def test_prepare_for_training_exception(loader_instance):
    faulty_dataset = [["to_nie_jest_tablica_zdjecia", 1]]
    result = loader_instance.prepare_for_training(faulty_dataset)
    assert result is None


