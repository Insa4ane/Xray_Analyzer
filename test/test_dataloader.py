from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch as tc

from DataLoader.Loader import DataLoader
from config.config import PATH, IMG_SIZE

N = 40


@pytest.fixture
def loader_instance():
    with patch("DataLoader.Loader.kagglehub.dataset_download", return_value="C:/fake/path"):
        return DataLoader()


def fake_splits():
    X_train = np.arange(N, dtype=np.float32).reshape(N, 1, 1, 1)
    y_train = np.array([i % 2 for i in range(N)], dtype=np.float32)
    X_test = np.arange(100, 110, dtype=np.float32).reshape(10, 1, 1, 1)
    y_test = np.array([i % 2 for i in range(10)], dtype=np.float32)
    return (X_train, y_train), (X_test, y_test)


def run_split(loader_instance):
    train, test = fake_splits()
    with patch.object(DataLoader, "build_dataset", side_effect=[["fake_train_item"], ["fake_test_item"]]), \
         patch.object(DataLoader, "prepare_for_training", side_effect=[test, train]):
        return loader_instance.split_data()



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


@patch("DataLoader.Loader.cv2.imread", return_value=None)
def test_process_image_unreadable_file_returns_none(mock_imread, loader_instance):
    assert loader_instance.process_image("fake/broken.jpg") is None


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


@patch("DataLoader.Loader.os.listdir", return_value=["a.jpg", "b.jpg", "c.jpg"])
@patch("DataLoader.Loader.CATEGORIES", {"NORMAL": 1, "PNEUMONIA": 0})
def test_count_training_images_per_class(mock_listdir, loader_instance):
    result = loader_instance.count_training_images_per_class()
    assert result == {"NORMAL": 3, "PNEUMONIA": 3}



@patch.object(DataLoader, "build_dataset")
@patch.object(DataLoader, "prepare_for_training")
def test_split_data(mock_prepare, mock_build, loader_instance):
    mock_build.side_effect = [["fake_train_item"], ["fake_test_item"]]
    (X_train_mock, y_train_mock), (X_test_mock, y_test_mock) = fake_splits()
    mock_prepare.side_effect = [(X_test_mock, y_test_mock), (X_train_mock, y_train_mock)]

    X_train, X_test, y_train, y_test, X_val, y_val = loader_instance.split_data()

    assert np.array_equal(X_test, X_test_mock)
    assert np.array_equal(y_test, y_test_mock)

    assert len(X_val) == 6
    assert len(X_train) == 34
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    train_ids = set(np.asarray(X_train).flatten().tolist())
    val_ids = set(np.asarray(X_val).flatten().tolist())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(X_train_mock.flatten().tolist())

    assert np.asarray(y_val).sum() == 3
    assert np.asarray(y_train).sum() == 17


def test_split_data_is_reproducible(loader_instance):
    X_train_1, _, _, _, X_val_1, y_val_1 = run_split(loader_instance)
    X_train_2, _, _, _, X_val_2, y_val_2 = run_split(loader_instance)

    assert np.array_equal(X_val_1, X_val_2)
    assert np.array_equal(y_val_1, y_val_2)
    assert np.array_equal(X_train_1, X_train_2)


@patch.object(DataLoader, "build_dataset")
def test_split_data_empty_dataset_raises(mock_build, loader_instance):
    mock_build.side_effect = [[], ["fake_test_item"]]
    with pytest.raises(FileNotFoundError):
        loader_instance.split_data()


@patch.object(DataLoader, "build_dataset")
@patch.object(DataLoader, "prepare_for_training", return_value=None)
def test_split_data_prepare_failure_raises(mock_prepare, mock_build, loader_instance):
    mock_build.side_effect = [["fake_train_item"], ["fake_test_item"]]
    with pytest.raises(ValueError):
        loader_instance.split_data()



def test_prepare_for_training_success(loader_instance):
    img_size = loader_instance.size
    fake_img = np.full((img_size, img_size), 255, dtype=np.uint8)
    fake_dataset = [[fake_img, 1], [fake_img, 0]]
    expected_x = tc.ones((2, 1, img_size, img_size), dtype=tc.float32)
    expected_y = tc.tensor([1, 0], dtype=tc.float32)
    X, y = loader_instance.prepare_for_training(fake_dataset)
    assert X is not None
    assert y is not None

    assert tc.equal(X, expected_x)
    assert tc.equal(y, expected_y)


def test_prepare_for_training_exception(loader_instance):
    faulty_dataset = [["to_nie_jest_tablica_zdjecia", 1]]
    result = loader_instance.prepare_for_training(faulty_dataset)
    assert result is None or result == (None, None)