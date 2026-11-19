from unittest.mock import patch
import pandas as pd
import cv2
import numpy as np
import pytest
import torch as tc

from DataLoader.Loader import DataLoader
from config.config import META_DIR, IMG_SIZE, CHECXPERT_FILE, SPLIT_FILE, META_FILE

N = 40


@pytest.fixture
def loader_instance():
    with patch("DataLoader.Loader.kagglehub.dataset_download", return_value="C:/fake/path"):
        return DataLoader()
@pytest.fixture
def meta_dir(tmp_path):
    split = pd.DataFrame({
        "dicom_id":   ["d1", "d2", "d3", "d4", "d5"],
        "study_id":   [1, 1, 2, 3, 4],
        "subject_id": [10000001, 10000001, 10000002, 10000003, 10000004],
        "split":      ["train", "train", "test", "validate", "validate"],
    })
    meta = pd.DataFrame({
        "dicom_id":     ["d1", "d2", "d3", "d4", "d5"],
        "ViewPosition": ["PA", "LATERAL", "AP", "LATERAL", "PA"],
    })
    chex = pd.DataFrame({
        "subject_id": [10000001, 10000002, 10000003, 10000004],
        "study_id":   [1, 2, 3, 4],
        "Pneumonia":  [1.0, 0.0, -1.0, 0.0],
        "Fracture":   [0.0, 0.0, 0.0, 0.0],
        "Cardiomegaly": [0.0, 0.0, 0.0, 0.0],
        "Enlarged Cardiomediastinum": [0.0, 0.0, 0.0, 0.0],
    })
    split.to_csv(tmp_path / SPLIT_FILE, index=False, compression="gzip")
    meta.to_csv(tmp_path / META_FILE, index=False, compression="gzip")
    chex.to_csv(tmp_path / CHECXPERT_FILE, index=False, compression="gzip")
    return str(tmp_path)

@pytest.fixture
def loader(meta_dir):
    loader = DataLoader()
    loader.meta_dir = meta_dir
    return loader


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


def test_build_dataset_keeps_only_frontal_views(loader):
    df = loader.build_dataset()
    assert len(df) == 3
    assert set(df["split"]) == {"train", "test", "validate"}
    assert "Fracture" not in df.columns


def test_build_dataset_builds_path(loader):
    df = loader.build_dataset()
    path = df.loc[df["subject_id"] == 10000001, "path"].iloc[0]
    assert path == "files/p10/p10000001/s1/d1.jpg"


def test_build_dataset_labels(loader):
    df = loader.build_dataset()
    labels = dict(zip(df["subject_id"], df["Pneumonia"]))
    assert labels[10000001] == 1
    assert labels[10000002] == 0


def test_no_patient_in_two_splits(loader):
    df = loader.build_dataset()
    assert df.groupby("subject_id")["split"].nunique().max() == 1


def test_missing_dir_raises(loader):
    loader.meta_dir = "nie/ma/takiego/folderu"
    with pytest.raises(FileNotFoundError):
        loader.build_dataset()


@patch("DataLoader.Loader.os.listdir", return_value=["a.jpg", "b.jpg", "c.jpg"])
@patch("DataLoader.Loader.CATEGORIES", {"NORMAL": 1, "PNEUMONIA": 0})
def test_count_training_images_per_class(mock_listdir, loader_instance):
    result = loader_instance.count_training_images_per_class()
    assert result == {"NORMAL": 3, "PNEUMONIA": 3}

def test_split_data_sizes(loader):
    train, test, val = loader.split_data()
    assert (len(train), len(test), len(val)) == (1, 1, 1)


def test_split_data_rows_belong_to_right_set(loader):
    train, val, test = loader.split_data()
    assert (train["split"] == "train").all()
    assert (test["split"] == "test").all()
    assert (val["split"] == "validate").all()


def test_split_data_covers_whole_dataset(loader):
    dataset = loader.build_dataset()
    train, val, test = loader.split_data()
    assert len(train) + len(test) + len(val) == len(dataset)


def test_split_data_no_patient_overlap(loader):
    train, val, test = loader.split_data()
    s_train, s_val, s_test = (set(t["subject_id"]) for t in (train, test, val))
    assert s_train.isdisjoint(s_test)
    assert s_train.isdisjoint(s_val)
    assert s_test.isdisjoint(s_val)


def test_split_data_is_reproducible(loader):
    first = loader.split_data()
    second = loader.split_data()
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)



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