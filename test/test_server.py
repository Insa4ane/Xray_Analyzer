import tempfile
from PIL import Image
from DataLoader.Loader import DataLoader
from agent.XGBoostAgent import XGBoostAgent
from agent.XrayAgent import XrayAgent
from server.server import Server, PredictionResponse
import pytest
from unittest.mock import patch
import os
import io
import torch as tc
import numpy as np

@pytest.fixture
def dummy_server():
    server = Server()
    server.agent=XrayAgent()
    server.classifier=XGBoostAgent(server.agent)
    return server

@patch.object(XGBoostAgent, 'load_model', return_value=True)
@patch.object(XrayAgent, 'load_model', return_value=True)
def test_get_agents_success(mock_xray, mock_xgb):
    agent, classifier = Server._get_agents()
    assert agent is not None
    assert isinstance(agent, XrayAgent)
    assert classifier is not None
    assert isinstance(classifier, XGBoostAgent)
    mock_xray.assert_called_once()
    mock_xgb.assert_called_once()


@patch.object(XrayAgent, 'load_model', return_value=False)
def test_get_agents_no_cnn(mock_xray):
    agent, classifier = Server._get_agents()
    assert agent is None
    assert classifier is None


@patch.object(XGBoostAgent, 'load_model', return_value=False)
@patch.object(XrayAgent, 'load_model', return_value=True)
def test_get_agents_xgb(mock_xray, mock_xgb):
    agent, classifier = Server._get_agents()
    assert agent is None
    assert classifier is None


@patch.object(XrayAgent, '__init__', side_effect=Exception("Błąd konfiguracji"))
def test_get_agents_exception(mock_init):
    agent, classifier = Server._get_agents()
    assert agent is None
    assert classifier is None


def test_get_directories_success(dummy_server):
    width, height = 224, 224
    fake_images_bytes=[]
    with tempfile.TemporaryDirectory() as tmp_dir:
        for fake_bytes in range(10):
            fake_bytes=os.urandom(width*height*1)
            image = Image.frombytes("L", (width, height), fake_bytes)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            fake_images_bytes.append(buffer.getvalue())
        results=dummy_server._get_directories(fake_images_bytes, tmp_dir)
        assert len(results)==len(fake_images_bytes)
        for result in results:
            assert  os.path.exists(result)

def test_get_directories_exception(dummy_server):
    pass

@patch.object(XGBoostAgent, 'predict', return_value=True)
@patch.object(DataLoader, 'process_image')
@patch.object(Server, '_get_directories')
def test_predict_success(mock_directory,mock_proccess,mock_predict, dummy_server):
    mock_directory.return_value=[f"tmp/hehe/{i}" for i in range(10)]
    mock_proccess.return_value=tc.rand(10,1, 224, 224)
    mock_predict.return_value=np.array([1,0,1,0,1,0,1,0,1,1])
    width, height = 224, 224
    fake_images_bytes=[]
    for fake_bytes in range(10):
        fake_bytes = os.urandom(width * height * 1)
        image = Image.frombytes("L", (width, height), fake_bytes)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        fake_images_bytes.append(buffer.getvalue())
    result = dummy_server.predict(fake_images_bytes)
    assert result is not None
    assert isinstance(result, list) and all(isinstance(item, PredictionResponse) for item in result)

def test_predict_exception(dummy_server):
    pass













