from unittest.mock import patch, MagicMock
import torch as tc
import pytest
from torch.utils.data import TensorDataset, DataLoader
from agent.XGBoostAgent import XGBoostAgent
import xgboost as xgb
import numpy as np
from agent.XrayAgent import XrayAgent

@pytest.fixture
def xgboost_init():
    xgb_agent= XGBoostAgent(XrayAgent())
    xgb_agent.agent=XrayAgent()
    xgb_agent.classifier=xgb.XGBClassifier()
    xgb_agent.batch_size=32
    return xgb_agent

def test_xgboost_initialization():
    agent=XGBoostAgent(XrayAgent())
    assert agent is not None
    assert hasattr(agent, 'batch_size')
    assert hasattr(agent, 'classifier')
    assert hasattr(agent, 'agent')

