from unittest.mock import MagicMock
from server import api
from fastapi.testclient import TestClient
from server.server import PredictionResponse

client = TestClient(api.app)


def test_predict_endpoint_success():
    api.server.agent = MagicMock()
    api.server.classifier = MagicMock()
    api.server.predict = MagicMock(return_value=[PredictionResponse(is_healthy=True, confidence=0.9)])

    response = client.post("/make_predict", files={"files": ("test.jpg", b"fake_bytes", "image/jpeg")})

    assert response.status_code == 200
    assert response.json()[0]["is_healthy"] is True