from pathlib import Path

import torch
from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_unsupported_file():
    response = client.post(
        "/predict",
        data={"model_name": "PointNet"},
        files={"file": ("protein.txt", b"ATOM data", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Input file must be a .pdb, .ply, or .off file"


def test_predict_rejects_empty_pdb_file():
    response = client.post(
        "/predict",
        data={"model_name": "PointNet"},
        files={"file": ("protein.pdb", b"", "chemical/x-pdb")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Input file is empty"


def test_predict_returns_prediction_with_mocked_model(monkeypatch):
    def fake_run_prediction(model_name, data_path):
        data_path = Path(data_path)

        assert model_name == "PointNet"
        assert data_path.suffix == ".ply"
        assert data_path.exists()
        assert data_path.read_bytes() == b"ATOM data"

        return torch.tensor([[0.1, 0.2, 0.7]]), None

    monkeypatch.setattr(api, "run_prediction", fake_run_prediction)

    response = client.post(
        "/predict",
        data={"model_name": "PointNet"},
        files={"file": ("protein.ply", b"ATOM data", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "PointNet",
        "prediction": 2,
    }
