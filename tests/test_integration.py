import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run the real model inference test.",
)
def test_predict_with_real_pointnet_model():
    """Test the full inference path: API -> main.py -> EDTSurf -> PointNet."""
    pdb_path = Path("data/1a03_A_5.pdb")

    assert pdb_path.exists()

    with pdb_path.open("rb") as pdb_file:
        response = client.post(
            "/predict",
            data={"model_name": "PointNet"},
            files={"file": (pdb_path.name, pdb_file, "chemical/x-pdb")},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "PointNet"
    assert isinstance(body["prediction"], int)
    assert 0 <= body["prediction"] < 99


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run the real model inference test.",
)
def test_predict_with_real_pointnet_precomputed_surface():
    """Test the Docker-friendly path: API -> existing .ply surface -> PointNet."""
    ply_path = Path("data/surfaces/1a03_A_5.ply")

    assert ply_path.exists()

    with ply_path.open("rb") as ply_file:
        response = client.post(
            "/predict",
            data={"model_name": "PointNet"},
            files={"file": (ply_path.name, ply_file, "application/octet-stream")},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "PointNet"
    assert isinstance(body["prediction"], int)
    assert 0 <= body["prediction"] < 99
