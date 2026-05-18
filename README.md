# Protein Surface Classifier API

FastAPI service for protein surface classification from `.pdb`, `.ply`, or `.off` files.

The project started as a local `main.py` inference script and was transformed into a small MLOps-ready API with tests and Docker packaging.

## What It Does

- accepts protein input files through an HTTP API
- supports `.pdb`, `.ply`, and `.off`
- converts `.pdb` to `.ply` with EDTSurf
- uses precomputed `.ply` / `.off` surfaces directly
- returns the predicted class as JSON

Recommended demo path:

```text
precomputed .ply
-> FastAPI
-> PointNet
-> JSON prediction
```

This path avoids EDTSurf and Open3D during inference.

## Project Structure

```text
MLOPs/
  api.py
  main.py
  Dockerfile
  requirements.txt
  requirements-dev.txt
  pytest.ini

  models/
    best_models/

  utils/
    EDTSurf
    EDTSurf_linux

  tests/
    test_api.py
    test_integration.py
    test_main_inputs.py
```

## Local Setup

Use Python 3.11.

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pip install -r requirements-dev.txt
```

Run the API locally:

```bash
python3.11 -m uvicorn api:app --reload
```

If model weights are not stored locally, set the remote model storage URL:

```bash
export MODEL_BASE_URL="https://<storage-account>.blob.core.windows.net/<container>"
```

The API first looks for:

```text
models/best_models/<model_name>.pt
```

If the file is missing, it downloads:

```text
$MODEL_BASE_URL/<model_name>.pt
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API Usage

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Prediction with a precomputed surface:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "model_name=PointNet" \
  -F "file=@data/surfaces/1a03_A_5.ply"
```

Example response:

```json
{"model_name":"PointNet","prediction":43}
```

Prediction from a PDB file:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "model_name=PointNet" \
  -F "file=@data/1a03_A_5.pdb"
```

## Tests

Run fast tests:

```bash
python3.11 -m pytest -q
```

Run the real integration tests:

```bash
RUN_INTEGRATION_TESTS=1 python3.11 -m pytest tests/test_integration.py -q
```

Test types:

```text
test_api.py          -> FastAPI behavior with a mocked model
test_main_inputs.py  -> input routing rules
test_integration.py  -> real model inference
```

## Docker

Build the image:

```bash
docker buildx build --platform linux/amd64 --load -t protein-api:latest .
```

Run the container:

```bash
docker run --platform linux/amd64 --name protein-api-container -p 8000:8000 protein-api:latest
```

Restart the same container later:

```bash
docker start -ai protein-api-container
```

Remove it if needed:

```bash
docker rm -f protein-api-container
```

Why `linux/amd64`?

```text
utils/EDTSurf_linux is an x86-64 Linux binary.
```

## MLOps Status

Current progress:

```text
~70%
```

Completed:

- working inference pipeline
- FastAPI service
- input validation
- automated tests
- integration tests
- Docker image
- Dockerized `/health`
- Dockerized `/predict` with `.ply` and `.pdb`

Next:

- GitHub repository
- GitHub Actions CI
- container registry push
- public deployment with Azure Container Apps
- logging and basic monitoring
- model versioning metadata
