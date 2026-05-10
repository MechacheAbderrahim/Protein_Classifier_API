from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


SUPPORTED_INPUT_SUFFIXES = {".pdb", ".ply", ".off"}

app = FastAPI(title="Protein Surface Classifier API")


def run_prediction(model_name, data_path):
    from main import main

    return main(model_name=model_name, data_path=data_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Input file is required")

    suffix = Path(file.filename).suffix.lower()

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Input file must be a .pdb, .ply, or .off file",
        )

    with TemporaryDirectory() as tmp_dir:
        pdb_path = Path(tmp_dir) / Path(file.filename).name
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Input file is empty")

        pdb_path.write_bytes(contents)

        try:
            outputs, _ = run_prediction(
                model_name=model_name,
                data_path=pdb_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = outputs.argmax(dim=1).detach().cpu().tolist()

    return {
        "model_name": model_name,
        "prediction": predictions[0],
    }
