import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


SUPPORTED_INPUT_SUFFIXES = {".pdb", ".ply", ".off"}

app = FastAPI(title="Protein Surface Classifier API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("protein_api")
logger.setLevel(logging.INFO)


def run_prediction(model_name, data_path):
    from main import main

    return main(model_name=model_name, data_path=data_path)


@app.get("/health")
def health():
    logger.info("health_check status=ok")
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...),
):
    started_at = perf_counter()

    if not file.filename:
        logger.warning(
            "predict_rejected model_name=%s reason=missing_file_name",
            model_name,
        )
        raise HTTPException(status_code=400, detail="Input file is required")

    file_name = Path(file.filename).name
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        logger.warning(
            "predict_rejected model_name=%s file_name=%s file_type=%s reason=unsupported_file_type",
            model_name,
            file_name,
            suffix or "none",
        )
        raise HTTPException(
            status_code=400,
            detail="Input file must be a .pdb, .ply, or .off file",
        )

    with TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / file_name
        contents = await file.read()
        file_size = len(contents)

        if not contents:
            logger.warning(
                "predict_rejected model_name=%s file_name=%s file_type=%s file_size_bytes=0 reason=empty_file",
                model_name,
                file_name,
                suffix,
            )
            raise HTTPException(status_code=400, detail="Input file is empty")

        input_path.write_bytes(contents)

        logger.info(
            "predict_started model_name=%s file_name=%s file_type=%s file_size_bytes=%s",
            model_name,
            file_name,
            suffix,
            file_size,
        )

        try:
            outputs, _ = run_prediction(
                model_name=model_name,
                data_path=input_path,
            )
        except FileNotFoundError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.warning(
                "predict_failed model_name=%s file_name=%s file_type=%s elapsed_ms=%s error_type=file_not_found error=%s",
                model_name,
                file_name,
                suffix,
                elapsed_ms,
                exc,
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.warning(
                "predict_failed model_name=%s file_name=%s file_type=%s elapsed_ms=%s error_type=value_error error=%s",
                model_name,
                file_name,
                suffix,
                elapsed_ms,
                exc,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "predict_failed model_name=%s file_name=%s file_type=%s elapsed_ms=%s error_type=runtime_error",
                model_name,
                file_name,
                suffix,
                elapsed_ms,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = outputs.argmax(dim=1).detach().cpu().tolist()
    prediction = predictions[0]
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    logger.info(
        "predict_success model_name=%s file_name=%s file_type=%s file_size_bytes=%s prediction=%s elapsed_ms=%s",
        model_name,
        file_name,
        suffix,
        file_size,
        prediction,
        elapsed_ms,
    )

    return {
        "model_name": model_name,
        "prediction": prediction,
    }
