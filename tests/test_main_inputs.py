from pathlib import Path

import main


def test_prepare_surface_uses_existing_ply_without_edtsurf(tmp_path, monkeypatch):
    surface_path = tmp_path / "surface.ply"
    surface_path.write_text("ply\n")

    def fail_if_edtsurf_is_called(*args, **kwargs):
        raise AssertionError("EDTSurf should not run for a precomputed .ply surface")

    monkeypatch.setattr(main, "generate_surface_from_single_pdb", fail_if_edtsurf_is_called)

    assert main.prepare_surface_input(surface_path) == surface_path


def test_prepare_surface_rejects_unsupported_extension(tmp_path):
    txt_path = tmp_path / "surface.txt"
    txt_path.write_text("not a supported mesh")

    try:
        main.prepare_surface_input(txt_path)
    except ValueError as exc:
        assert ".pdb" in str(exc)
        assert ".ply" in str(exc)
        assert ".off" in str(exc)
    else:
        raise AssertionError("Unsupported input extension should raise ValueError")


def test_existing_model_weights_are_used_without_download(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_path = model_dir / "PointNet.pt"
    model_dir.mkdir()
    model_path.write_bytes(b"local weights")

    monkeypatch.delenv("MODEL_BASE_URL", raising=False)

    assert main.ensure_model_weights("PointNet", model_dir=model_dir) == model_path


def test_missing_model_weights_are_downloaded_from_model_base_url(tmp_path, monkeypatch):
    source_dir = tmp_path / "blob"
    target_dir = tmp_path / "models"
    source_dir.mkdir()
    source_model_path = source_dir / "PointNet.pt"
    source_model_path.write_bytes(b"remote weights")

    monkeypatch.setenv("MODEL_BASE_URL", source_dir.as_uri())

    model_path = main.ensure_model_weights("PointNet", model_dir=target_dir)

    assert model_path == target_dir / "PointNet.pt"
    assert model_path.read_bytes() == b"remote weights"


def test_model_weights_url_supports_container_sas_token(monkeypatch):
    monkeypatch.setenv(
        "MODEL_BASE_URL",
        "https://account.blob.core.windows.net/models?sv=demo&sig=secret",
    )

    assert (
        main.get_model_weights_url("PointNet")
        == "https://account.blob.core.windows.net/models/PointNet.pt?sv=demo&sig=secret"
    )


def test_missing_model_weights_without_model_base_url_raises_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)

    try:
        main.ensure_model_weights("PointNet", model_dir=tmp_path / "models")
    except FileNotFoundError as exc:
        assert "MODEL_BASE_URL" in str(exc)
    else:
        raise AssertionError("Missing model weights should raise FileNotFoundError")
