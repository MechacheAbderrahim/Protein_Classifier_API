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
