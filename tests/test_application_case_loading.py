from __future__ import annotations

from pathlib import Path

from echoquality.application import load_case_from_manifest
from echoquality.data.synthetic import create_demo_dataset


def test_manifest_case_loader_returns_ed_and_es(tmp_path: Path) -> None:
    manifest = create_demo_dataset(tmp_path / "demo", n_patients=2, seed=77, size=64)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    first = manifest.iloc[0]
    case = load_case_from_manifest(
        manifest_path,
        patient_id=str(first["patient_id"]),
        view=str(first["view"]),
    )
    assert set(case.frames) == {"ED", "ES"}
    assert case.frames["ED"].image.shape == case.frames["ED"].mask.shape
    assert case.frames["ES"].image.shape == case.frames["ES"].mask.shape
    assert case.source == "manifest"
