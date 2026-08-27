import json

import pytest

from src.scientific_protocol.project_store import (
    duplicate_cell,
    load_project,
    new_project,
    prepare_for_save,
    save_project,
    validate_project,
)


def test_new_project_has_schema_20_and_cell():
    project = new_project("Estudo Nb2L", "Nb2L", "NaCl 1M")
    assert project["schema_version"] == "2.0"
    assert list(project["electrolytes"]["NaCl 1M"]["cells"]) == ["cell_001"]


def test_duplicate_cell_copies_settings_but_clears_files():
    project = new_project("P", "Nb2", "NaCl")
    cell = project["electrolytes"]["NaCl"]["cells"]["cell_001"]
    cell.update({"mass_g": 0.0056, "cv": {"0.005": "cv.txt"}, "gcd": "gcd.txt"})
    new_id = duplicate_cell(project, "NaCl", "cell_001")
    copied = project["electrolytes"]["NaCl"]["cells"][new_id]
    assert copied["replicate"] == 2
    assert copied["mass_g"] == pytest.approx(0.0056)
    assert copied["cv"] == {}
    assert copied["gcd"] == ""


def test_validation_rejects_invalid_electrode_fraction(tmp_path):
    project = new_project("P", "Nb2", "NaCl")
    cell = project["electrolytes"]["NaCl"]["cells"]["cell_001"]
    cell.update(
        {
            "mass_g": 0.005,
            "cell_area_cm2": 1.0,
            "electrodes": {
                "WE": {"mass_g": 0.0025, "potential_fraction": 0.7},
                "CE": {"mass_g": 0.0025, "potential_fraction": 0.7},
            },
        }
    )
    assert any("frações" in message for message in validate_project(project))


def test_save_and_load_preserve_relative_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cv = data_dir / "cv.txt"
    cv.write_text("sample", encoding="utf-8")
    project = new_project("P", "Nb2", "NaCl")
    cell = project["electrolytes"]["NaCl"]["cells"]["cell_001"]
    cell.update(
        {
            "mass_g": 0.005,
            "cell_area_cm2": 1.0,
            "cv": {"0.005": str(cv)},
        }
    )
    path = save_project(
        project, tmp_path / "scientific_protocol_projects" / "p" / "project.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert (
        raw["electrolytes"]["NaCl"]["cells"]["cell_001"]["cv"]["0.005"]
        == "../../data/cv.txt"
    )
    loaded = load_project(path)
    assert loaded["electrolytes"]["NaCl"]["cells"]["cell_001"]["cv"]["0.005"] == str(
        cv.resolve()
    )


def test_prepare_for_save_adds_reproducibility_metadata(tmp_path):
    project = new_project("P", "Nb2", "NaCl")
    saved = prepare_for_save(project, tmp_path)
    assert saved["metadata"]["schema_version"] == "2.0"
    assert saved["metadata"]["ionflow_version"]


def test_load_legacy_schema_migrates_to_schema_20(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"electrolytes": {"NaCl": {"mass_g": 0.005}}}), encoding="utf-8"
    )
    loaded = load_project(path)
    cell = loaded["electrolytes"]["NaCl"]["cells"]["cell_001"]
    assert loaded["schema_version"] == "2.0"
    assert loaded["metadata"]["migrated_from_legacy"] is True
    assert cell["mass_g"] == pytest.approx(0.005)


def test_load_schema_20_preserves_multiple_cells(tmp_path):
    project = new_project("P", "Nb2", "NaCl")
    project["electrolytes"]["NaCl"]["cells"]["cell_002"] = {
        "replicate": 2,
        "cell_id": "cell_002",
        "cv": {},
        "gcd": "",
        "mass_g": 0.005,
        "cell_area_cm2": 1.0,
        "current_sequence_a_g": [],
        "electrodes": {},
        "eis": {},
    }
    path = tmp_path / "project.json"
    save_project(project, path)
    loaded = load_project(path)
    assert list(loaded["electrolytes"]["NaCl"]["cells"]) == ["cell_001", "cell_002"]
