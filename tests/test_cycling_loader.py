"""Tests for src.cycling_loader — loading cycling data files."""

import pytest

from src.cycling_loader import load_cycling_files


class TestLoadCyclingFiles:
    def test_empty_directory(self, tmp_path):
        result = load_cycling_files(tmp_path)
        assert result == {}

    def test_loads_valid_file(self, tmp_path):
        content = (
            "Time (s);WE(1).Current (A);WE(1).Potential (V);Cycle\n"
            "0,0;0,1;1,0;1\n"
            "1,0;0,1;1,2;1\n"
            "2,0;0,1;0,9;2\n"
        )
        f = tmp_path / "sample.txt"
        f.write_text(content)
        result = load_cycling_files(tmp_path)
        assert "sample" in result
        assert list(result["sample"].columns) == ["tempo", "corrente", "potencial", "ciclo"]

    def test_missing_columns_raises(self, tmp_path):
        content = "col_a;col_b\n1;2\n3;4\n"
        f = tmp_path / "bad.txt"
        f.write_text(content)
        with pytest.raises(ValueError, match="missing required columns"):
            load_cycling_files(tmp_path)
