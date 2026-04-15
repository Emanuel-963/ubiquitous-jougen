"""Tests for the GUI MVC layer (Day 13).

All tests are headless — no tkinter / customtkinter required.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.gui.models import (
    AppState,
    DRT_DEFAULT_PRESET,
    DRT_PRESETS,
    PlotItem,
)
from src.gui.controller import PipelineController
from src.gui.main_window import MainWindow


# ═══════════════════════════════════════════════════════════════════════
#  PlotItem
# ═══════════════════════════════════════════════════════════════════════


class TestPlotItem:
    def test_creation(self):
        p = PlotItem(title="Nyquist", path="/tmp/nyq.png")
        assert p.title == "Nyquist"
        assert p.path == "/tmp/nyq.png"

    def test_equality(self):
        a = PlotItem("A", "/a.png")
        b = PlotItem("A", "/a.png")
        assert a == b

    def test_inequality(self):
        a = PlotItem("A", "/a.png")
        b = PlotItem("B", "/b.png")
        assert a != b


# ═══════════════════════════════════════════════════════════════════════
#  DRT presets
# ═══════════════════════════════════════════════════════════════════════


class TestDRTPresets:
    def test_preset_keys(self):
        assert set(DRT_PRESETS.keys()) == {
            "Rápido",
            "Balanceado",
            "Alta resolução",
        }

    def test_default_preset_exists(self):
        assert DRT_DEFAULT_PRESET in DRT_PRESETS

    def test_each_preset_has_required_keys(self):
        for name, values in DRT_PRESETS.items():
            assert "lambda_reg" in values, f"{name} missing lambda_reg"
            assert "n_taus" in values, f"{name} missing n_taus"


# ═══════════════════════════════════════════════════════════════════════
#  AppState
# ═══════════════════════════════════════════════════════════════════════


class TestAppStateDefaults:
    def test_all_dataframes_none(self):
        s = AppState()
        for attr in (
            "eis_df", "rank_df", "df_pca", "circuit_df",
            "cic_df",
            "drt_df", "drt_peaks_df", "drt_summary_df", "drt_eis_df",
        ):
            assert getattr(s, attr) is None

    def test_all_dicts_empty(self):
        s = AppState()
        for attr in (
            "raw_eis", "cic_results", "cic_plot_map",
            "drt_results", "drt_plot_map", "gui_settings",
        ):
            assert getattr(s, attr) == {}

    def test_plot_items_empty(self):
        s = AppState()
        assert s.plot_items == []

    def test_default_status(self):
        s = AppState()
        assert s.status == "pronto"
        assert s.progress_text == "Pronto"
        assert s.is_running is False

    def test_drt_ui_prefs_default(self):
        s = AppState()
        assert s.drt_ui_prefs == {
            "sample": "",
            "mode": "Espectro",
            "overlay_text": "",
        }

    def test_independent_instances(self):
        """Two AppState instances must not share mutable defaults."""
        a = AppState()
        b = AppState()
        a.raw_eis["x"] = pd.DataFrame()
        assert "x" not in b.raw_eis

    def test_independent_plot_items(self):
        a = AppState()
        b = AppState()
        a.plot_items.append(PlotItem("X", "/x"))
        assert len(b.plot_items) == 0


class TestAppStateQueries:
    def test_has_eis_data_false(self):
        assert AppState().has_eis_data() is False

    def test_has_eis_data_true(self):
        s = AppState(eis_df=pd.DataFrame({"a": [1]}))
        assert s.has_eis_data() is True

    def test_has_cycling_data_false(self):
        assert AppState().has_cycling_data() is False

    def test_has_cycling_data_true(self):
        s = AppState(cic_df=pd.DataFrame({"b": [2]}))
        assert s.has_cycling_data() is True

    def test_has_drt_data_false(self):
        assert AppState().has_drt_data() is False

    def test_has_drt_data_true(self):
        s = AppState(drt_df=pd.DataFrame({"c": [3]}))
        assert s.has_drt_data() is True


class TestAppStateClear:
    def test_clear_eis(self):
        s = AppState(
            eis_df=pd.DataFrame(), rank_df=pd.DataFrame(),
            df_pca=pd.DataFrame(), circuit_df=pd.DataFrame(),
            raw_eis={"k": pd.DataFrame()},
        )
        s.clear_eis()
        assert s.eis_df is None
        assert s.rank_df is None
        assert s.df_pca is None
        assert s.circuit_df is None
        assert s.raw_eis == {}

    def test_clear_cycling(self):
        s = AppState(
            cic_df=pd.DataFrame(),
            cic_results={"k": pd.DataFrame()},
            cic_plot_map={"k": "/p"},
        )
        s.clear_cycling()
        assert s.cic_df is None
        assert s.cic_results == {}
        assert s.cic_plot_map == {}

    def test_clear_drt(self):
        s = AppState(
            drt_df=pd.DataFrame(), drt_peaks_df=pd.DataFrame(),
            drt_summary_df=pd.DataFrame(), drt_eis_df=pd.DataFrame(),
            drt_results={"k": {}}, drt_plot_map={"k": "/p"},
        )
        s.clear_drt()
        assert s.drt_df is None
        assert s.drt_peaks_df is None
        assert s.drt_summary_df is None
        assert s.drt_eis_df is None
        assert s.drt_results == {}
        assert s.drt_plot_map == {}

    def test_clear_plots(self):
        s = AppState()
        s.plot_items.extend([PlotItem("A", "/a"), PlotItem("B", "/b")])
        s.clear_plots()
        assert s.plot_items == []

    def test_clear_all(self):
        s = AppState(
            eis_df=pd.DataFrame(), cic_df=pd.DataFrame(),
            drt_df=pd.DataFrame(),
        )
        s.plot_items.append(PlotItem("X", "/x"))
        s.clear_all()
        assert s.eis_df is None
        assert s.cic_df is None
        assert s.drt_df is None
        assert s.plot_items == []


class TestAppStateSummary:
    def test_to_summary_empty(self):
        summary = AppState().to_summary()
        assert summary == {
            "has_eis": False,
            "has_cycling": False,
            "has_drt": False,
            "n_plots": 0,
            "n_raw_eis": 0,
            "n_cic_results": 0,
            "n_drt_results": 0,
            "status": "pronto",
            "is_running": False,
        }

    def test_to_summary_with_data(self):
        s = AppState(
            eis_df=pd.DataFrame(),
            cic_df=pd.DataFrame(),
            raw_eis={"a": pd.DataFrame(), "b": pd.DataFrame()},
            drt_results={"x": {}},
            status="rodando",
            is_running=True,
        )
        s.plot_items.append(PlotItem("P", "/p"))
        summary = s.to_summary()
        assert summary["has_eis"] is True
        assert summary["has_cycling"] is True
        assert summary["has_drt"] is False
        assert summary["n_plots"] == 1
        assert summary["n_raw_eis"] == 2
        assert summary["n_drt_results"] == 1
        assert summary["is_running"] is True

    def test_to_summary_is_serialisable(self):
        summary = AppState().to_summary()
        serialised = json.dumps(summary)
        assert isinstance(serialised, str)


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — Event Bus
# ═══════════════════════════════════════════════════════════════════════


class TestControllerEventBus:
    def test_on_and_emit(self):
        ctrl = PipelineController()
        received: List[str] = []
        ctrl.on("ping", lambda msg: received.append(msg))
        ctrl.emit("ping", "hello")
        assert received == ["hello"]

    def test_multiple_listeners(self):
        ctrl = PipelineController()
        a: List[int] = []
        b: List[int] = []
        ctrl.on("tick", lambda v: a.append(v))
        ctrl.on("tick", lambda v: b.append(v))
        ctrl.emit("tick", 42)
        assert a == [42]
        assert b == [42]

    def test_off_removes_listener(self):
        ctrl = PipelineController()
        received: List[str] = []
        cb = lambda msg: received.append(msg)
        ctrl.on("evt", cb)
        ctrl.off("evt", cb)
        ctrl.emit("evt", "nope")
        assert received == []

    def test_off_nonexistent_is_noop(self):
        ctrl = PipelineController()
        ctrl.off("nonexistent", lambda: None)  # should not raise

    def test_emit_unknown_event_is_noop(self):
        ctrl = PipelineController()
        ctrl.emit("unknown", 1, 2, 3)  # should not raise

    def test_listener_count(self):
        ctrl = PipelineController()
        assert ctrl.listener_count("x") == 0
        ctrl.on("x", lambda: None)
        assert ctrl.listener_count("x") == 1
        ctrl.on("x", lambda: None)
        assert ctrl.listener_count("x") == 2

    def test_emit_with_kwargs(self):
        ctrl = PipelineController()
        received: List[dict] = []
        ctrl.on("kw", lambda **kw: received.append(kw))
        ctrl.emit("kw", key="value")
        assert received == [{"key": "value"}]

    def test_default_state_created_when_none(self):
        ctrl = PipelineController()
        assert isinstance(ctrl.state, AppState)

    def test_explicit_state_used(self):
        state = AppState(status="custom")
        ctrl = PipelineController(state)
        assert ctrl.state.status == "custom"


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — EIS processing
# ═══════════════════════════════════════════════════════════════════════


class TestProcessEIS:
    def _make_eis_result(self) -> dict:
        cap_df = pd.DataFrame(
            {"C": [1.0, 2.0], "E": [3.0, 4.0]},
            index=["sample_a", "sample_b"],
        )
        return {
            "cap_energy": cap_df,
            "df_ranked": pd.DataFrame({"rank": [1, 2]}),
            "df_pca": pd.DataFrame({"pc1": [0.1]}),
            "raw_eis": {"a": pd.DataFrame()},
            "circuit_table": pd.DataFrame({"circuit": ["R0"]}),
            "pca_paths": ["/out/pca_scores.png", "/out/graph.png"],
        }

    def test_success(self):
        ctrl = PipelineController()
        ok = ctrl.process_eis_result(self._make_eis_result())
        assert ok is True
        assert ctrl.state.has_eis_data()
        assert ctrl.state.status == "EIS concluído"
        assert ctrl.state.is_running is False

    def test_eis_df_has_arquivo_column(self):
        ctrl = PipelineController()
        ctrl.process_eis_result(self._make_eis_result())
        assert "Arquivo" in ctrl.state.eis_df.columns

    def test_plots_added(self):
        ctrl = PipelineController()
        ctrl.process_eis_result(self._make_eis_result())
        assert len(ctrl.state.plot_items) == 2
        assert ctrl.state.plot_items[0].title == "PCA"

    def test_none_result(self):
        ctrl = PipelineController()
        ok = ctrl.process_eis_result(None)
        assert ok is False
        assert ctrl.state.status == "erro no EIS"
        assert ctrl.state.is_running is False

    def test_no_cap_energy(self):
        ctrl = PipelineController()
        ok = ctrl.process_eis_result({"cap_energy": None})
        assert ok is True
        assert ctrl.state.eis_df is None

    def test_emits_events(self):
        ctrl = PipelineController()
        events: List[str] = []
        ctrl.on("eis_completed", lambda s: events.append(f"eis:{s}"))
        ctrl.on("table_updated", lambda k: events.append(f"table:{k}"))
        ctrl.on("status_changed", lambda s: events.append(f"status:{s}"))
        ctrl.process_eis_result(self._make_eis_result())
        assert "eis:True" in events
        assert "table:eis" in events
        assert "table:circuit" in events
        assert "status:EIS concluído" in events


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — Cycling processing
# ═══════════════════════════════════════════════════════════════════════


class TestProcessCycling:
    def _make_cic_result(self, tmp_path) -> dict:
        # Create a dummy plot file so os.path.abspath works
        plot_file = tmp_path / "plot.png"
        plot_file.write_text("img")
        return {
            "merged_table": pd.DataFrame({"cycle": [1, 2]}),
            "results": {"file1": pd.DataFrame()},
            "plot_paths": [("file1", str(plot_file))],
            "energy_power_paths": [("file1", str(plot_file))],
        }

    def test_success(self, tmp_path):
        ctrl = PipelineController()
        ok = ctrl.process_cycling_result(self._make_cic_result(tmp_path))
        assert ok is True
        assert ctrl.state.has_cycling_data()
        assert ctrl.state.status == "Ciclagem concluída"

    def test_plot_map_populated(self, tmp_path):
        ctrl = PipelineController()
        ctrl.process_cycling_result(self._make_cic_result(tmp_path))
        assert len(ctrl.state.cic_plot_map) == 1

    def test_plots_added(self, tmp_path):
        ctrl = PipelineController()
        ctrl.process_cycling_result(self._make_cic_result(tmp_path))
        assert len(ctrl.state.plot_items) == 2  # integral + energy×power

    def test_none_result(self):
        ctrl = PipelineController()
        ok = ctrl.process_cycling_result(None)
        assert ok is False
        assert ctrl.state.status == "erro na Ciclagem"

    def test_emits_log(self, tmp_path):
        ctrl = PipelineController()
        logs: List[str] = []
        ctrl.on("log", lambda msg: logs.append(msg))
        ctrl.process_cycling_result(self._make_cic_result(tmp_path))
        assert any("1 arquivo(s)" in m for m in logs)


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — Both processing
# ═══════════════════════════════════════════════════════════════════════


class TestProcessBoth:
    def _make_both_result(self, tmp_path):
        plot_file = tmp_path / "plot.png"
        plot_file.write_text("img")
        eis = {
            "cap_energy": pd.DataFrame({"C": [1]}, index=["s"]),
            "df_ranked": None,
            "df_pca": None,
            "raw_eis": {},
            "circuit_table": None,
            "pca_paths": [],
        }
        cic = {
            "merged_table": pd.DataFrame({"cycle": [1]}),
            "results": {"f": pd.DataFrame()},
            "plot_paths": [("f", str(plot_file))],
            "energy_power_paths": [],
        }
        return eis, cic

    def test_success(self, tmp_path):
        ctrl = PipelineController()
        ok = ctrl.process_both_result(self._make_both_result(tmp_path))
        assert ok is True
        assert ctrl.state.has_eis_data()
        assert ctrl.state.has_cycling_data()
        assert ctrl.state.status == "Ambos concluídos"

    def test_none_result(self):
        ctrl = PipelineController()
        ok = ctrl.process_both_result(None)
        assert ok is False
        assert ctrl.state.status == "erro ao rodar ambos"

    def test_emits_events(self, tmp_path):
        ctrl = PipelineController()
        events: List[str] = []
        ctrl.on("both_completed", lambda s: events.append(f"both:{s}"))
        ctrl.process_both_result(self._make_both_result(tmp_path))
        assert "both:True" in events


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — DRT processing
# ═══════════════════════════════════════════════════════════════════════


class TestProcessDRT:
    def _make_drt_result(self, tmp_path) -> dict:
        plot_file = tmp_path / "drt.png"
        plot_file.write_text("img")
        return {
            "drt_table": pd.DataFrame({"tau": [0.1]}),
            "drt_peaks_table": pd.DataFrame({"peak": [1]}),
            "drt_summary_table": pd.DataFrame({
                "Sample": ["s1", "s2"],
                "gamma_peak_main": [0.5, 0.3],
            }),
            "per_file_results": {"s1": {"gamma": [0.5]}},
            "plot_paths": [("s1", str(plot_file))],
            "errors": {},
            "run_meta": {
                "n_files": 2, "n_success": 2,
                "n_failed": 0, "lambda_reg": 1e-3, "n_taus": 50,
            },
        }

    def test_success(self, tmp_path):
        ctrl = PipelineController()
        ok = ctrl.process_drt_result(self._make_drt_result(tmp_path))
        assert ok is True
        assert ctrl.state.has_drt_data()
        assert ctrl.state.status == "DRT concluído"

    def test_none_result(self):
        ctrl = PipelineController()
        ok = ctrl.process_drt_result(None)
        assert ok is False
        assert ctrl.state.status == "erro no DRT"

    def test_plot_map_populated(self, tmp_path):
        ctrl = PipelineController()
        ctrl.process_drt_result(self._make_drt_result(tmp_path))
        assert len(ctrl.state.drt_plot_map) == 1

    def test_summary_logged(self, tmp_path):
        ctrl = PipelineController()
        logs: List[str] = []
        ctrl.on("log", lambda msg: logs.append(msg))
        ctrl.process_drt_result(self._make_drt_result(tmp_path))
        assert any("DRT resumo" in m for m in logs)
        assert any("s1" in m for m in logs)

    def test_errors_logged(self, tmp_path):
        result = self._make_drt_result(tmp_path)
        result["errors"] = {"bad_file": "parse error"}
        ctrl = PipelineController()
        logs: List[str] = []
        ctrl.on("log", lambda msg: logs.append(msg))
        ctrl.process_drt_result(result)
        assert any("1 falha(s)" in m for m in logs)

    def test_meta_logged(self, tmp_path):
        ctrl = PipelineController()
        logs: List[str] = []
        ctrl.on("log", lambda msg: logs.append(msg))
        ctrl.process_drt_result(self._make_drt_result(tmp_path))
        assert any("DRT meta" in m for m in logs)

    def test_emits_events(self, tmp_path):
        ctrl = PipelineController()
        events: List[str] = []
        ctrl.on("drt_completed", lambda s: events.append(f"drt:{s}"))
        ctrl.on("table_updated", lambda k: events.append(f"table:{k}"))
        ctrl.process_drt_result(self._make_drt_result(tmp_path))
        assert "drt:True" in events
        assert "table:drt" in events
        assert "table:drt_peaks" in events


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — start_pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestStartPipeline:
    def test_sets_running(self):
        ctrl = PipelineController()
        ctrl.state.plot_items.append(PlotItem("old", "/old"))
        ctrl.start_pipeline("EIS")
        assert ctrl.state.is_running is True
        assert ctrl.state.status == "rodando EIS"
        assert ctrl.state.plot_items == []

    def test_emits_events(self):
        ctrl = PipelineController()
        events: List[str] = []
        ctrl.on("buttons_disable", lambda: events.append("disable"))
        ctrl.on("status_changed", lambda s: events.append(f"status:{s}"))
        ctrl.on("progress_start", lambda t: events.append(f"start:{t}"))
        ctrl.start_pipeline("DRT")
        assert "disable" in events
        assert "status:rodando DRT" in events


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — Settings
# ═══════════════════════════════════════════════════════════════════════


class TestSettings:
    def test_load_missing_file(self, tmp_path):
        ctrl = PipelineController(
            settings_path=str(tmp_path / "nonexistent.json")
        )
        result = ctrl.load_settings()
        assert result == {}

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "settings.json")
        ctrl = PipelineController(settings_path=path)
        ctrl.state.gui_settings = {"theme": "dark", "lang": "pt"}
        assert ctrl.save_settings() is True

        ctrl2 = PipelineController(settings_path=path)
        loaded = ctrl2.load_settings()
        assert loaded == {"theme": "dark", "lang": "pt"}
        assert ctrl2.state.gui_settings == loaded

    def test_save_no_path(self):
        ctrl = PipelineController(settings_path="")
        assert ctrl.save_settings() is False

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("NOT JSON {{}", encoding="utf-8")
        ctrl = PipelineController(settings_path=str(path))
        result = ctrl.load_settings()
        assert result == {}

    def test_load_non_dict_json(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        ctrl = PipelineController(settings_path=str(path))
        result = ctrl.load_settings()
        assert result == {}

    def test_remember_dialog_dir(self, tmp_path):
        path = str(tmp_path / "settings.json")
        ctrl = PipelineController(settings_path=path)
        # Use tmp_path itself as the remembered directory
        ctrl.remember_dialog_dir("raw", str(tmp_path))
        assert ctrl.state.gui_settings[f"last_dir_raw"] == str(tmp_path)
        # Verify it was persisted
        with open(path, "r") as f:
            data = json.load(f)
        assert data[f"last_dir_raw"] == str(tmp_path)

    def test_remember_dialog_dir_empty(self):
        ctrl = PipelineController()
        ctrl.remember_dialog_dir("raw", "")  # should not raise

    def test_get_initial_dialog_dir_found(self, tmp_path):
        ctrl = PipelineController()
        ctrl.state.gui_settings["last_dir_raw"] = str(tmp_path)
        assert ctrl.get_initial_dialog_dir("raw") == str(tmp_path)

    def test_get_initial_dialog_dir_missing(self):
        ctrl = PipelineController()
        assert ctrl.get_initial_dialog_dir("raw") is None


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — DRT presets
# ═══════════════════════════════════════════════════════════════════════


class TestDRTPresets:
    def test_get_drt_preset_existing(self):
        preset = PipelineController.get_drt_preset("Balanceado")
        assert preset is not None
        assert preset["lambda_reg"] == "1e-3"
        assert preset["n_taus"] == "50"

    def test_get_drt_preset_nonexistent(self):
        assert PipelineController.get_drt_preset("Inexistente") is None

    def test_validate_drt_params_valid(self):
        lam, n = PipelineController.validate_drt_params("1e-3", "50")
        assert lam == 1e-3
        assert n == 50

    def test_validate_drt_params_lambda_zero(self):
        with pytest.raises(ValueError, match="λ DRT deve ser > 0"):
            PipelineController.validate_drt_params("0", "50")

    def test_validate_drt_params_lambda_negative(self):
        with pytest.raises(ValueError, match="λ DRT deve ser > 0"):
            PipelineController.validate_drt_params("-1e-3", "50")

    def test_validate_drt_params_n_taus_too_small(self):
        with pytest.raises(ValueError, match="n_taus deve ser >= 10"):
            PipelineController.validate_drt_params("1e-3", "5")

    def test_validate_drt_params_invalid_string(self):
        with pytest.raises(ValueError):
            PipelineController.validate_drt_params("abc", "50")

    def test_reset_drt_defaults(self):
        ctrl = PipelineController()
        ctrl.state.drt_ui_prefs = {
            "sample": "test",
            "mode": "Heatmap",
            "overlay_text": "abc",
        }
        preset = ctrl.reset_drt_defaults()
        assert preset == DRT_PRESETS[DRT_DEFAULT_PRESET]
        assert ctrl.state.drt_ui_prefs["sample"] == ""
        assert ctrl.state.drt_ui_prefs["mode"] == "Espectro"

    def test_reset_drt_defaults_emits_log(self):
        ctrl = PipelineController()
        logs: List[str] = []
        ctrl.on("log", lambda msg: logs.append(msg))
        ctrl.reset_drt_defaults()
        assert any("Balanceado" in m for m in logs)


# ═══════════════════════════════════════════════════════════════════════
#  PipelineController — Utility
# ═══════════════════════════════════════════════════════════════════════


class TestControllerUtility:
    def test_normalize_sample_name_basic(self):
        assert PipelineController.normalize_sample_name("  My Sample  ") == "my_sample"

    def test_normalize_sample_name_already_clean(self):
        assert PipelineController.normalize_sample_name("abc") == "abc"

    def test_normalize_sample_name_spaces(self):
        assert PipelineController.normalize_sample_name("a b c") == "a_b_c"


# ═══════════════════════════════════════════════════════════════════════
#  MainWindow
# ═══════════════════════════════════════════════════════════════════════


class TestMainWindow:
    def test_creates_state_and_controller(self):
        mw = MainWindow()
        assert isinstance(mw.state, AppState)
        assert isinstance(mw.controller, PipelineController)
        assert mw.controller.state is mw.state

    def test_settings_loaded_from_file(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text('{"theme": "dark"}', encoding="utf-8")
        mw = MainWindow(settings_path=str(path))
        assert mw.state.gui_settings.get("theme") == "dark"

    def test_on_delegates_to_controller(self):
        mw = MainWindow()
        received: List[str] = []
        mw.on("test_evt", lambda v: received.append(v))
        mw.controller.emit("test_evt", "hello")
        assert received == ["hello"]

    def test_off_delegates_to_controller(self):
        mw = MainWindow()
        received: List[str] = []
        cb = lambda v: received.append(v)
        mw.on("evt", cb)
        mw.off("evt", cb)
        mw.controller.emit("evt", "nope")
        assert received == []

    def test_is_running_property(self):
        mw = MainWindow()
        assert mw.is_running is False
        mw.state.is_running = True
        assert mw.is_running is True

    def test_status_property(self):
        mw = MainWindow()
        assert mw.status == "pronto"
        mw.state.status = "rodando"
        assert mw.status == "rodando"

    def test_missing_settings_file(self, tmp_path):
        mw = MainWindow(settings_path=str(tmp_path / "missing.json"))
        assert mw.state.gui_settings == {}
