"""Professional command-line interface for the IonFlow Pipeline.

Provides subcommands for each pipeline (EIS, Cycling, DRT), combined
analysis with AI, configuration management, and data validation.

Usage examples
--------------
::

    ionflow eis --data-dir data/raw --config config.json --output outputs/
    ionflow cycling --data-dir data/processed --scan-rate 0.1
    ionflow drt --data-dir data/raw --lambda 1e-3
    ionflow analyze --all --ai --export-pdf report.pdf
    ionflow config --init
    ionflow validate --data-dir data/raw

Return codes
------------
0 — success
1 — error (pipeline failure, missing data, etc.)
2 — warning (partial success, e.g. some files failed validation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import PipelineConfig
from src.logger import setup_logging

logger = logging.getLogger(__name__)

# ── Return codes ─────────────────────────────────────────────────────
RC_OK = 0
RC_ERROR = 1
RC_WARNING = 2


# ═══════════════════════════════════════════════════════════════════════
# Progress helper
# ═══════════════════════════════════════════════════════════════════════

class _ProgressReporter:
    """Thin wrapper for tqdm-style progress.  Falls back to print if tqdm
    is not installed.
    """

    def __init__(self, total: int, desc: str, *, use_json: bool = False):
        self._use_json = use_json
        self._desc = desc
        self._total = total
        self._bar: Any = None

        if use_json:
            # In JSON mode, no progress bar — just collect messages
            return

        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            self._bar = tqdm(total=total, desc=desc, unit="step", ncols=80)
        except ImportError:
            pass

    def update(self, n: int = 1, *, msg: str = "") -> None:
        if self._use_json:
            return
        if self._bar is not None:
            if msg:
                self._bar.set_postfix_str(msg)
            self._bar.update(n)
        else:
            sys.stderr.write(f"  [{self._desc}] {msg}\n")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


# ═══════════════════════════════════════════════════════════════════════
# Config loader helper
# ═══════════════════════════════════════════════════════════════════════

def _load_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from CLI arguments."""
    config_path = getattr(args, "config", None)
    if config_path and Path(config_path).exists():
        cfg = PipelineConfig.from_json(config_path)
    else:
        cfg = PipelineConfig.default()

    # Override specific fields from CLI flags
    if getattr(args, "data_dir", None):
        cfg.data_dir = args.data_dir
    if getattr(args, "output", None):
        cfg.output_dir = args.output
        cfg.tables_dir = str(Path(args.output) / "tables")
        cfg.figures_dir = str(Path(args.output) / "figures")
        cfg.circuits_fig_dir = str(Path(args.output) / "figures" / "circuits")
        cfg.analytics_fig_dir = str(Path(args.output) / "figures" / "analytics")
        cfg.drt_fig_dir = str(Path(args.output) / "figures" / "drt")
        cfg.reports_dir = str(Path(args.output) / "circuit_reports")
        cfg.excel_dir = str(Path(args.output) / "excel")
    if getattr(args, "language", None):
        cfg.language = args.language

    return cfg


def _use_json(args: argparse.Namespace) -> bool:
    """Check if --json flag is set."""
    return getattr(args, "json", False)


def _print_json(data: Dict[str, Any]) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: eis
# ═══════════════════════════════════════════════════════════════════════

def cmd_eis(args: argparse.Namespace) -> int:
    """Run the EIS analysis pipeline."""
    cfg = _load_config(args)
    json_mode = _use_json(args)

    progress = _ProgressReporter(5, "EIS Pipeline", use_json=json_mode)

    try:
        progress.update(msg="Loading configuration")
        cfg.ensure_dirs()

        progress.update(msg="Importing EIS pipeline")
        from main import run_eis_pipeline

        progress.update(msg="Running EIS pipeline")
        result = run_eis_pipeline(config=cfg)

        progress.update(msg="Processing results")

        n_files = len(result.get("raw_eis", {})) if result else 0
        out_dir = result.get("out_dir", cfg.tables_dir) if result else cfg.tables_dir

        progress.update(msg="Done")
        progress.close()

        if json_mode:
            _print_json({
                "status": "success",
                "pipeline": "eis",
                "files_processed": n_files,
                "output_dir": str(out_dir),
                "has_ranked_df": result.ranked_df is not None if result else False,
                "has_pca": result.pca is not None and result.pca.df_pca is not None if result else False,
                "has_circuit_table": result.circuit_table is not None if result else False,
            })
        else:
            print(f"\n✅ EIS pipeline complete — {n_files} files processed")
            print(f"   Output: {out_dir}")

        return RC_OK

    except FileNotFoundError as exc:
        progress.close()
        if json_mode:
            _print_json({"status": "error", "pipeline": "eis", "error": str(exc)})
        else:
            print(f"\n❌ Error: {exc}", file=sys.stderr)
        return RC_ERROR

    except Exception as exc:
        progress.close()
        logger.error("EIS pipeline failed: %s", exc, exc_info=True)
        if json_mode:
            _print_json({"status": "error", "pipeline": "eis", "error": str(exc)})
        else:
            print(f"\n❌ EIS pipeline failed: {exc}", file=sys.stderr)
        return RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: cycling
# ═══════════════════════════════════════════════════════════════════════

def cmd_cycling(args: argparse.Namespace) -> int:
    """Run the cycling analysis pipeline."""
    cfg = _load_config(args)
    json_mode = _use_json(args)
    scan_rate = getattr(args, "scan_rate", None) or cfg.scan_rate

    progress = _ProgressReporter(4, "Cycling Pipeline", use_json=json_mode)

    try:
        progress.update(msg="Loading configuration")
        cfg.ensure_dirs()

        # Override processed dir if data_dir given
        if getattr(args, "data_dir", None):
            cfg.processed_dir = args.data_dir

        progress.update(msg="Importing cycling pipeline")
        from main_cycling import run_ciclagem_pipeline

        progress.update(msg="Running cycling pipeline")
        result = run_ciclagem_pipeline(
            scan_rate=scan_rate,
            show_plots=False,
            config=cfg,
        )

        n_files = len(result.get("results", {})) if result else 0

        progress.update(msg="Done")
        progress.close()

        if json_mode:
            _print_json({
                "status": "success",
                "pipeline": "cycling",
                "scan_rate": scan_rate,
                "files_processed": n_files,
            })
        else:
            print(f"\n✅ Cycling pipeline complete — {n_files} files processed")
            print(f"   Scan rate: {scan_rate} A/g")

        return RC_OK

    except FileNotFoundError as exc:
        progress.close()
        if json_mode:
            _print_json({"status": "error", "pipeline": "cycling", "error": str(exc)})
        else:
            print(f"\n❌ Error: {exc}", file=sys.stderr)
        return RC_ERROR

    except Exception as exc:
        progress.close()
        logger.error("Cycling pipeline failed: %s", exc, exc_info=True)
        if json_mode:
            _print_json({"status": "error", "pipeline": "cycling", "error": str(exc)})
        else:
            print(f"\n❌ Cycling pipeline failed: {exc}", file=sys.stderr)
        return RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: drt
# ═══════════════════════════════════════════════════════════════════════

def cmd_drt(args: argparse.Namespace) -> int:
    """Run the DRT analysis pipeline."""
    cfg = _load_config(args)
    json_mode = _use_json(args)
    lambda_reg = getattr(args, "lambda_reg", None) or cfg.drt_lambda

    progress = _ProgressReporter(4, "DRT Pipeline", use_json=json_mode)

    try:
        progress.update(msg="Loading configuration")
        cfg.ensure_dirs()

        progress.update(msg="Importing DRT pipeline")
        from main_drt import run_drt_pipeline

        progress.update(msg="Running DRT pipeline")
        result = run_drt_pipeline(
            lambda_reg=lambda_reg,
            show_plots=False,
            config=cfg,
        )

        n_success = result.get("run_meta", {}).get("n_success", 0) if result else 0
        n_failed = result.get("run_meta", {}).get("n_failed", 0) if result else 0

        progress.update(msg="Done")
        progress.close()

        rc = RC_OK if n_failed == 0 else RC_WARNING

        if json_mode:
            _print_json({
                "status": "success" if n_failed == 0 else "warning",
                "pipeline": "drt",
                "lambda_reg": lambda_reg,
                "files_success": n_success,
                "files_failed": n_failed,
            })
        else:
            icon = "✅" if n_failed == 0 else "⚠️"
            print(f"\n{icon} DRT pipeline complete — {n_success} OK, {n_failed} failed")
            print(f"   λ = {lambda_reg}")

        return rc

    except FileNotFoundError as exc:
        progress.close()
        if json_mode:
            _print_json({"status": "error", "pipeline": "drt", "error": str(exc)})
        else:
            print(f"\n❌ Error: {exc}", file=sys.stderr)
        return RC_ERROR

    except Exception as exc:
        progress.close()
        logger.error("DRT pipeline failed: %s", exc, exc_info=True)
        if json_mode:
            _print_json({"status": "error", "pipeline": "drt", "error": str(exc)})
        else:
            print(f"\n❌ DRT pipeline failed: {exc}", file=sys.stderr)
        return RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: analyze
# ═══════════════════════════════════════════════════════════════════════

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run combined analysis with optional AI interpretation."""
    cfg = _load_config(args)
    json_mode = _use_json(args)
    run_all = getattr(args, "all", False)
    run_ai = getattr(args, "ai", False)
    export_pdf = getattr(args, "export_pdf", None)

    pipelines_run: List[str] = []
    pipeline_results: Dict[str, Any] = {}
    warnings: List[str] = []
    errors: List[str] = []

    total_steps = (3 if run_all else 0) + (1 if run_ai else 0) + (1 if export_pdf else 0) + 1
    progress = _ProgressReporter(total_steps, "Analysis", use_json=json_mode)

    try:
        progress.update(msg="Loading configuration")
        cfg.ensure_dirs()

        # ── Run individual pipelines if --all ─────────────────────
        if run_all:
            # EIS
            progress.update(msg="Running EIS pipeline")
            try:
                from main import run_eis_pipeline
                eis_result = run_eis_pipeline(config=cfg)
                pipelines_run.append("eis")
                pipeline_results["eis"] = eis_result
            except Exception as exc:
                warnings.append(f"EIS pipeline failed: {exc}")
                logger.warning("EIS pipeline failed in analyze: %s", exc)

            # Cycling
            progress.update(msg="Running cycling pipeline")
            try:
                from main_cycling import run_ciclagem_pipeline
                cyc_result = run_ciclagem_pipeline(
                    scan_rate=cfg.scan_rate, show_plots=False, config=cfg,
                )
                pipelines_run.append("cycling")
                pipeline_results["cycling"] = cyc_result
            except Exception as exc:
                warnings.append(f"Cycling pipeline failed: {exc}")
                logger.warning("Cycling pipeline failed in analyze: %s", exc)

            # DRT
            progress.update(msg="Running DRT pipeline")
            try:
                from main_drt import run_drt_pipeline
                drt_result = run_drt_pipeline(show_plots=False, config=cfg)
                pipelines_run.append("drt")
                pipeline_results["drt"] = drt_result
            except Exception as exc:
                warnings.append(f"DRT pipeline failed: {exc}")
                logger.warning("DRT pipeline failed in analyze: %s", exc)

        # ── AI Analysis ──────────────────────────────────────────
        ai_summary: Optional[str] = None
        if run_ai:
            progress.update(msg="Running AI analysis")
            try:
                from src.gui.tabs.ai_panel import run_ai_analysis, AIPanelConfig
                from src.gui.models import AppState

                state = AppState()
                # Populate state from pipeline results
                if "eis" in pipeline_results:
                    eis_r = pipeline_results["eis"]
                    state.ranked_df = eis_r.ranked_df
                    state.circuit_table = eis_r.circuit_table
                if "cycling" in pipeline_results:
                    cyc_r = pipeline_results["cycling"]
                    state.cycling_merged = cyc_r.merged_table
                if "drt" in pipeline_results:
                    drt_r = pipeline_results["drt"]
                    state.drt_table = drt_r.drt_table
                    state.drt_peaks_table = drt_r.drt_peaks_table
                    state.drt_summary_table = drt_r.drt_summary_table

                ai_result = run_ai_analysis(
                    state,
                    AIPanelConfig(),
                    pipeline_config=cfg,
                )
                ai_summary = ai_result.executive_summary
            except Exception as exc:
                warnings.append(f"AI analysis failed: {exc}")
                logger.warning("AI analysis failed in analyze: %s", exc)

        # ── Export PDF ───────────────────────────────────────────
        if export_pdf:
            progress.update(msg="Exporting PDF")
            try:
                from src.report_generator import ReportGenerator
                gen = ReportGenerator(config=cfg)
                gen.generate(export_pdf, pipeline_results, ai_summary=ai_summary)
            except ImportError:
                warnings.append("PDF export requires report_generator module (Day 23)")
                logger.warning("report_generator not yet available")
            except Exception as exc:
                warnings.append(f"PDF export failed: {exc}")
                logger.warning("PDF export failed: %s", exc)

        progress.close()

        # ── Determine return code ────────────────────────────────
        if errors:
            rc = RC_ERROR
        elif warnings:
            rc = RC_WARNING
        else:
            rc = RC_OK

        if json_mode:
            output = {
                "status": "success" if rc == RC_OK else ("warning" if rc == RC_WARNING else "error"),
                "pipelines_run": pipelines_run,
                "warnings": warnings,
                "errors": errors,
                "ai_summary": ai_summary,
                "export_pdf": export_pdf,
            }
            _print_json(output)
        else:
            icon = "✅" if rc == RC_OK else ("⚠️" if rc == RC_WARNING else "❌")
            print(f"\n{icon} Analysis complete — pipelines: {', '.join(pipelines_run) or 'none'}")
            if ai_summary:
                print(f"\n🤖 AI Summary:\n{ai_summary}")
            if export_pdf:
                print(f"   PDF: {export_pdf}")
            for w in warnings:
                print(f"   ⚠️  {w}", file=sys.stderr)
            for e in errors:
                print(f"   ❌ {e}", file=sys.stderr)

        return rc

    except Exception as exc:
        progress.close()
        logger.error("Analysis failed: %s", exc, exc_info=True)
        if json_mode:
            _print_json({"status": "error", "error": str(exc)})
        else:
            print(f"\n❌ Analysis failed: {exc}", file=sys.stderr)
        return RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: config
# ═══════════════════════════════════════════════════════════════════════

def cmd_config(args: argparse.Namespace) -> int:
    """Manage pipeline configuration."""
    json_mode = _use_json(args)
    init = getattr(args, "init", False)
    show = getattr(args, "show", False)
    output_path = getattr(args, "output", None) or "config.json"

    if init:
        cfg = PipelineConfig.default()
        cfg.to_json(output_path)
        if json_mode:
            _print_json({
                "status": "success",
                "action": "config_init",
                "path": str(output_path),
            })
        else:
            print(f"✅ Default config written to {output_path}")
        return RC_OK

    if show:
        config_path = getattr(args, "config", None)
        if config_path and Path(config_path).exists():
            cfg = PipelineConfig.from_json(config_path)
        else:
            cfg = PipelineConfig.default()
        if json_mode:
            _print_json(cfg.to_dict())
        else:
            for key, value in cfg.to_dict().items():
                print(f"  {key}: {value}")
        return RC_OK

    # No action specified — show help
    if json_mode:
        _print_json({
            "status": "error",
            "error": "No config action specified. Use --init or --show.",
        })
    else:
        print("No config action specified. Use --init or --show.", file=sys.stderr)
    return RC_ERROR


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: validate
# ═══════════════════════════════════════════════════════════════════════

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate EIS data files (quality check + Kramers-Kronig)."""
    cfg = _load_config(args)
    json_mode = _use_json(args)

    data_dir = cfg.data_dir
    data_path = Path(data_dir)

    if not data_path.exists():
        if json_mode:
            _print_json({"status": "error", "error": f"Directory not found: {data_dir}"})
        else:
            print(f"❌ Directory not found: {data_dir}", file=sys.stderr)
        return RC_ERROR

    txt_files = sorted(f for f in os.listdir(data_path) if f.lower().endswith(".txt"))
    if not txt_files:
        if json_mode:
            _print_json({"status": "error", "error": f"No .txt files in {data_dir}"})
        else:
            print(f"❌ No .txt files in {data_dir}", file=sys.stderr)
        return RC_ERROR

    progress = _ProgressReporter(len(txt_files), "Validating", use_json=json_mode)

    results: List[Dict[str, Any]] = []
    n_valid = 0
    n_invalid = 0
    n_errors = 0

    from src.loader import load_eis_file
    from src.preprocessing import preprocess
    from src.validation import validate_eis_full
    from src.kramers_kronig import KramersKronigValidator

    kk_validator = KramersKronigValidator()

    for filename in txt_files:
        filepath = str(data_path / filename)
        file_result: Dict[str, Any] = {"file": filename}

        try:
            df = preprocess(load_eis_file(filepath))

            # Data validation
            vr = validate_eis_full(df)
            file_result["validation_ok"] = vr.ok
            file_result["validation_errors"] = len(vr.errors)
            file_result["validation_warnings"] = len(vr.warnings)

            # Kramers-Kronig test
            freq = df["frequency"].values
            z_real = df["zreal"].values
            z_imag = df["zimag"].values
            kk_result = kk_validator.validate(freq, z_real, z_imag)

            file_result["kk_valid"] = kk_result.kk_valid
            file_result["kk_classification"] = kk_result.classification
            file_result["kk_mean_residual"] = round(kk_result.mean_residual, 6)
            file_result["kk_max_residual"] = round(kk_result.max_residual, 6)

            if vr.ok and kk_result.kk_valid:
                n_valid += 1
                file_result["status"] = "valid"
            else:
                n_invalid += 1
                file_result["status"] = "invalid"

        except Exception as exc:
            n_errors += 1
            file_result["status"] = "error"
            file_result["error"] = str(exc)

        results.append(file_result)
        status_icon = {"valid": "✓", "invalid": "⚠", "error": "✗"}.get(
            file_result["status"], "?"
        )
        progress.update(msg=f"{status_icon} {filename}")

    progress.close()

    # ── Determine return code ────────────────────────────────────
    if n_errors > 0:
        rc = RC_ERROR
    elif n_invalid > 0:
        rc = RC_WARNING
    else:
        rc = RC_OK

    if json_mode:
        _print_json({
            "status": "success" if rc == RC_OK else ("warning" if rc == RC_WARNING else "error"),
            "total_files": len(txt_files),
            "valid": n_valid,
            "invalid": n_invalid,
            "errors": n_errors,
            "files": results,
        })
    else:
        icon = "✅" if rc == RC_OK else ("⚠️" if rc == RC_WARNING else "❌")
        print(f"\n{icon} Validation complete — {len(txt_files)} files")
        print(f"   ✓ Valid:   {n_valid}")
        print(f"   ⚠ Invalid: {n_invalid}")
        print(f"   ✗ Errors:  {n_errors}")
        print()
        for r in results:
            s = r["status"]
            icon_f = {"valid": "✓", "invalid": "⚠", "error": "✗"}[s]
            extra = ""
            if s == "invalid":
                parts = []
                if not r.get("validation_ok", True):
                    parts.append(f"{r.get('validation_errors', 0)} validation errors")
                if not r.get("kk_valid", True):
                    parts.append(f"KK {r.get('kk_classification', 'suspeito')}")
                extra = f" ({', '.join(parts)})"
            elif s == "error":
                extra = f" ({r.get('error', '')})"
            print(f"  {icon_f} {r['file']}{extra}")

    return rc


# ═══════════════════════════════════════════════════════════════════════
# Subcommand: version
# ═══════════════════════════════════════════════════════════════════════

def cmd_version(args: argparse.Namespace) -> int:
    """Print version information."""
    from src import __version__
    json_mode = _use_json(args)

    if json_mode:
        _print_json({"version": __version__})
    else:
        print(f"IonFlow Pipeline v{__version__}")
    return RC_OK


# ═══════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI argument parser with all subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready for ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        prog="ionflow",
        description="IonFlow Pipeline — EIS analytics toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ionflow eis --data-dir data/raw\n"
            "  ionflow cycling --scan-rate 0.1\n"
            "  ionflow drt --lambda 1e-3\n"
            "  ionflow analyze --all --ai\n"
            "  ionflow config --init\n"
            "  ionflow validate --data-dir data/raw\n"
        ),
    )

    # Global flags
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output results as JSON (for programmatic integration)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a config.json file",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        choices=["pt", "en", "es"],
        help="Interface language",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Enable verbose (DEBUG) logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── eis ───────────────────────────────────────────────────────
    sp_eis = subparsers.add_parser(
        "eis", help="Run the EIS impedance analysis pipeline",
    )
    sp_eis.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing raw EIS .txt files (default: data/raw)",
    )
    sp_eis.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: outputs/)",
    )
    sp_eis.set_defaults(func=cmd_eis)

    # ── cycling ──────────────────────────────────────────────────
    sp_cyc = subparsers.add_parser(
        "cycling", help="Run the galvanostatic cycling analysis pipeline",
    )
    sp_cyc.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing processed cycling .txt files",
    )
    sp_cyc.add_argument(
        "--scan-rate", type=float, default=None,
        help="Scan rate in A/g (default: from config)",
    )
    sp_cyc.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    sp_cyc.set_defaults(func=cmd_cycling)

    # ── drt ──────────────────────────────────────────────────────
    sp_drt = subparsers.add_parser(
        "drt", help="Run the DRT (Distribution of Relaxation Times) pipeline",
    )
    sp_drt.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing raw EIS .txt files for DRT",
    )
    sp_drt.add_argument(
        "--lambda", dest="lambda_reg", type=float, default=None,
        help="Tikhonov regularisation parameter λ (default: 1e-3)",
    )
    sp_drt.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    sp_drt.set_defaults(func=cmd_drt)

    # ── analyze ──────────────────────────────────────────────────
    sp_analyze = subparsers.add_parser(
        "analyze", help="Run combined analysis with optional AI",
    )
    sp_analyze.add_argument(
        "--all", action="store_true", default=False,
        help="Run all pipelines (EIS + Cycling + DRT)",
    )
    sp_analyze.add_argument(
        "--ai", action="store_true", default=False,
        help="Include AI-powered interpretation",
    )
    sp_analyze.add_argument(
        "--export-pdf", type=str, default=None, metavar="FILE",
        help="Export results to a PDF report",
    )
    sp_analyze.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory",
    )
    sp_analyze.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    sp_analyze.set_defaults(func=cmd_analyze)

    # ── config ───────────────────────────────────────────────────
    sp_config = subparsers.add_parser(
        "config", help="Manage pipeline configuration",
    )
    sp_config.add_argument(
        "--init", action="store_true", default=False,
        help="Generate a default config.json file",
    )
    sp_config.add_argument(
        "--show", action="store_true", default=False,
        help="Display current configuration values",
    )
    sp_config.add_argument(
        "--output", type=str, default=None,
        help="Output path for --init (default: config.json)",
    )
    sp_config.set_defaults(func=cmd_config)

    # ── validate ─────────────────────────────────────────────────
    sp_validate = subparsers.add_parser(
        "validate", help="Validate EIS data files (quality + Kramers-Kronig)",
    )
    sp_validate.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing EIS .txt files to validate",
    )
    sp_validate.add_argument(
        "--output", type=str, default=None,
        help="Output directory (for logs)",
    )
    sp_validate.set_defaults(func=cmd_validate)

    # ── version ──────────────────────────────────────────────────
    sp_version = subparsers.add_parser(
        "version", help="Show version information",
    )
    sp_version.set_defaults(func=cmd_version)

    return parser


# ═══════════════════════════════════════════════════════════════════════
# Main entry-point
# ═══════════════════════════════════════════════════════════════════════

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Return code (0=ok, 1=error, 2=warning).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set up logging
    verbose = getattr(args, "verbose", False)
    console_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(console_level=console_level, force=True)

    # No command given → print help
    if not hasattr(args, "func") or args.func is None:
        parser.print_help()
        return RC_ERROR

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
