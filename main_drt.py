"""DRT pipeline — Distribution of Relaxation Times analysis.

Reads every .txt from ``data/raw``, fits a DRT spectrum per file using
Tikhonov regularisation, extracts relaxation peaks, exports a summary table
and one PNG plot per sample, then returns a unified result dictionary.

Return contract
---------------
``run_drt_pipeline(**kwargs) -> dict``

Keys
~~~~
drt_table        : pd.DataFrame
    One row per successfully processed file.
    Columns:
        Arquivo       — original filename (str)
        R_inf         — high-frequency resistance [Ω]  (float)
        n_peaks       — number of DRT peaks detected   (int)
        tau_peak_1    — relaxation time of peak 1 [s]  (float | NaN)
        gamma_peak_1  — DRT amplitude of peak 1 [Ω]    (float | NaN)
        tau_peak_2    — relaxation time of peak 2 [s]  (float | NaN)
        gamma_peak_2  — DRT amplitude of peak 2 [Ω]    (float | NaN)
        tau_peak_3    — relaxation time of peak 3 [s]  (float | NaN)
        gamma_peak_3  — DRT amplitude of peak 3 [Ω]    (float | NaN)

per_file_results : dict[str, DRTResult]
    ``{filename_stem: DRTResult}`` — raw arrays / peaks for programmatic use.
    Keys of each DRTResult: tau, gamma, r_inf, peaks, residuals,
    lambda_reg, n_taus.  See ``src.drt_analysis`` for full contract.

plot_paths       : list[tuple[str, str]]
    ``[(filename_stem, abs_path_to_png), ...]``
    Plots are written to ``outputs/figures/drt/<stem>_drt.png``.

drt_peaks_table  : pd.DataFrame
    Uma linha por pico detectado (granularidade de pico).
    Colunas: ``Arquivo``, ``Sample``, ``peak_order``, ``tau_peak``,
    ``gamma_peak``, ``width_decades``.

drt_summary_table : pd.DataFrame
    Resumo por amostra com principais métricas agregadas.
    Colunas: ``Arquivo``, ``Sample``, ``n_peaks``, ``R_inf``,
    ``tau_peak_main``, ``gamma_peak_main``, ``residual_mean``,
    ``residual_max``.

run_meta        : dict
    Metadados da execução DRT para rastreabilidade.
    Chaves: ``lambda_reg``, ``n_taus``, ``data_dir``, ``n_files``,
    ``n_success``, ``n_failed``, ``generated_at``.

errors           : dict[str, str]
    ``{filename: error_message}`` for files that could not be processed.
    These files are *absent* from ``drt_table`` and ``per_file_results``.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.drt_analysis import DRTResult, compute_drt
from src.drt_visualization import plot_drt_spectrum
from src.loader import load_eis_file
from src.preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum number of peak columns exported to drt_table
_MAX_PEAKS_EXPORTED = 3

# Column schema for an empty drt_table (returned when no files succeed)
_DRT_TABLE_COLUMNS = [
    "Arquivo",
    "R_inf",
    "n_peaks",
    "tau_peak_1",
    "gamma_peak_1",
    "tau_peak_2",
    "gamma_peak_2",
    "tau_peak_3",
    "gamma_peak_3",
]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_drt_pipeline(
    lambda_reg: float | None = None,
    n_taus: int | None = None,
    data_dir: str | None = None,
    show_plots: bool = False,
    config: Optional[PipelineConfig] = None,
) -> dict[str, Any]:
    """Run the DRT pipeline on all ``.txt`` files in *data_dir*.

    Parameters
    ----------
    lambda_reg : float, optional
        Tikhonov regularisation parameter λ.  Larger → smoother γ(τ).
        Default from config (``1e-3``).  Typical range: 1e-5 … 1e-1.
    n_taus : int, optional
        Number of log-uniform τ discretisation points.  Default from config.
    data_dir : str, optional
        Directory containing raw EIS ``.txt`` files.  Default from config.
    show_plots : bool, optional
        Call ``plt.show()`` interactively.  Set ``False`` when running from
        the GUI thread.  Default ``False``.
    config : PipelineConfig, optional
        Centralised config; uses defaults when ``None``.

    Returns
    -------
    dict
        See module docstring for the full key contract.

    Raises


    ------
    FileNotFoundError
        If *data_dir* does not exist.
    """
    cfg = config or PipelineConfig.default()
    lambda_reg = lambda_reg if lambda_reg is not None else cfg.drt_lambda
    n_taus = n_taus if n_taus is not None else cfg.drt_n_taus
    data_dir = data_dir if data_dir is not None else cfg.data_dir
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    figures_dir = Path(cfg.drt_fig_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = Path(cfg.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_file_results: dict[str, DRTResult] = {}
    plot_paths: list[tuple[str, str]] = []
    errors: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    txt_files = sorted(
        f for f in os.listdir(data_path) if f.lower().endswith(".txt")
    )

    if not txt_files:
        logger.warning("No .txt files found in %s", data_dir)

    for filename in txt_files:
        filepath = str(data_path / filename)
        stem = Path(filename).stem

        try:
            df = preprocess(load_eis_file(filepath))
            freq = df["frequency"].values
            z_real = df["zreal"].values
            # zimag is stored as -|Z''| by loader → passed as-is; compute_drt
            # negates it internally to obtain the positive -Z'' for the kernel.
            z_imag = df["zimag"].values

            result = compute_drt(
                freq, z_real, z_imag,
                n_taus=n_taus,
                lambda_reg=lambda_reg,
            )
            per_file_results[stem] = result

            residuals = np.asarray(result.get("residuals", []), dtype=float)
            peaks = result.get("peaks", []) or []
            main_peak = peaks[0] if peaks else {}

            # ------------------------------------------------------------------
            # Build table row (up to _MAX_PEAKS_EXPORTED peaks exported)
            # ------------------------------------------------------------------
            row: dict[str, Any] = {
                "Arquivo": filename,
                "R_inf": round(result["r_inf"], 6),
                "n_peaks": len(peaks),
            }
            for k in range(1, cfg.drt_max_peaks_exported + 1):
                if k <= len(peaks):
                    pk = peaks[k - 1]
                    row[f"tau_peak_{k}"] = pk["tau_peak"]
                    row[f"gamma_peak_{k}"] = pk["gamma_peak"]
                else:
                    row[f"tau_peak_{k}"] = np.nan
                    row[f"gamma_peak_{k}"] = np.nan
            rows.append(row)

            for order, peak in enumerate(peaks, start=1):
                peak_rows.append(
                    {
                        "Arquivo": filename,
                        "Sample": stem,
                        "peak_order": order,
                        "tau_peak": peak.get("tau_peak", np.nan),
                        "gamma_peak": peak.get("gamma_peak", np.nan),
                        "width_decades": peak.get("width_decades", np.nan),
                    }
                )

            summary_rows.append(
                {
                    "Arquivo": filename,
                    "Sample": stem,
                    "n_peaks": len(peaks),
                    "R_inf": result.get("r_inf", np.nan),
                    "tau_peak_main": main_peak.get("tau_peak", np.nan),
                    "gamma_peak_main": main_peak.get("gamma_peak", np.nan),
                    "residual_mean": (
                        float(np.nanmean(residuals))
                        if residuals.size > 0
                        else np.nan
                    ),
                    "residual_max": (
                        float(np.nanmax(residuals)) if residuals.size > 0 else np.nan
                    ),
                }
            )

            # ------------------------------------------------------------------
            # Save plot
            # ------------------------------------------------------------------
            plot_path = plot_drt_spectrum(
                result,
                stem,
                out_dir=figures_dir,
                show=show_plots,
                save=True,
            )
            plot_paths.append((stem, plot_path))

            logger.info(
                "DRT OK: %-40s  peaks=%d  R_inf=%.4f Ω",
                filename, len(result["peaks"]), result["r_inf"],
            )

        except Exception as exc:
            logger.warning("DRT FAILED: %s — %s", filename, exc)
            errors[filename] = str(exc)

    # ------------------------------------------------------------------
    # Assemble drt_table
    # ------------------------------------------------------------------
    if rows:
        drt_table = pd.DataFrame(rows, columns=_DRT_TABLE_COLUMNS)
    else:
        drt_table = pd.DataFrame(columns=_DRT_TABLE_COLUMNS)

    # Export CSV
    out_csv = tables_dir / "drt_results.csv"
    drt_table.to_csv(out_csv, index=False)
    logger.info("DRT table saved → %s  (%d rows)", out_csv, len(drt_table))

    drt_peaks_table = pd.DataFrame(
        peak_rows,
        columns=[
            "Arquivo",
            "Sample",
            "peak_order",
            "tau_peak",
            "gamma_peak",
            "width_decades",
        ],
    )
    drt_peaks_csv = tables_dir / "drt_peaks.csv"
    drt_peaks_table.to_csv(drt_peaks_csv, index=False)

    drt_summary_table = pd.DataFrame(
        summary_rows,
        columns=[
            "Arquivo",
            "Sample",
            "n_peaks",
            "R_inf",
            "tau_peak_main",
            "gamma_peak_main",
            "residual_mean",
            "residual_max",
        ],
    )
    drt_summary_csv = tables_dir / "drt_summary.csv"
    drt_summary_table.to_csv(drt_summary_csv, index=False)

    if errors:
        logger.warning(
            "DRT pipeline: %d file(s) failed: %s",
            len(errors), ", ".join(errors.keys()),
        )

    run_meta = {
        "lambda_reg": float(lambda_reg),
        "n_taus": int(n_taus),
        "data_dir": str(data_dir),
        "n_files": int(len(txt_files)),
        "n_success": int(len(per_file_results)),
        "n_failed": int(len(errors)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {
        "drt_table": drt_table,
        "drt_peaks_table": drt_peaks_table,
        "drt_summary_table": drt_summary_table,
        "per_file_results": per_file_results,
        "plot_paths": plot_paths,
        "errors": errors,
        "run_meta": run_meta,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_drt_pipeline()
