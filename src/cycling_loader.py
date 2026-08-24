"""Loader for cycling data files (.txt) in data/processed.

Loads files with columns: tempo, corrente, potencial, ciclo.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_CYCLING_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")
_CYCLING_SEPARATORS = (";", "\t", ",", None)


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _read_cycling_file(file: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for sep in _CYCLING_SEPARATORS:
        for enc in _CYCLING_ENCODINGS:
            try:
                kwargs = dict(
                    filepath_or_buffer=file,
                    sep=sep,
                    decimal=",",
                    encoding=enc,
                )
                if sep is None:
                    kwargs["engine"] = "python"
                else:
                    kwargs["low_memory"] = False

                df = pd.read_csv(**kwargs)
                if df.shape[1] >= 2:
                    return df
            except Exception as exc:
                last_error = exc
                continue
    msg = f"File {file} could not be decoded/parsed"
    if last_error is not None:
        msg += f": {last_error}"
    raise ValueError(msg)


def load_cycling_files(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all cycling ``.txt`` files from a directory.

    Files are expected to use semicolon separators and European decimal
    format (comma). Columns are renamed to the internal convention
    (*tempo*, *corrente*, *potencial*, *ciclo*) and rows with NaN/inf
    values are dropped.

    Parameters
    ----------
    directory : pathlib.Path
        Folder containing ``.txt`` cycling data files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of file stem → cleaned DataFrame with columns
        *tempo*, *corrente*, *potencial*, and *ciclo*.

    Raises
    ------
    ValueError
        If a file does not contain the expected columns after renaming.
    """
    files = list(directory.glob("*.txt"))
    data = {}
    for file in files:
        df = _read_cycling_file(file)

        col_tempo = None
        col_corrente = None
        col_potencial = None
        col_ciclo = None

        for col in df.columns:
            c = _normalize_col_name(col)
            if col_tempo is None and ("time" in c or "tempo" in c):
                col_tempo = col
            elif col_corrente is None and ("current" in c or c == "ima"):
                col_corrente = col
            elif col_potencial is None and (
                "potential" in c or "ecell" in c or "voltage" in c
            ):
                col_potencial = col
            elif col_ciclo is None and ("cycle" in c or "cyclenumber" in c):
                col_ciclo = col

        column_mapping = {}
        if col_tempo is not None:
            column_mapping[col_tempo] = "tempo"
        if col_corrente is not None:
            column_mapping[col_corrente] = "corrente"
        if col_potencial is not None:
            column_mapping[col_potencial] = "potencial"
        if col_ciclo is not None:
            column_mapping[col_ciclo] = "ciclo"

        df = df.rename(columns=column_mapping)
        # Ensure required columns exist
        required_cols = ["tempo", "corrente", "potencial", "ciclo"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"File {file} missing required columns after mapping: {required_cols}"
            )
        # Select only required columns
        df = df[required_cols]
        for col in required_cols:
            df[col] = _to_numeric(df[col])

        # Drop rows with NaN/inf before converting ciclo to int
        df = df.dropna(subset=required_cols)
        finite_mask = np.isfinite(df[required_cols]).all(axis=1)
        df = df[finite_mask]
        df["ciclo"] = df["ciclo"].round().astype(int)
        data[file.stem] = df

    if files and not data:
        logger.warning("No valid cycling files were loaded from %s", directory)
    return data
