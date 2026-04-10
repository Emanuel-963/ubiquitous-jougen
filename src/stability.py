import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def extract_sample_id(filename: str) -> str:
    """Extrai um identificador de amostra a partir do nome do arquivo.

    Tenta padrões como: "1 Nb2 ..." → "1Nb2". Se não casar, retorna
    o nome (sem extensão) como fallback.
    """
    m = re.match(r"^(\d+)\s+(\S+)", filename)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    # fallback: sem extensão
    return re.sub(r"\.[^.]+$", "", filename)


def stability_metrics(df: pd.DataFrame, param: str) -> pd.DataFrame:
    grouped = df.groupby("Sample")[param]

    mean = grouped.mean()
    std = grouped.std()

    # Evitar divisão por zero
    mean_clean = mean.mask(mean == 0)
    cv = std / mean_clean

    result = pd.DataFrame({"Mean": mean, "Std": std, "CV": cv})

    return result
