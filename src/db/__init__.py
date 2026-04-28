"""IonFlow SQLite backend (Phase 7).

Provides a persistent relational store for all pipeline results and a
SQLite-backed replacement for the JSON-based :class:`FeatureStore`.

Quick start
-----------
::

    from src.db import IonFlowRepository, FeatureStoreV2

    repo = IonFlowRepository("data/ionflow.db")
    run_id = repo.add_sample("MyRun-2026-05-01", "eis")
    repo.save_eis_results(run_id, ranked_df)

    store = FeatureStoreV2("data/ionflow.db")   # shares the same file
    print(store.summary_text())
"""

from src.db.feature_store_v2 import FeatureStoreV2  # noqa: F401
from src.db.migrations import run_migrations  # noqa: F401
from src.db.repository import IonFlowRepository  # noqa: F401
from src.db.schema import init_db  # noqa: F401

__all__ = [
    "FeatureStoreV2",
    "IonFlowRepository",
    "init_db",
    "run_migrations",
]
