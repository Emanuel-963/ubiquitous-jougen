"""IonFlow Pipeline — EIS analytics package."""

__version__ = "0.1.0"

# Convenience re-exports so users can do ``from src import tr, check_for_updates``
from src.config import PipelineConfig  # noqa: F401
from src.feature_store import FeatureStore, FittingHistory  # noqa: F401
from src.i18n import get_language, set_language, tr  # noqa: F401
from src.ml_circuit_selector import CircuitMLSelector  # noqa: F401
from src.logger import setup_logging, GUIQueueHandler, get_logger  # noqa: F401
from src.models import CyclingResult, DRTPipelineResult, EISResult, PCAResult  # noqa: F401
from src.updater import check_for_updates  # noqa: F401
from src.validation import (  # noqa: F401
    ValidationResult,
    validate_eis_dataframe,
    validate_cycling_dataframe,
    validate_frequency_range,
    validate_impedance_quality,
    validate_eis_full,
)
