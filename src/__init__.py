"""IonFlow Pipeline — EIS analytics package."""

__version__ = "0.1.0"

# Convenience re-exports so users can do ``from src import tr, check_for_updates``
from src.config import PipelineConfig  # noqa: F401
from src.i18n import get_language, set_language, tr  # noqa: F401
from src.updater import check_for_updates  # noqa: F401
