"""IonFlow Pipeline — EIS analytics package."""

__version__ = "0.1.0"

# Convenience re-exports so users can do ``from src import tr, check_for_updates``
from src.circuit_composer import CircuitBlock, CircuitComposer  # noqa: F401
from src.config import PipelineConfig  # noqa: F401
from src.gui import AppState, MainWindow, PipelineController, PlotItem  # noqa: F401
from src.gui import AIPanelConfig, AIPanelResult, run_ai_analysis  # noqa: F401
from src.fitting_diagnostics import FittingDiagnostics, QualityIndicator, assess_quality  # noqa: F401
from src.fitting_report import FittingReport, FittingReportGenerator  # noqa: F401
from src.feature_store import FeatureStore, FittingHistory  # noqa: F401
from src.i18n import get_language, set_language, tr  # noqa: F401
from src.kramers_kronig import KKResult, KramersKronigValidator  # noqa: F401
from src.ml_circuit_selector import CircuitMLSelector  # noqa: F401
from src.logger import setup_logging, GUIQueueHandler, get_logger  # noqa: F401
from src.models import CyclingResult, DRTPipelineResult, EISResult, PCAResult  # noqa: F401
from src.uncertainty import UncertaintyAnalyzer, MonteCarloResult, BootstrapResult  # noqa: F401
from src.updater import check_for_updates  # noqa: F401
from src.ai import ElectrochemicalRule, KnowledgeBase, RuleMatch, Severity  # noqa: F401
from src.ai import AnalysisReport, Anomaly, Finding, InferenceEngine, Priority, Recommendation  # noqa: F401
from src.ai import CyclingPrediction, DegradationMechanism, DegradationPrediction, Improvement, ImprovementArea, PerformancePredictor  # noqa: F401
from src.ai import ProcessAdvisor, ProcessReport, ProductionRec, RecommendationArea  # noqa: F401
from src.ai import LLMAdapter, LLMConfig, LLMProvider, NullAdapter, OpenAIAdapter, OllamaAdapter  # noqa: F401
from src.ai import create_adapter, create_adapter_from_config, enrich_report, enrich_summary  # noqa: F401
from src.cli import build_parser, main as cli_main, RC_OK, RC_ERROR, RC_WARNING  # noqa: F401
from src.report_generator import (  # noqa: F401
    GenerationHistory,
    GenerationRecord,
    ReportConfig,
    ReportGenerator,
    generate_markdown,
)
from src.validation import (  # noqa: F401
    ValidationResult,
    validate_eis_dataframe,
    validate_cycling_dataframe,
    validate_frequency_range,
    validate_impedance_quality,
    validate_eis_full,
)
