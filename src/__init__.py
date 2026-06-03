"""IonFlow Pipeline — EIS analytics package."""

__version__ = "0.4.11"

from src.ai import (  # noqa: F401
    AnalysisReport,
    Anomaly,
    CyclingPrediction,
    DegradationMechanism,
    DegradationPrediction,
    ElectrochemicalRule,
    Finding,
    Improvement,
    ImprovementArea,
    InferenceEngine,
    KnowledgeBase,
    LLMAdapter,
    LLMConfig,
    LLMProvider,
    NullAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    PerformancePredictor,
    Priority,
    ProcessAdvisor,
    ProcessReport,
    ProductionRec,
    Recommendation,
    RecommendationArea,
    RuleMatch,
    Severity,
    create_adapter,
    create_adapter_from_config,
    enrich_report,
    enrich_summary,
)
from src.batch_processor import (  # noqa: F401
    BatchProcessor,
    BatchProgress,
    BatchResult,
    ParallelFitter,
    max_workers,
)

# Convenience re-exports so users can do ``from src import tr, check_for_updates``
from src.circuit_composer import CircuitBlock, CircuitComposer  # noqa: F401
from src.circuit_fitting import fit_template, orazem_sigma  # noqa: F401
from src.cli import main as cli_main  # noqa: F401
from src.config import PipelineConfig  # noqa: F401
from src.db import FeatureStoreV2, IonFlowRepository  # noqa: F401
from src.feature_store import FeatureStore, FittingHistory  # noqa: F401
from src.fitting_diagnostics import (  # noqa: F401
    FittingDiagnostics,
    QualityIndicator,
    assess_quality,
)
from src.fitting_report import FittingReport, FittingReportGenerator  # noqa: F401
from src.gui import (  # noqa: F401
    AIPanelConfig,
    AIPanelResult,
    AppState,
    MainWindow,
    PipelineController,
    PlotItem,
    run_ai_analysis,
)
from src.gui.shortcuts import (  # noqa: F401
    DEFAULT_BINDINGS,
    DEFAULT_TOOLTIPS,
    AccessibilitySettings,
    ShortcutAction,
    ShortcutBinding,
    ShortcutManager,
    StatusBarState,
    TooltipRegistry,
)
from src.i18n import (  # noqa: F401
    LANGUAGES,
    SECTIONS,
    available_keys,
    get_language,
    get_languages,
    get_section,
    missing_keys,
    reload_strings,
    set_language,
    tr,
    tr_section,
    translation_coverage,
)
from src.kramers_kronig import KKResult, KramersKronigValidator  # noqa: F401
from src.logger import GUIQueueHandler, get_logger, setup_logging  # noqa: F401
from src.ml_circuit_selector import CircuitMLSelector  # noqa: F401
from src.models import (  # noqa: F401
    CyclingResult,
    DRTPipelineResult,
    EISResult,
    PCAResult,
)
from src.report_generator import (  # noqa: F401
    GenerationHistory,
    GenerationRecord,
    ReportConfig,
    ReportGenerator,
    generate_markdown,
)
from src.uncertainty import (  # noqa: F401
    BootstrapResult,
    MonteCarloResult,
    UncertaintyAnalyzer,
)
from src.updater import check_for_updates  # noqa: F401
from src.validation import (  # noqa: F401
    ValidationResult,
    detect_powerline_noise,
    estimate_critical_frequency,
    remove_highest_frequency_point,
    truncate_above_fc,
    validate_cycling_dataframe,
    validate_eis_dataframe,
    validate_eis_full,
    validate_frequency_range,
    validate_impedance_quality,
)

# v0.4.11 new modules
from src.ai.auto_summary import (  # noqa: F401
    generate_auto_summary,
    generate_next_steps,
)
from src.figure_pack import export_figure_pack  # noqa: F401
from src.journal_styles import (  # noqa: F401
    JOURNAL_STYLES,
    STYLE_LABELS,
    apply_journal_style,
    get_available_styles,
    journal_style_context,
)
from src.loader import EISLoadError  # noqa: F401
