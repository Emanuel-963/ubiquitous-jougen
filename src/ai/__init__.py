"""IonFlow AI — Electrochemical intelligence package.

Day 15 introduces the knowledge base; Day 16+ adds the inference engine,
performance predictor and process advisor.

Re-exports
----------
>>> from src.ai import ElectrochemicalRule, KnowledgeBase
>>> from src.ai import ProcessAdvisor, ProcessReport
"""

from src.ai.knowledge_base import (  # noqa: F401
    ElectrochemicalRule,
    KnowledgeBase,
    RuleMatch,
    Severity,
)
from src.ai.inference_engine import (  # noqa: F401
    AnalysisReport,
    Anomaly,
    Finding,
    InferenceEngine,
    Priority,
    Recommendation,
)
from src.ai.performance_predictor import (  # noqa: F401
    CyclingPrediction,
    DegradationMechanism,
    DegradationPrediction,
    Improvement,
    ImprovementArea,
    PerformancePredictor,
)
from src.ai.process_advisor import (  # noqa: F401
    ProcessAdvisor,
    ProcessReport,
    ProductionRec,
    RecommendationArea,
)
