"""IonFlow AI — Electrochemical intelligence package.

Day 15 introduces the knowledge base; Day 16+ adds the inference engine,
performance predictor and process advisor.

Re-exports
----------
>>> from src.ai import ElectrochemicalRule, KnowledgeBase
"""

from src.ai.knowledge_base import (  # noqa: F401
    ElectrochemicalRule,
    KnowledgeBase,
    RuleMatch,
    Severity,
)
