"""Tests for the electrochemistry knowledge base (Day 15).

Covers:
- ElectrochemicalRule dataclass construction, serialisation, from_dict
- Severity enum
- RuleMatch dataclass
- KnowledgeBase CRUD (add, remove, get, by_category, by_severity, by_parameter)
- Condition micro-DSL evaluation (_eval_condition, _resolve_thresholds)
- KnowledgeBase.evaluate() with measurements and thresholds
- JSON round-trip (to_json → from_json)
- KnowledgeBase.default() with 56 bundled rules
- thresholds_from_config() helper
- Package-level imports (src.ai)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ai.knowledge_base import (
    ElectrochemicalRule,
    KnowledgeBase,
    RuleMatch,
    Severity,
    _builtin_rules,
    _eval_condition,
    _resolve_thresholds,
)
from src.config import PipelineConfig


# ═══════════════════════════════════════════════════════════════════════
#  Severity enum
# ═══════════════════════════════════════════════════════════════════════


class TestSeverity:
    def test_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"

    def test_str(self):
        assert str(Severity.INFO) == "info"
        assert str(Severity.CRITICAL) == "critical"

    def test_from_string(self):
        assert Severity("warning") == Severity.WARNING


# ═══════════════════════════════════════════════════════════════════════
#  ElectrochemicalRule
# ═══════════════════════════════════════════════════════════════════════


class TestElectrochemicalRule:
    def _make(self, **kwargs):
        defaults = dict(
            rule_id="TEST_01",
            category="impedance",
            condition="Rs > 10",
            parameter="Rs",
            interpretation="Test interpretation",
            possible_causes=["cause1"],
            recommendations=["rec1"],
            severity=Severity.WARNING,
            references=["ref1"],
        )
        defaults.update(kwargs)
        return ElectrochemicalRule(**defaults)

    def test_creation(self):
        r = self._make()
        assert r.rule_id == "TEST_01"
        assert r.severity == Severity.WARNING
        assert r.category == "impedance"

    def test_defaults(self):
        r = ElectrochemicalRule()
        assert r.rule_id == ""
        assert r.severity == Severity.INFO
        assert r.possible_causes == []
        assert r.recommendations == []
        assert r.references == []

    def test_to_dict(self):
        r = self._make()
        d = r.to_dict()
        assert d["rule_id"] == "TEST_01"
        assert d["severity"] == "warning"
        assert isinstance(d["possible_causes"], list)

    def test_from_dict(self):
        d = {
            "rule_id": "FROM_DICT",
            "category": "cycling",
            "condition": "retention < 80",
            "parameter": "retention",
            "interpretation": "Test",
            "possible_causes": [],
            "recommendations": ["fix"],
            "severity": "critical",
            "references": [],
        }
        r = ElectrochemicalRule.from_dict(d)
        assert r.rule_id == "FROM_DICT"
        assert r.severity == Severity.CRITICAL
        assert r.category == "cycling"

    def test_from_dict_unknown_severity(self):
        d = {"rule_id": "X", "severity": "unknown_level"}
        r = ElectrochemicalRule.from_dict(d)
        assert r.severity == Severity.INFO  # fallback

    def test_round_trip(self):
        r = self._make()
        d = r.to_dict()
        r2 = ElectrochemicalRule.from_dict(d)
        assert r2.rule_id == r.rule_id
        assert r2.severity == r.severity
        assert r2.condition == r.condition
        assert r2.recommendations == r.recommendations


# ═══════════════════════════════════════════════════════════════════════
#  RuleMatch
# ═══════════════════════════════════════════════════════════════════════


class TestRuleMatch:
    def test_creation(self):
        rule = ElectrochemicalRule(rule_id="X")
        m = RuleMatch(rule=rule, actual_value=42.0, resolved_condition="Rs > 10")
        assert m.actual_value == 42.0
        assert m.resolved_condition == "Rs > 10"

    def test_defaults(self):
        rule = ElectrochemicalRule()
        m = RuleMatch(rule=rule)
        assert m.actual_value is None
        assert m.resolved_condition == ""


# ═══════════════════════════════════════════════════════════════════════
#  Condition evaluator
# ═══════════════════════════════════════════════════════════════════════


class TestResolveThresholds:
    def test_simple(self):
        result = _resolve_thresholds("Rs > {Rs_high}", {"Rs_high": 10.0})
        assert result == "Rs > 10.0"

    def test_multiple(self):
        result = _resolve_thresholds(
            "n < {n_low} and n > {n_min}",
            {"n_low": 0.7, "n_min": 0.3},
        )
        assert "0.7" in result
        assert "0.3" in result

    def test_unresolved_kept(self):
        result = _resolve_thresholds("x > {unknown}", {})
        assert "{unknown}" in result

    def test_no_placeholders(self):
        result = _resolve_thresholds("Rs > 10", {})
        assert result == "Rs > 10"


class TestEvalCondition:
    def test_gt_true(self):
        hit, val = _eval_condition("Rs > 10", {"Rs": 15.0})
        assert hit is True
        assert val == 15.0

    def test_gt_false(self):
        hit, val = _eval_condition("Rs > 10", {"Rs": 5.0})
        assert hit is False
        assert val == 5.0

    def test_lt_true(self):
        hit, _ = _eval_condition("n < 0.5", {"n": 0.3})
        assert hit is True

    def test_gte(self):
        hit, _ = _eval_condition("Rs >= 10", {"Rs": 10.0})
        assert hit is True

    def test_lte(self):
        hit, _ = _eval_condition("n <= 0.5", {"n": 0.5})
        assert hit is True

    def test_eq(self):
        hit, _ = _eval_condition("n_peaks == 1", {"n_peaks": 1.0})
        assert hit is True

    def test_neq(self):
        hit, _ = _eval_condition("n_peaks != 1", {"n_peaks": 2.0})
        assert hit is True

    def test_missing_param(self):
        hit, val = _eval_condition("Rs > 10", {"Rp": 5.0})
        assert hit is False
        assert val is None

    def test_invalid_expression(self):
        hit, val = _eval_condition("not a valid cond", {"Rs": 1})
        assert hit is False

    def test_negative_value(self):
        hit, _ = _eval_condition("Rp > -5", {"Rp": -3.0})
        assert hit is True

    def test_scientific_notation(self):
        hit, _ = _eval_condition("C_mean < 1e-06", {"C_mean": 5e-07})
        assert hit is True


# ═══════════════════════════════════════════════════════════════════════
#  KnowledgeBase — CRUD
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseCRUD:
    def _rule(self, rid="R1", cat="impedance", param="Rs"):
        return ElectrochemicalRule(
            rule_id=rid, category=cat, parameter=param,
            condition=f"{param} > 1", severity=Severity.INFO,
        )

    def test_empty(self):
        kb = KnowledgeBase()
        assert len(kb) == 0
        assert kb.rules == []

    def test_add_rule(self):
        kb = KnowledgeBase()
        kb.add_rule(self._rule("R1"))
        assert len(kb) == 1
        assert "R1" in kb

    def test_add_rules(self):
        kb = KnowledgeBase()
        kb.add_rules([self._rule("R1"), self._rule("R2")])
        assert len(kb) == 2

    def test_get(self):
        kb = KnowledgeBase()
        kb.add_rule(self._rule("R1"))
        assert kb.get("R1") is not None
        assert kb.get("NOPE") is None

    def test_remove(self):
        kb = KnowledgeBase()
        kb.add_rule(self._rule("R1"))
        assert kb.remove_rule("R1") is True
        assert len(kb) == 0
        assert kb.remove_rule("R1") is False

    def test_by_category(self):
        kb = KnowledgeBase()
        kb.add_rules([
            self._rule("R1", cat="impedance"),
            self._rule("R2", cat="cycling"),
            self._rule("R3", cat="impedance"),
        ])
        assert len(kb.by_category("impedance")) == 2
        assert len(kb.by_category("cycling")) == 1
        assert len(kb.by_category("drt")) == 0

    def test_by_category_case_insensitive(self):
        kb = KnowledgeBase()
        kb.add_rule(self._rule("R1", cat="Impedance"))
        assert len(kb.by_category("impedance")) == 1

    def test_by_severity(self):
        kb = KnowledgeBase()
        r1 = self._rule("R1")
        r1.severity = Severity.CRITICAL
        r2 = self._rule("R2")
        r2.severity = Severity.INFO
        kb.add_rules([r1, r2])
        assert len(kb.by_severity(Severity.CRITICAL)) == 1
        assert len(kb.by_severity(Severity.INFO)) == 1

    def test_by_parameter(self):
        kb = KnowledgeBase()
        kb.add_rules([
            self._rule("R1", param="Rs"),
            self._rule("R2", param="Rp"),
            self._rule("R3", param="Rs"),
        ])
        assert len(kb.by_parameter("Rs")) == 2
        assert len(kb.by_parameter("rp")) == 1  # case-insensitive

    def test_categories(self):
        kb = KnowledgeBase()
        kb.add_rules([
            self._rule("R1", cat="drt"),
            self._rule("R2", cat="impedance"),
            self._rule("R3", cat="cycling"),
        ])
        assert kb.categories == ["cycling", "drt", "impedance"]

    def test_init_with_rules(self):
        rules = [self._rule("A"), self._rule("B")]
        kb = KnowledgeBase(rules)
        assert len(kb) == 2

    def test_overwrite_rule(self):
        kb = KnowledgeBase()
        r1 = self._rule("R1")
        r1.interpretation = "old"
        kb.add_rule(r1)
        r1_new = self._rule("R1")
        r1_new.interpretation = "new"
        kb.add_rule(r1_new)
        assert len(kb) == 1
        assert kb.get("R1").interpretation == "new"


# ═══════════════════════════════════════════════════════════════════════
#  KnowledgeBase.evaluate()
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseEvaluate:
    def test_basic_match(self):
        rule = ElectrochemicalRule(
            rule_id="RS_HIGH",
            condition="Rs > 10",
            parameter="Rs",
            severity=Severity.WARNING,
        )
        kb = KnowledgeBase([rule])
        matches = kb.evaluate({"Rs": 15.0})
        assert len(matches) == 1
        assert matches[0].actual_value == 15.0
        assert matches[0].rule.rule_id == "RS_HIGH"

    def test_no_match(self):
        rule = ElectrochemicalRule(
            rule_id="RS_HIGH",
            condition="Rs > 10",
            parameter="Rs",
        )
        kb = KnowledgeBase([rule])
        matches = kb.evaluate({"Rs": 5.0})
        assert len(matches) == 0

    def test_threshold_substitution(self):
        rule = ElectrochemicalRule(
            rule_id="RS_CUSTOM",
            condition="Rs > {Rs_threshold}",
            parameter="Rs",
            severity=Severity.WARNING,
        )
        kb = KnowledgeBase([rule])
        # With threshold = 20, Rs=15 should NOT match
        matches = kb.evaluate({"Rs": 15.0}, {"Rs_threshold": 20.0})
        assert len(matches) == 0
        # With threshold = 10, Rs=15 SHOULD match
        matches = kb.evaluate({"Rs": 15.0}, {"Rs_threshold": 10.0})
        assert len(matches) == 1

    def test_sorted_by_severity(self):
        rules = [
            ElectrochemicalRule(
                rule_id="INFO_R", condition="Rs > 0", parameter="Rs",
                severity=Severity.INFO,
            ),
            ElectrochemicalRule(
                rule_id="CRIT_R", condition="Rs > 0", parameter="Rs",
                severity=Severity.CRITICAL,
            ),
            ElectrochemicalRule(
                rule_id="WARN_R", condition="Rs > 0", parameter="Rs",
                severity=Severity.WARNING,
            ),
        ]
        kb = KnowledgeBase(rules)
        matches = kb.evaluate({"Rs": 5.0})
        assert len(matches) == 3
        assert matches[0].rule.severity == Severity.CRITICAL
        assert matches[1].rule.severity == Severity.WARNING
        assert matches[2].rule.severity == Severity.INFO

    def test_category_filter(self):
        rules = [
            ElectrochemicalRule(
                rule_id="IMP", condition="Rs > 1", parameter="Rs",
                category="impedance",
            ),
            ElectrochemicalRule(
                rule_id="CYC", condition="retention < 80", parameter="retention",
                category="cycling",
            ),
        ]
        kb = KnowledgeBase(rules)
        matches = kb.evaluate(
            {"Rs": 5.0, "retention": 70.0},
            categories=["impedance"],
        )
        assert len(matches) == 1
        assert matches[0].rule.rule_id == "IMP"

    def test_empty_measurements(self):
        rule = ElectrochemicalRule(
            rule_id="X", condition="Rs > 10", parameter="Rs",
        )
        kb = KnowledgeBase([rule])
        assert kb.evaluate({}) == []

    def test_multiple_matches(self):
        rules = [
            ElectrochemicalRule(
                rule_id="A", condition="Rs > 5", parameter="Rs",
                severity=Severity.WARNING,
            ),
            ElectrochemicalRule(
                rule_id="B", condition="Rs > 10", parameter="Rs",
                severity=Severity.CRITICAL,
            ),
        ]
        kb = KnowledgeBase(rules)
        matches = kb.evaluate({"Rs": 15.0})
        assert len(matches) == 2

    def test_resolved_condition_in_match(self):
        rule = ElectrochemicalRule(
            rule_id="T", condition="Rs > {th}", parameter="Rs",
        )
        kb = KnowledgeBase([rule])
        matches = kb.evaluate({"Rs": 5.0}, {"th": 1.0})
        assert matches[0].resolved_condition == "Rs > 1.0"


# ═══════════════════════════════════════════════════════════════════════
#  JSON round-trip
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseJSON:
    def test_round_trip(self, tmp_path):
        rules = [
            ElectrochemicalRule(
                rule_id="A",
                category="impedance",
                condition="Rs > 10",
                parameter="Rs",
                interpretation="High Rs",
                possible_causes=["cause"],
                recommendations=["fix"],
                severity=Severity.WARNING,
                references=["ref"],
            ),
            ElectrochemicalRule(
                rule_id="B",
                category="cycling",
                condition="retention < 60",
                parameter="retention",
                severity=Severity.CRITICAL,
            ),
        ]
        kb = KnowledgeBase(rules)
        path = tmp_path / "test_rules.json"
        kb.to_json(path)
        assert path.exists()

        kb2 = KnowledgeBase.from_json(path)
        assert len(kb2) == 2
        assert "A" in kb2
        assert "B" in kb2
        assert kb2.get("A").severity == Severity.WARNING
        assert kb2.get("B").severity == Severity.CRITICAL
        assert kb2.get("A").recommendations == ["fix"]

    def test_from_json_safe_missing(self, tmp_path):
        kb = KnowledgeBase.from_json_safe(tmp_path / "missing.json")
        assert len(kb) == 0

    def test_from_json_safe_corrupt(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("NOT JSON")
        kb = KnowledgeBase.from_json_safe(p)
        assert len(kb) == 0

    def test_json_file_valid(self, tmp_path):
        rules = [ElectrochemicalRule(rule_id="X", condition="Rs > 1")]
        kb = KnowledgeBase(rules)
        path = tmp_path / "out.json"
        kb.to_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["rule_id"] == "X"


# ═══════════════════════════════════════════════════════════════════════
#  Bundled rules
# ═══════════════════════════════════════════════════════════════════════


class TestBuiltinRules:
    def test_count(self):
        rules = _builtin_rules()
        assert len(rules) >= 50

    def test_all_have_ids(self):
        for r in _builtin_rules():
            assert r.rule_id, f"Rule missing ID: {r}"

    def test_unique_ids(self):
        rules = _builtin_rules()
        ids = [r.rule_id for r in rules]
        assert len(ids) == len(set(ids)), "Duplicate rule IDs found"

    def test_categories_present(self):
        rules = _builtin_rules()
        cats = {r.category for r in rules}
        assert "impedance" in cats
        assert "cycling" in cats
        assert "drt" in cats
        assert "correlation" in cats
        assert "general" in cats

    def test_severities_present(self):
        rules = _builtin_rules()
        sevs = {r.severity for r in rules}
        assert Severity.INFO in sevs
        assert Severity.WARNING in sevs
        assert Severity.CRITICAL in sevs

    def test_all_have_conditions(self):
        for r in _builtin_rules():
            assert r.condition, f"Rule {r.rule_id} missing condition"

    def test_all_have_parameter(self):
        for r in _builtin_rules():
            assert r.parameter, f"Rule {r.rule_id} missing parameter"


# ═══════════════════════════════════════════════════════════════════════
#  KnowledgeBase.default()
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeBaseDefault:
    def test_default_loads(self):
        kb = KnowledgeBase.default()
        assert len(kb) >= 50

    def test_default_evaluate_impedance(self):
        kb = KnowledgeBase.default()
        matches = kb.evaluate({"Rs": 55.0})
        ids = {m.rule.rule_id for m in matches}
        assert "RS_VERY_HIGH" in ids

    def test_default_evaluate_retention(self):
        kb = KnowledgeBase.default()
        matches = kb.evaluate({"retention": 35.0})
        ids = {m.rule.rule_id for m in matches}
        assert "RET_VERY_POOR" in ids
        assert "RET_POOR" in ids

    def test_default_evaluate_n_low(self):
        kb = KnowledgeBase.default()
        matches = kb.evaluate({"n": 0.25})
        ids = {m.rule.rule_id for m in matches}
        assert "N_VERY_LOW" in ids
        assert "N_WARBURG_LIKE" in ids

    def test_default_evaluate_drt(self):
        kb = KnowledgeBase.default()
        matches = kb.evaluate({"n_peaks": 1.0}, categories=["drt"])
        ids = {m.rule.rule_id for m in matches}
        assert "DRT_SINGLE_PEAK" in ids

    def test_default_evaluate_no_match(self):
        kb = KnowledgeBase.default()
        # Very normal values — only INFO-level matches expected
        matches = kb.evaluate({"Rs": 1.5, "Rp": 30.0, "n": 0.88})
        critical = [m for m in matches if m.rule.severity == Severity.CRITICAL]
        assert len(critical) == 0


# ═══════════════════════════════════════════════════════════════════════
#  thresholds_from_config
# ═══════════════════════════════════════════════════════════════════════


class TestThresholdsFromConfig:
    def test_extracts_numeric_fields(self):
        cfg = PipelineConfig.default()
        th = KnowledgeBase.thresholds_from_config(cfg)
        assert "voltage" in th
        assert th["voltage"] == 1.0
        assert "n_head" in th
        assert "drt_lambda" in th

    def test_skips_non_numeric(self):
        cfg = PipelineConfig.default()
        th = KnowledgeBase.thresholds_from_config(cfg)
        assert "data_dir" not in th
        assert "language" not in th

    def test_with_invalid_config(self):
        th = KnowledgeBase.thresholds_from_config(42)  # not a config object
        assert th == {}


# ═══════════════════════════════════════════════════════════════════════
#  Package imports
# ═══════════════════════════════════════════════════════════════════════


class TestPackageImports:
    def test_import_from_ai(self):
        from src.ai import ElectrochemicalRule, KnowledgeBase, RuleMatch, Severity
        assert callable(KnowledgeBase)
        assert callable(ElectrochemicalRule.from_dict)

    def test_import_severity(self):
        from src.ai import Severity
        assert Severity.CRITICAL.value == "critical"


# ═══════════════════════════════════════════════════════════════════════
#  JSON file on disk (data/knowledge/)
# ═══════════════════════════════════════════════════════════════════════


class TestJSONOnDisk:
    def test_json_file_exists(self):
        path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "electrochemistry_rules.json"
        assert path.exists(), f"Bundled rules JSON not found at {path}"

    def test_json_loads_as_list(self):
        path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "electrochemistry_rules.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 50

    def test_json_all_have_rule_id(self):
        path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "electrochemistry_rules.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            assert "rule_id" in item
            assert item["rule_id"]

    def test_json_round_trip_matches_builtin(self):
        path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "electrochemistry_rules.json"
        kb_json = KnowledgeBase.from_json(path)
        kb_builtin = KnowledgeBase(_builtin_rules())
        assert len(kb_json) == len(kb_builtin)
        for rule in kb_builtin.rules:
            assert rule.rule_id in kb_json, f"{rule.rule_id} missing from JSON"
