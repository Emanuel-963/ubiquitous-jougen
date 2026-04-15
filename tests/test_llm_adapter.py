"""Tests for the LLM integration layer (src/ai/llm_adapter.py).

Day 20 — covers all adapters, config, factory, and enrichment helpers
using mocks (no real API calls).
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

from src.ai.llm_adapter import (
    LLMAdapter,
    LLMConfig,
    LLMProvider,
    NullAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    create_adapter,
    create_adapter_from_config,
    enrich_report,
    enrich_summary,
)


# ═══════════════════════════════════════════════════════════════════════
#  LLMProvider enum
# ═══════════════════════════════════════════════════════════════════════

class TestLLMProvider(unittest.TestCase):
    """Tests for the LLMProvider enum."""

    def test_values(self):
        self.assertEqual(LLMProvider.NONE, "none")
        self.assertEqual(LLMProvider.OPENAI, "openai")
        self.assertEqual(LLMProvider.OLLAMA, "ollama")

    def test_str_representation(self):
        self.assertEqual(str(LLMProvider.OPENAI), "openai")

    def test_from_string(self):
        self.assertIs(LLMProvider("none"), LLMProvider.NONE)
        self.assertIs(LLMProvider("openai"), LLMProvider.OPENAI)
        self.assertIs(LLMProvider("ollama"), LLMProvider.OLLAMA)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            LLMProvider("unknown_provider")


# ═══════════════════════════════════════════════════════════════════════
#  LLMConfig
# ═══════════════════════════════════════════════════════════════════════

class TestLLMConfig(unittest.TestCase):
    """Tests for the LLMConfig dataclass."""

    def test_defaults(self):
        cfg = LLMConfig()
        self.assertEqual(cfg.provider, "none")
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.api_key, "")
        self.assertEqual(cfg.base_url, "")
        self.assertAlmostEqual(cfg.temperature, 0.3)
        self.assertEqual(cfg.max_tokens, 1024)
        self.assertAlmostEqual(cfg.timeout, 30.0)
        self.assertIn("electrochemist", cfg.system_prompt)

    def test_resolved_provider_valid(self):
        cfg = LLMConfig(provider="openai")
        self.assertIs(cfg.resolved_provider, LLMProvider.OPENAI)

    def test_resolved_provider_case_insensitive(self):
        cfg = LLMConfig(provider="  OpenAI  ")
        self.assertIs(cfg.resolved_provider, LLMProvider.OPENAI)

    def test_resolved_provider_unknown_falls_back(self):
        cfg = LLMConfig(provider="anthropic")
        self.assertIs(cfg.resolved_provider, LLMProvider.NONE)

    def test_resolved_base_url_explicit(self):
        cfg = LLMConfig(provider="openai", base_url="https://custom.api/v1")
        self.assertEqual(cfg.resolved_base_url, "https://custom.api/v1")

    def test_resolved_base_url_ollama_default(self):
        cfg = LLMConfig(provider="ollama")
        self.assertEqual(cfg.resolved_base_url, "http://localhost:11434")

    def test_resolved_base_url_none_provider(self):
        cfg = LLMConfig(provider="none")
        self.assertEqual(cfg.resolved_base_url, "")

    def test_is_enabled_false_for_none(self):
        cfg = LLMConfig(provider="none")
        self.assertFalse(cfg.is_enabled)

    def test_is_enabled_true_for_openai(self):
        cfg = LLMConfig(provider="openai")
        self.assertTrue(cfg.is_enabled)

    def test_is_enabled_true_for_ollama(self):
        cfg = LLMConfig(provider="ollama")
        self.assertTrue(cfg.is_enabled)

    def test_to_dict_excludes_api_key(self):
        cfg = LLMConfig(provider="openai", api_key="sk-secret")
        d = cfg.to_dict()
        self.assertNotIn("api_key", d)
        self.assertEqual(d["provider"], "openai")
        self.assertEqual(d["model"], "gpt-4o-mini")

    def test_to_dict_includes_expected_keys(self):
        d = LLMConfig().to_dict()
        expected_keys = {"provider", "model", "base_url", "temperature", "max_tokens", "timeout"}
        self.assertEqual(set(d.keys()), expected_keys)


# ═══════════════════════════════════════════════════════════════════════
#  LLMAdapter ABC
# ═══════════════════════════════════════════════════════════════════════

class TestLLMAdapterABC(unittest.TestCase):
    """Tests for the abstract base class."""

    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            LLMAdapter()

    def test_provider_name(self):
        adapter = NullAdapter()
        self.assertEqual(adapter.provider_name, "NullAdapter")

    def test_default_is_available(self):
        adapter = NullAdapter()
        self.assertTrue(adapter.is_available)

    def test_config_defaults_when_none(self):
        adapter = NullAdapter(config=None)
        self.assertEqual(adapter.config.provider, "none")


# ═══════════════════════════════════════════════════════════════════════
#  NullAdapter
# ═══════════════════════════════════════════════════════════════════════

class TestNullAdapter(unittest.TestCase):
    """Tests for the NullAdapter (offline fallback)."""

    def setUp(self):
        self.adapter = NullAdapter()

    def test_interpret_returns_message(self):
        result = self.adapter.interpret("context", "question")
        self.assertIn("not configured", result)
        self.assertIsInstance(result, str)

    def test_enrich_summary_returns_original(self):
        original = "The sample shows high impedance."
        result = self.adapter.enrich_summary(original, "context data")
        self.assertEqual(result, original)

    def test_suggest_experiments_returns_hint(self):
        result = self.adapter.suggest_experiments("context")
        self.assertIn("LLM provider", result)

    def test_compare_with_literature_returns_hint(self):
        result = self.adapter.compare_with_literature("context")
        self.assertIn("LLM provider", result)

    def test_is_available_always_true(self):
        self.assertTrue(self.adapter.is_available)

    def test_provider_name(self):
        self.assertEqual(self.adapter.provider_name, "NullAdapter")


# ═══════════════════════════════════════════════════════════════════════
#  OpenAIAdapter
# ═══════════════════════════════════════════════════════════════════════

class TestOpenAIAdapter(unittest.TestCase):
    """Tests for the OpenAI adapter using mocks."""

    def setUp(self):
        self.config = LLMConfig(
            provider="openai",
            api_key="sk-test-key-12345",
            model="gpt-4o-mini",
        )
        self.adapter = OpenAIAdapter(self.config)

    def test_is_available_with_key(self):
        self.assertTrue(self.adapter.is_available)

    def test_is_available_without_key(self):
        adapter = OpenAIAdapter(LLMConfig(provider="openai", api_key=""))
        self.assertFalse(adapter.is_available)

    def test_provider_name(self):
        self.assertEqual(self.adapter.provider_name, "OpenAIAdapter")

    @patch("src.ai.llm_adapter.OpenAIAdapter._chat")
    def test_interpret_calls_chat(self, mock_chat):
        mock_chat.return_value = "Rs indicates high ohmic resistance."
        result = self.adapter.interpret("eis data...", "What does Rs mean?")
        mock_chat.assert_called_once()
        self.assertIn("Rs", result)

    @patch("src.ai.llm_adapter.OpenAIAdapter._chat")
    def test_enrich_summary(self, mock_chat):
        mock_chat.return_value = "Enhanced summary with physical mechanisms."
        result = self.adapter.enrich_summary("Basic summary.", "full context")
        mock_chat.assert_called_once()
        self.assertIn("Enhanced", result)

    @patch("src.ai.llm_adapter.OpenAIAdapter._chat")
    def test_suggest_experiments(self, mock_chat):
        mock_chat.return_value = "1. Test at higher concentration"
        result = self.adapter.suggest_experiments("context")
        mock_chat.assert_called_once()
        self.assertIn("concentration", result)

    @patch("src.ai.llm_adapter.OpenAIAdapter._chat")
    def test_compare_with_literature(self, mock_chat):
        mock_chat.return_value = "Published values for Rs are typically 1-10 Ω."
        result = self.adapter.compare_with_literature("context")
        mock_chat.assert_called_once()
        self.assertIn("Published", result)

    def test_get_client_import_error(self):
        """If openai package is not installed, raise ImportError."""
        with patch.dict("sys.modules", {"openai": None}):
            adapter = OpenAIAdapter(self.config)
            adapter._client = None  # Force re-init
            with self.assertRaises(ImportError):
                adapter._get_client()

    @patch("src.ai.llm_adapter.OpenAIAdapter._get_client")
    def test_chat_api_error_returns_error_string(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API limit")
        mock_get_client.return_value = mock_client
        result = self.adapter._chat("test prompt")
        self.assertIn("OpenAI error", result)
        self.assertIn("API limit", result)

    @patch("src.ai.llm_adapter.OpenAIAdapter._get_client")
    def test_chat_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Great answer  "
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_get_client.return_value = mock_client
        result = self.adapter._chat("test prompt")
        self.assertEqual(result, "Great answer")

    def test_chat_uses_system_prompt(self):
        """Verify the system prompt is included in the request."""
        with patch.object(self.adapter, "_get_client") as mock_gc:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "answer"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            mock_gc.return_value = mock_client

            self.adapter._chat("hello")

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("electrochemist", messages[0]["content"])


# ═══════════════════════════════════════════════════════════════════════
#  OllamaAdapter
# ═══════════════════════════════════════════════════════════════════════

class TestOllamaAdapter(unittest.TestCase):
    """Tests for the Ollama adapter using mocks."""

    def setUp(self):
        self.config = LLMConfig(provider="ollama", model="llama3")
        self.adapter = OllamaAdapter(self.config)

    def test_provider_name(self):
        self.assertEqual(self.adapter.provider_name, "OllamaAdapter")

    @patch("src.ai.llm_adapter.OllamaAdapter._chat")
    def test_interpret(self, mock_chat):
        mock_chat.return_value = "The impedance shows a clear semicircle."
        result = self.adapter.interpret("context", "What does this mean?")
        mock_chat.assert_called_once()
        self.assertIn("semicircle", result)

    @patch("src.ai.llm_adapter.OllamaAdapter._chat")
    def test_enrich_summary(self, mock_chat):
        mock_chat.return_value = "Enriched from Ollama."
        result = self.adapter.enrich_summary("original summary", "ctx")
        self.assertEqual(result, "Enriched from Ollama.")

    @patch("src.ai.llm_adapter.OllamaAdapter._chat")
    def test_suggest_experiments(self, mock_chat):
        mock_chat.return_value = "Try higher scan rates."
        result = self.adapter.suggest_experiments("ctx")
        self.assertIn("scan rates", result)

    @patch("src.ai.llm_adapter.OllamaAdapter._chat")
    def test_compare_with_literature(self, mock_chat):
        mock_chat.return_value = "Typical Rs for NiO is 5 Ω."
        result = self.adapter.compare_with_literature("ctx")
        self.assertIn("Rs", result)

    @patch("urllib.request.urlopen")
    def test_is_available_true(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = lambda s: MagicMock()
        mock_urlopen.return_value.__exit__ = lambda s, *a: None
        self.assertTrue(self.adapter.is_available)

    @patch("urllib.request.urlopen")
    def test_is_available_false_on_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        self.assertFalse(self.adapter.is_available)

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen):
        response_body = json.dumps({
            "message": {"content": "  Ollama says hello  "}
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp

        result = self.adapter._chat("test prompt")
        self.assertEqual(result, "Ollama says hello")

    @patch("urllib.request.urlopen")
    def test_chat_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = self.adapter._chat("prompt")
        self.assertIn("Ollama error", result)

    @patch("urllib.request.urlopen")
    def test_chat_json_error(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp
        result = self.adapter._chat("prompt")
        self.assertIn("Ollama error", result)

    @patch("urllib.request.urlopen")
    def test_chat_missing_content_key(self, mock_urlopen):
        response_body = json.dumps({"status": "ok"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp
        result = self.adapter._chat("prompt")
        self.assertEqual(result, "")

    @patch("urllib.request.urlopen")
    def test_chat_sends_correct_payload(self, mock_urlopen):
        response_body = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = lambda s, *a: None
        mock_urlopen.return_value = mock_resp

        self.adapter._chat("my prompt")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertIn("/api/chat", req.full_url)
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["messages"][1]["content"], "my prompt")
        self.assertFalse(payload["stream"])


# ═══════════════════════════════════════════════════════════════════════
#  Factory: create_adapter
# ═══════════════════════════════════════════════════════════════════════

class TestCreateAdapter(unittest.TestCase):
    """Tests for the create_adapter factory function."""

    def test_none_config_returns_null(self):
        adapter = create_adapter(None)
        self.assertIsInstance(adapter, NullAdapter)

    def test_none_provider_returns_null(self):
        adapter = create_adapter(LLMConfig(provider="none"))
        self.assertIsInstance(adapter, NullAdapter)

    def test_openai_provider(self):
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        adapter = create_adapter(cfg)
        self.assertIsInstance(adapter, OpenAIAdapter)

    def test_ollama_provider(self):
        cfg = LLMConfig(provider="ollama")
        adapter = create_adapter(cfg)
        self.assertIsInstance(adapter, OllamaAdapter)

    def test_unknown_provider_returns_null(self):
        cfg = LLMConfig(provider="claude")
        adapter = create_adapter(cfg)
        self.assertIsInstance(adapter, NullAdapter)

    def test_adapter_receives_config(self):
        cfg = LLMConfig(provider="openai", api_key="sk-test", model="gpt-4")
        adapter = create_adapter(cfg)
        self.assertEqual(adapter.config.model, "gpt-4")


# ═══════════════════════════════════════════════════════════════════════
#  Factory: create_adapter_from_config
# ═══════════════════════════════════════════════════════════════════════

class TestCreateAdapterFromConfig(unittest.TestCase):
    """Tests for creating adapters from PipelineConfig-like objects."""

    def test_no_llm_attributes_returns_null(self):
        pc = SimpleNamespace()
        adapter = create_adapter_from_config(pc)
        self.assertIsInstance(adapter, NullAdapter)

    def test_openai_from_pipeline_config(self):
        pc = SimpleNamespace(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_api_key="sk-abc",
            llm_base_url="",
            llm_temperature=0.5,
            llm_max_tokens=2048,
        )
        adapter = create_adapter_from_config(pc)
        self.assertIsInstance(adapter, OpenAIAdapter)
        self.assertEqual(adapter.config.model, "gpt-4")
        self.assertAlmostEqual(adapter.config.temperature, 0.5)

    def test_ollama_from_pipeline_config(self):
        pc = SimpleNamespace(
            llm_provider="ollama",
            llm_model="llama3",
        )
        adapter = create_adapter_from_config(pc)
        self.assertIsInstance(adapter, OllamaAdapter)
        self.assertEqual(adapter.config.model, "llama3")

    def test_partial_attributes_use_defaults(self):
        pc = SimpleNamespace(llm_provider="openai", llm_api_key="sk-test")
        adapter = create_adapter_from_config(pc)
        self.assertIsInstance(adapter, OpenAIAdapter)
        self.assertEqual(adapter.config.model, "gpt-4o-mini")  # default
        self.assertEqual(adapter.config.max_tokens, 1024)  # default


# ═══════════════════════════════════════════════════════════════════════
#  enrich_report helper
# ═══════════════════════════════════════════════════════════════════════

class TestEnrichReport(unittest.TestCase):
    """Tests for the enrich_report helper function."""

    def test_null_adapter_returns_unchanged(self):
        original = "Full report text."
        result = enrich_report(original, adapter=NullAdapter())
        self.assertEqual(result, original)

    def test_none_adapter_returns_unchanged(self):
        result = enrich_report("report", adapter=None)
        self.assertEqual(result, "report")

    def test_enrichment_appends_literature(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.compare_with_literature.return_value = "Papers show Rs ~ 5 Ω."
        adapter.suggest_experiments.return_value = "Try 2M H2SO4."
        # Ensure it's not treated as NullAdapter
        adapter.__class__ = OpenAIAdapter

        result = enrich_report("report text", adapter=adapter)
        self.assertIn("report text", result)
        self.assertIn("Literature Comparison", result)
        self.assertIn("Papers show", result)
        self.assertIn("Suggested Experiments", result)
        self.assertIn("H2SO4", result)

    def test_enrichment_only_literature(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.compare_with_literature.return_value = "Literature says..."
        adapter.__class__ = OpenAIAdapter

        result = enrich_report("base", adapter=adapter, sections=["literature"])
        self.assertIn("Literature", result)
        self.assertNotIn("Experiments", result)

    def test_enrichment_only_experiments(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.suggest_experiments.return_value = "Experiment X"
        adapter.__class__ = OpenAIAdapter

        result = enrich_report("base", adapter=adapter, sections=["experiments"])
        self.assertNotIn("Literature", result)
        self.assertIn("Experiments", result)

    def test_error_in_llm_graceful(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.compare_with_literature.side_effect = RuntimeError("API down")
        adapter.suggest_experiments.side_effect = RuntimeError("API down")
        adapter.__class__ = OpenAIAdapter

        result = enrich_report("base report", adapter=adapter)
        self.assertEqual(result, "base report")

    def test_error_string_skipped(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.compare_with_literature.return_value = "[OpenAI error: timeout]"
        adapter.suggest_experiments.return_value = "[Ollama error: down]"
        adapter.__class__ = OpenAIAdapter

        result = enrich_report("base report", adapter=adapter)
        self.assertEqual(result, "base report")


# ═══════════════════════════════════════════════════════════════════════
#  enrich_summary helper
# ═══════════════════════════════════════════════════════════════════════

class TestEnrichSummary(unittest.TestCase):
    """Tests for the enrich_summary helper function."""

    def test_null_adapter_returns_original(self):
        result = enrich_summary("original", adapter=NullAdapter())
        self.assertEqual(result, "original")

    def test_none_adapter_returns_original(self):
        result = enrich_summary("original", adapter=None)
        self.assertEqual(result, "original")

    def test_llm_enriches_summary(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.enrich_summary.return_value = "A natural rewrite."
        adapter.__class__ = OpenAIAdapter

        result = enrich_summary("basic summary", context="ctx", adapter=adapter)
        self.assertEqual(result, "A natural rewrite.")

    def test_error_returns_original(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.enrich_summary.side_effect = RuntimeError("down")
        adapter.__class__ = OpenAIAdapter

        result = enrich_summary("original", adapter=adapter)
        self.assertEqual(result, "original")

    def test_error_string_returns_original(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.enrich_summary.return_value = "[OpenAI error: limit]"
        adapter.__class__ = OpenAIAdapter

        result = enrich_summary("original", adapter=adapter)
        self.assertEqual(result, "original")

    def test_empty_response_returns_original(self):
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.enrich_summary.return_value = ""
        adapter.__class__ = OpenAIAdapter

        result = enrich_summary("original", adapter=adapter)
        self.assertEqual(result, "original")


# ═══════════════════════════════════════════════════════════════════════
#  Integration-level: full adapter + enrichment flow
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationFlowNull(unittest.TestCase):
    """End-to-end flow with NullAdapter (offline)."""

    def test_full_flow_no_change(self):
        adapter = create_adapter(None)
        report = "📊 Executive Summary\n──\nSample analysis..."
        enriched_report = enrich_report(report, adapter=adapter)
        enriched_summ = enrich_summary("Sample is good.", adapter=adapter)
        self.assertEqual(enriched_report, report)
        self.assertEqual(enriched_summ, "Sample is good.")


class TestIntegrationFlowOpenAI(unittest.TestCase):
    """End-to-end flow with mocked OpenAI adapter."""

    @patch("src.ai.llm_adapter.OpenAIAdapter._chat")
    def test_full_enrichment(self, mock_chat):
        mock_chat.side_effect = [
            "Literature shows Rs ~5 Ω for NiO in H2SO4.",
            "Try EIS at 80°C for accelerated aging.",
        ]
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        adapter = create_adapter(cfg)
        report = "Some base report."
        enriched = enrich_report(report, adapter=adapter)
        self.assertIn("Some base report.", enriched)
        self.assertIn("Literature", enriched)
        self.assertIn("Experiments", enriched)
        self.assertEqual(mock_chat.call_count, 2)


class TestIntegrationFlowOllama(unittest.TestCase):
    """End-to-end flow with mocked Ollama adapter."""

    @patch("src.ai.llm_adapter.OllamaAdapter._chat")
    def test_full_enrichment(self, mock_chat):
        mock_chat.side_effect = [
            "Rs values compare well with literature.",
            "Experiment at higher pH.",
        ]
        cfg = LLMConfig(provider="ollama", model="llama3")
        adapter = create_adapter(cfg)
        enriched = enrich_report("base", adapter=adapter)
        self.assertIn("Literature", enriched)
        self.assertIn("Experiment", enriched)


# ═══════════════════════════════════════════════════════════════════════
#  Edge cases & config integration
# ═══════════════════════════════════════════════════════════════════════

class TestLLMConfigEdgeCases(unittest.TestCase):
    """Edge cases for config resolution."""

    def test_empty_string_provider(self):
        cfg = LLMConfig(provider="")
        self.assertIs(cfg.resolved_provider, LLMProvider.NONE)

    def test_whitespace_provider(self):
        cfg = LLMConfig(provider="   ")
        self.assertIs(cfg.resolved_provider, LLMProvider.NONE)

    def test_ollama_explicit_base_url(self):
        cfg = LLMConfig(provider="ollama", base_url="http://gpu-server:11434")
        self.assertEqual(cfg.resolved_base_url, "http://gpu-server:11434")


class TestAdapterWithCustomConfig(unittest.TestCase):
    """Verify adapters properly use config fields."""

    @patch("src.ai.llm_adapter.OpenAIAdapter._get_client")
    def test_openai_uses_model_and_temperature(self, mock_gc):
        cfg = LLMConfig(
            provider="openai", api_key="sk-x", model="gpt-4",
            temperature=0.7, max_tokens=512,
        )
        adapter = OpenAIAdapter(cfg)

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "reply"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_gc.return_value = mock_client

        adapter._chat("hello")

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model"), "gpt-4")
        self.assertAlmostEqual(call_kwargs.kwargs.get("temperature"), 0.7)
        self.assertEqual(call_kwargs.kwargs.get("max_tokens"), 512)


class TestNullAdapterWithConfig(unittest.TestCase):
    """NullAdapter should accept and ignore config."""

    def test_accepts_config(self):
        cfg = LLMConfig(provider="none", model="ignored")
        adapter = NullAdapter(cfg)
        self.assertEqual(adapter.config.model, "ignored")
        # But still returns static text
        self.assertIn("not configured", adapter.interpret("c", "q"))


if __name__ == "__main__":
    unittest.main()
