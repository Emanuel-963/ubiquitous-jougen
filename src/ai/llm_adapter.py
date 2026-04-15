"""LLM integration layer — optional, modular adapters for AI enrichment.

This module provides a pluggable architecture for **optionally** routing
text through a Large Language Model.  Three concrete adapters ship out of
the box:

* :class:`NullAdapter` — no-op / rule-based fallback (default).  The
  system works 100 % offline with this adapter.
* :class:`OpenAIAdapter` — wraps the OpenAI chat-completions API.
* :class:`OllamaAdapter` — wraps a locally running Ollama server.

Every adapter inherits from the :class:`LLMAdapter` ABC and satisfies
the same contract, so calling code never cares which backend is active.

Factory
-------
Use :func:`create_adapter` (or :func:`create_adapter_from_config`) to
obtain the right adapter from an :class:`LLMConfig` or a
:class:`~src.config.PipelineConfig`.

Enrichment helpers
------------------
:func:`enrich_report` and :func:`enrich_summary` accept pre-built text
from the rule-based engine and optionally enhance it via the active LLM.

Usage::

    from src.ai.llm_adapter import create_adapter, LLMConfig, enrich_report
    cfg = LLMConfig(provider="openai", api_key="sk-…")
    adapter = create_adapter(cfg)
    enriched = enrich_report(original_report, adapter=adapter)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Provider enum
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(str, Enum):
    """Supported LLM provider identifiers."""

    NONE = "none"
    OPENAI = "openai"
    OLLAMA = "ollama"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# ═══════════════════════════════════════════════════════════════════════
#  Configuration dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """Configuration for the LLM integration layer.

    Attributes
    ----------
    provider : str
        One of ``"none"``, ``"openai"``, ``"ollama"``.
    model : str
        Model identifier (e.g. ``"gpt-4o-mini"``, ``"llama3"``).
    api_key : str
        API key for cloud providers.  Unused for ``"ollama"`` / ``"none"``.
    base_url : str
        Base URL override.  Ollama default: ``"http://localhost:11434"``.
    temperature : float
        Sampling temperature (0 = deterministic).
    max_tokens : int
        Maximum response tokens.
    timeout : float
        Request timeout in seconds.
    system_prompt : str
        System prompt prepended to every request.
    """

    provider: str = "none"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: float = 30.0
    system_prompt: str = (
        "You are an expert electrochemist and materials scientist. "
        "Interpret impedance spectroscopy (EIS), cycling, and DRT results. "
        "Be concise, precise, and cite physical mechanisms."
    )

    # ── Helpers ───────────────────────────────────────────────────

    @property
    def resolved_provider(self) -> LLMProvider:
        """Return the normalised :class:`LLMProvider` enum value."""
        try:
            return LLMProvider(self.provider.lower().strip())
        except ValueError:
            logger.warning("Unknown LLM provider '%s' → falling back to none", self.provider)
            return LLMProvider.NONE

    @property
    def resolved_base_url(self) -> str:
        """Return a base URL, filling in the Ollama default when empty."""
        if self.base_url:
            return self.base_url
        if self.resolved_provider == LLMProvider.OLLAMA:
            return "http://localhost:11434"
        return ""

    @property
    def is_enabled(self) -> bool:
        """True if a real LLM backend is configured (not 'none')."""
        return self.resolved_provider != LLMProvider.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict (excludes api_key for safety)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Abstract base class
# ═══════════════════════════════════════════════════════════════════════

class LLMAdapter(ABC):
    """Abstract interface for LLM-backed text generation.

    Every concrete adapter **must** implement:

    * :meth:`interpret` — open-ended question → answer.
    * :meth:`enrich_summary` — take rule-based summary, return richer text.
    * :meth:`suggest_experiments` — suggest follow-up experiments.
    * :meth:`compare_with_literature` — contextualise results vs. literature.

    The adapter also exposes :attr:`is_available` so the GUI can grey-out
    buttons when the backend is not reachable.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    # ── Required abstract methods ────────────────────────────────

    @abstractmethod
    def interpret(self, context: str, question: str) -> str:
        """Answer a free-form *question* given a data *context*.

        Parameters
        ----------
        context : str
            Textual representation of the analysis results.
        question : str
            The user's question.

        Returns
        -------
        str
            The adapter's answer.
        """

    @abstractmethod
    def enrich_summary(self, summary: str, context: str) -> str:
        """Rewrite *summary* using information from *context*.

        Parameters
        ----------
        summary : str
            The rule-based executive summary.
        context : str
            Full analysis report for reference.

        Returns
        -------
        str
            A more natural, enriched summary.
        """

    @abstractmethod
    def suggest_experiments(self, context: str) -> str:
        """Propose follow-up experiments based on *context*.

        Returns
        -------
        str
            Bullet-list of suggested experiments.
        """

    @abstractmethod
    def compare_with_literature(self, context: str) -> str:
        """Compare results in *context* with published literature.

        Returns
        -------
        str
            A brief comparison paragraph.
        """

    # ── Common interface ─────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Human-readable name of this adapter's backend."""
        return self.__class__.__name__

    @property
    def is_available(self) -> bool:
        """True if the backend is reachable / properly configured.

        Base implementation returns True; subclasses may override.
        """
        return True


# ═══════════════════════════════════════════════════════════════════════
#  NullAdapter — offline fallback (default)
# ═══════════════════════════════════════════════════════════════════════

class NullAdapter(LLMAdapter):
    """Pass-through adapter that uses no external LLM.

    This is the default when no LLM is configured.  It returns the
    rule-based text unchanged or a polite "not available" message,
    keeping the system fully functional offline.
    """

    def interpret(self, context: str, question: str) -> str:  # noqa: D401
        """Return a static message (no LLM available)."""
        return (
            "LLM interpretation is not configured. "
            "The rule-based analysis above contains all available insights."
        )

    def enrich_summary(self, summary: str, context: str) -> str:
        """Return the original *summary* unchanged."""
        return summary

    def suggest_experiments(self, context: str) -> str:
        """Return a generic hint pointing the user to the rule-based report."""
        return (
            "Enable an LLM provider (OpenAI or Ollama) for AI-generated "
            "experiment suggestions.  See Settings → AI → Provider."
        )

    def compare_with_literature(self, context: str) -> str:
        """Return a static unavailable message."""
        return (
            "Literature comparison requires an LLM provider.  "
            "Configure one in Settings → AI → Provider."
        )


# ═══════════════════════════════════════════════════════════════════════
#  OpenAIAdapter — cloud-based (requires API key)
# ═══════════════════════════════════════════════════════════════════════

class OpenAIAdapter(LLMAdapter):
    """Adapter for the OpenAI chat-completions API.

    Requires the ``openai`` package (optional dependency) and a valid
    ``api_key`` in :attr:`config`.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        super().__init__(config)
        self._client: Any = None

    # ── Lazy client init ─────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazily initialise the OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIAdapter.  "
                "Install it with:  pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        kwargs["timeout"] = self.config.timeout
        self._client = openai.OpenAI(**kwargs)
        return self._client

    # ── Core chat call ───────────────────────────────────────────

    def _chat(self, user_message: str) -> str:
        """Send a single user message and return the assistant reply."""
        client = self._get_client()
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            return f"[OpenAI error: {exc}]"

    # ── Abstract method implementations ──────────────────────────

    def interpret(self, context: str, question: str) -> str:
        prompt = (
            f"Based on the following electrochemical analysis:\n\n"
            f"{context}\n\n"
            f"Answer this question:\n{question}"
        )
        return self._chat(prompt)

    def enrich_summary(self, summary: str, context: str) -> str:
        prompt = (
            f"Here is a rule-based executive summary of an electrochemical "
            f"analysis:\n\n{summary}\n\nFull analysis context:\n{context}\n\n"
            f"Rewrite the summary in a more natural and insightful way. "
            f"Keep it to one paragraph, mention physical mechanisms."
        )
        return self._chat(prompt)

    def suggest_experiments(self, context: str) -> str:
        prompt = (
            f"Based on this electrochemical analysis:\n\n{context}\n\n"
            f"Suggest 3-5 follow-up experiments that would help improve "
            f"the material performance or deepen understanding. "
            f"Be specific about conditions (concentrations, scan rates, "
            f"temperatures, etc.)."
        )
        return self._chat(prompt)

    def compare_with_literature(self, context: str) -> str:
        prompt = (
            f"Based on this electrochemical analysis:\n\n{context}\n\n"
            f"Compare these results with published literature on similar "
            f"electrode materials and electrolyte systems.  Mention typical "
            f"reported values for Rs, Rp, capacitance, and cycling retention."
        )
        return self._chat(prompt)

    @property
    def is_available(self) -> bool:
        """True when an API key is configured."""
        return bool(self.config.api_key)


# ═══════════════════════════════════════════════════════════════════════
#  OllamaAdapter — local LLM (no API key needed)
# ═══════════════════════════════════════════════════════════════════════

class OllamaAdapter(LLMAdapter):
    """Adapter for a locally running `Ollama <https://ollama.ai>`_ server.

    Communicates via the ``/api/chat`` REST endpoint.
    No API key required — the server runs on localhost.
    """

    # ── Core HTTP call ───────────────────────────────────────────

    def _chat(self, user_message: str) -> str:
        """Send a chat request to the Ollama server."""
        import urllib.request
        import urllib.error

        base = self.config.resolved_base_url
        url = f"{base}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as exc:
            logger.error("Ollama connection error: %s", exc)
            return f"[Ollama error: {exc}]"
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"[Ollama error: {exc}]"

    # ── Abstract method implementations ──────────────────────────

    def interpret(self, context: str, question: str) -> str:
        prompt = (
            f"Based on the following electrochemical analysis:\n\n"
            f"{context}\n\n"
            f"Answer this question:\n{question}"
        )
        return self._chat(prompt)

    def enrich_summary(self, summary: str, context: str) -> str:
        prompt = (
            f"Here is a rule-based executive summary of an electrochemical "
            f"analysis:\n\n{summary}\n\nFull analysis context:\n{context}\n\n"
            f"Rewrite the summary in a more natural and insightful way. "
            f"Keep it to one paragraph, mention physical mechanisms."
        )
        return self._chat(prompt)

    def suggest_experiments(self, context: str) -> str:
        prompt = (
            f"Based on this electrochemical analysis:\n\n{context}\n\n"
            f"Suggest 3-5 follow-up experiments that would help improve "
            f"the material performance or deepen understanding. "
            f"Be specific about conditions."
        )
        return self._chat(prompt)

    def compare_with_literature(self, context: str) -> str:
        prompt = (
            f"Based on this electrochemical analysis:\n\n{context}\n\n"
            f"Compare these results with published literature on similar "
            f"electrode materials and electrolyte systems."
        )
        return self._chat(prompt)

    @property
    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        import urllib.request
        import urllib.error

        base = self.config.resolved_base_url
        try:
            req = urllib.request.Request(f"{base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
#  Factory functions
# ═══════════════════════════════════════════════════════════════════════

def create_adapter(config: Optional[LLMConfig] = None) -> LLMAdapter:
    """Create the appropriate adapter for *config*.

    Parameters
    ----------
    config : LLMConfig | None
        When ``None`` or provider ``"none"``, returns a :class:`NullAdapter`.

    Returns
    -------
    LLMAdapter
    """
    if config is None:
        return NullAdapter()

    provider = config.resolved_provider

    if provider == LLMProvider.OPENAI:
        return OpenAIAdapter(config)
    if provider == LLMProvider.OLLAMA:
        return OllamaAdapter(config)

    return NullAdapter(config)


def create_adapter_from_config(pipeline_config: Any) -> LLMAdapter:
    """Create an adapter using fields from a :class:`PipelineConfig`.

    Reads ``llm_provider``, ``llm_model``, ``llm_api_key``,
    ``llm_base_url``, ``llm_temperature``, ``llm_max_tokens`` from the
    pipeline config if present; otherwise falls back to defaults.
    """
    llm_cfg = LLMConfig(
        provider=getattr(pipeline_config, "llm_provider", "none"),
        model=getattr(pipeline_config, "llm_model", "gpt-4o-mini"),
        api_key=getattr(pipeline_config, "llm_api_key", ""),
        base_url=getattr(pipeline_config, "llm_base_url", ""),
        temperature=getattr(pipeline_config, "llm_temperature", 0.3),
        max_tokens=getattr(pipeline_config, "llm_max_tokens", 1024),
    )
    return create_adapter(llm_cfg)


# ═══════════════════════════════════════════════════════════════════════
#  Enrichment helpers
# ═══════════════════════════════════════════════════════════════════════

def enrich_report(
    report_text: str,
    *,
    adapter: Optional[LLMAdapter] = None,
    sections: Optional[List[str]] = None,
) -> str:
    """Optionally enrich a pre-built report via the active LLM.

    Parameters
    ----------
    report_text : str
        The full formatted report from :func:`run_ai_analysis`.
    adapter : LLMAdapter | None
        The adapter to use.  ``None`` or ``NullAdapter`` → return unchanged.
    sections : list of str, optional
        Which enrichment sections to append.  Defaults to
        ``["literature", "experiments"]``.

    Returns
    -------
    str
        The (possibly enriched) report text.
    """
    if adapter is None or isinstance(adapter, NullAdapter):
        return report_text

    if sections is None:
        sections = ["literature", "experiments"]

    sep = "─" * 48
    parts: List[str] = [report_text]

    if "literature" in sections:
        try:
            lit = adapter.compare_with_literature(report_text)
            if lit and not lit.startswith("["):
                parts.append("")
                parts.append("📚 Literature Comparison (AI)")
                parts.append(sep)
                parts.append(lit)
        except Exception as exc:
            logger.warning("LLM literature comparison failed: %s", exc)

    if "experiments" in sections:
        try:
            exp = adapter.suggest_experiments(report_text)
            if exp and not exp.startswith("["):
                parts.append("")
                parts.append("🧪 Suggested Experiments (AI)")
                parts.append(sep)
                parts.append(exp)
        except Exception as exc:
            logger.warning("LLM experiment suggestions failed: %s", exc)

    return "\n".join(parts)


def enrich_summary(
    summary: str,
    *,
    context: str = "",
    adapter: Optional[LLMAdapter] = None,
) -> str:
    """Optionally rewrite the executive summary using the LLM.

    If the adapter is ``None`` or a ``NullAdapter``, returns *summary*
    unchanged.
    """
    if adapter is None or isinstance(adapter, NullAdapter):
        return summary

    try:
        enriched = adapter.enrich_summary(summary, context)
        if enriched and not enriched.startswith("["):
            return enriched
    except Exception as exc:
        logger.warning("LLM summary enrichment failed: %s", exc)

    return summary
