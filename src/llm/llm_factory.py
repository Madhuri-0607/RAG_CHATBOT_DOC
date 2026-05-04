"""
src/llm/llm_factory.py — LLM provider factory.

Single entry point: get_llm() reads config.LLM_PROVIDER and returns
the correct LLM instance.  Adding a new provider = adding one branch.
"""

import logging
from config import LLM_PROVIDER

logger = logging.getLogger(__name__)


def get_llm():
    """
    Factory function — returns the configured LLM.

    Supported providers (set LLM_PROVIDER in config.py or .env):
      "ollama"  → local Ollama model (development / offline)
      "claude"  → Anthropic Claude API (demo / production) - fallback if credits low
    """
    provider = LLM_PROVIDER.strip().lower()
    logger.info(f"LLM provider selected: '{provider}'")

    if provider == "ollama":
        from src.llm.ollama_llm import get_ollama_llm
        return get_ollama_llm()

    elif provider == "claude":
        try:
            from src.llm.claude_llm import get_claude_llm
            return get_claude_llm()
        except Exception as e:
            logger.warning(f"Claude init failed (check credits/key): {str(e)[:100]}. Falling back to ollama (free local).")
            from src.llm.ollama_llm import get_ollama_llm
            return get_ollama_llm()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'.\n"
            "Supported values: 'ollama', 'claude'"
        )
