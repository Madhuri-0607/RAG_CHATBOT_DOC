"""
src/llm/ollama_llm.py — Ollama local LLM wrapper.

Wraps LangChain's OllamaLLM for use in the QA chain.
Requires Ollama running locally: https://ollama.com
"""

import logging
from langchain_ollama import OllamaLLM

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_TOKENS

logger = logging.getLogger(__name__)


def get_ollama_llm() -> OllamaLLM:
    """
    Return optimized LangChain OllamaLLM for phi3 (fast CPU inference).
    
    Ensure Ollama running: `ollama serve`
    Model: `ollama pull phi3`
    """
    logger.info(f"Initialising Ollama LLM: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,      # deterministic for speed
        num_predict=MAX_TOKENS,
    )
    return llm
