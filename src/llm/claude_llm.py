"""
src/llm/claude_llm.py — Anthropic Claude API wrapper.

Uses LangChain's ChatAnthropic integration.
API key is read from the environment variable ANTHROPIC_API_KEY.
"""

import logging
from langchain_anthropic import ChatAnthropic

from config import CLAUDE_MODEL, ANTHROPIC_API_KEY, MAX_TOKENS

logger = logging.getLogger(__name__)


def get_claude_llm() -> ChatAnthropic:
    """
    Return a LangChain ChatAnthropic instance.

    Requires ANTHROPIC_API_KEY to be set in .env (or the environment).
    """
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-..."
        )

    logger.info(f"Initialising Claude LLM: {CLAUDE_MODEL}")
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )
    return llm
