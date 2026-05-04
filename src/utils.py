"""
src/utils.py — Shared helper utilities.
"""

import logging
import sys
from typing import List

from src.qa_chain import QAResult


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "faiss", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Display helpers ───────────────────────────────────────────────────────────

def print_result(result: QAResult, show_scores: bool = False) -> None:
    """Pretty-print a QAResult to the terminal."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"❓ Question: {result.question}")
    print(sep)
    print(f"\n💬 Answer:\n{result.answer}\n")
    print(sep)
    if result.sources:
        print("📚 Sources retrieved:")
        for s in result.sources:
            score_str = f"  [score={s['score']}]" if show_scores else ""
            print(f"  • {s['source']}  |  Page {s['page']}{score_str}")
    print(sep + "\n")


def print_banner() -> None:
    """Print startup banner."""
    banner = r"""
  ____      _     ____  ___   ____        _
 |  _ \ ___| |_  |  _ \|__ \ |  _ \  ___| |_
 | |_) / _ \ __| | | | | / / | |_) |/ _ \ __|
 |  _ <  __/ |_  | |_| |/ /_ |  _ <  __/ |_
 |_| \_\___|\__| |____/|____||_| \_\___|\__|

     Document Q&A Bot  •  RAG  •  LangChain
    """
    print(banner)
