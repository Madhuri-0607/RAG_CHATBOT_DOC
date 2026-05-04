"""
src/ingestion.py — Document loading & text extraction.

Supports: PDF, DOCX, TXT
Preserves metadata: source filename + page number where available.
"""

import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pdf(filepath: str) -> List[Document]:
    """Load a PDF using PyPDFLoader; each page becomes one Document."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    # Ensure source metadata is the bare filename
    for doc in pages:
        doc.metadata["source"] = os.path.basename(filepath)
    return pages


def _load_docx(filepath: str) -> List[Document]:
    """Load a .docx file; returns a single Document (no native page info)."""
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(filepath)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(filepath)
        doc.metadata.setdefault("page", 0)
    return docs


def _load_txt(filepath: str) -> List[Document]:
    """Load a plain-text file."""
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(filepath)
        doc.metadata.setdefault("page", 0)
    return docs


# ── Public API ────────────────────────────────────────────────────────────────

LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".txt":  _load_txt,
}


def load_documents(data_dir: str) -> List[Document]:
    """
    Recursively scan *data_dir* for supported files and load them.

    Returns a flat list of LangChain Document objects with metadata:
      - source: filename
      - page:   page index (0-based) or 0 for non-paginated formats
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_docs: List[Document] = []
    found_files = []

    for ext, loader_fn in LOADERS.items():
        for filepath in data_path.rglob(f"*{ext}"):
            found_files.append(filepath)
            logger.info(f"Loading: {filepath.name}")
            try:
                docs = loader_fn(str(filepath))
                all_docs.extend(docs)
                logger.info(f"  → {len(docs)} page(s)/section(s) loaded")
            except Exception as e:
                logger.error(f"  ✗ Failed to load {filepath.name}: {e}")

    if not found_files:
        raise ValueError(
            f"No supported files (.pdf, .docx, .txt) found in: {data_dir}"
        )

    logger.info(f"\nTotal documents loaded: {len(all_docs)} from {len(found_files)} files")
    return all_docs
