"""
src/chunking.py — Text splitting strategy.

Uses RecursiveCharacterTextSplitter (LangChain) which tries to split
on paragraphs → sentences → words before falling back to characters.
Metadata (source, page) is preserved on every chunk.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller overlapping chunks.

    Args:
        documents:     Raw documents from the ingestion step.
        chunk_size:    Maximum characters per chunk (default from config).
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        List of chunk Documents, each carrying the parent's metadata
        plus a `chunk_id` field for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on meaningful boundaries first
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,   # records char offset in metadata
    )

    chunks = splitter.split_documents(documents)

    # Add a human-readable chunk_id to every chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    logger.info(
        f"Chunking complete: {len(documents)} docs → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
