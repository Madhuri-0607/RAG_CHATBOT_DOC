"""
src/retrieval.py — Semantic chunk retrieval from the vector store.

Given a natural-language query, returns the top-k most relevant chunks
along with their metadata (source file, page, chunk_id).
"""

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import TOP_K

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    vector_store: FAISS,
    top_k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Perform a similarity search and return (Document, score) pairs.

    Args:
        query:        User's natural-language question.
        vector_store: Loaded FAISS index.
        top_k:        Number of chunks to return.

    Returns:
        List of (Document, relevance_score) tuples, highest score first.
        Score is cosine similarity (0–1 range after normalization).
    """
    logger.info(f"Retrieving top-{top_k} chunks for query: '{query}'")

    results: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_score(query, k=top_k)
    )

    if not results:
        logger.warning("No relevant chunks found for the query.")

    for i, (doc, score) in enumerate(results):
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        logger.debug(f"  [{i+1}] score={score:.4f}  {src} (page {page})")

    return results


def format_context(results: List[Tuple[Document, float]]) -> str:
    """
    Combine retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labelled with its source and page so the model can cite them.
    """
    parts = []
    for i, (doc, _score) in enumerate(results):
        src   = doc.metadata.get("source", "unknown")
        page  = doc.metadata.get("page", "N/A")
        chunk = doc.metadata.get("chunk_id", i)
        header = f"[Source {i+1}: {src} | Page {page} | Chunk {chunk}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
