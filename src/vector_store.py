"""
src/vector_store.py — FAISS vector database (persisted to disk).

Two public functions:
  build_vector_store(chunks)  → embed & save
  load_vector_store()         → load from disk for querying
"""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import VECTOR_STORE_DIR
from src.embedding import get_embeddings

logger = logging.getLogger(__name__)

INDEX_NAME = "rag_index"   # FAISS saves two files: <name>.faiss + <name>.pkl


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed all chunks and persist the FAISS index to disk.

    Args:
        chunks: Chunked Documents (with metadata) from chunking.py

    Returns:
        Loaded FAISS vector store object.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    logger.info(f"Building FAISS index for {len(chunks)} chunks …")
    embeddings = get_embeddings()

    # FAISS.from_documents embeds in batches automatically
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    vector_store.save_local(VECTOR_STORE_DIR, index_name=INDEX_NAME)
    logger.info(f"FAISS index saved → {VECTOR_STORE_DIR}/{INDEX_NAME}.*")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load a previously built FAISS index from disk.

    Raises:
        FileNotFoundError: if the index files are missing (run indexing first).
    """
    faiss_file = os.path.join(VECTOR_STORE_DIR, f"{INDEX_NAME}.faiss")
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_file}.\n"
            "Run indexing first:  python app.py --index"
        )

    logger.info(f"Loading FAISS index from {VECTOR_STORE_DIR} …")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,   # required by LangChain ≥0.1
    )
    logger.info("FAISS index loaded ✓")
    return vector_store
