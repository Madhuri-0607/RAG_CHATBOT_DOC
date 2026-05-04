"""
src/embedding.py — HuggingFace embedding wrapper.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, high quality).
Returns a LangChain-compatible embedding object used by the vector store.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — avoids reloading the model on every call
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    The model is downloaded once from HuggingFace Hub and cached locally
    by the sentence-transformers library (~90 MB for all-MiniLM-L6-v2).
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},       # switch to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity friendly
        )
        logger.info("Embedding model loaded ✓")
    return _embeddings_instance
