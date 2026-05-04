"""
config.py — Central configuration for the RAG Document Q&A Bot.
All tuneable parameters live here. No hardcoded values elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Switch between "ollama" (local) and "claude" (Anthropic API)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "claude" (Claude needs credits)

# ── Model names ───────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")          # or "llama3
# CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # HuggingFace sentence-transformers

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 9))   # number of chunks to retrieve per query (increased for multi-document coverage)

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 400))  # limit output length

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")

# ── Anthropic ─────────────────────────────────────────────────────────────────
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
