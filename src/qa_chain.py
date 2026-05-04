"""
src/qa_chain.py — Core RAG question-answering logic.

Ties together: retrieval → prompt construction → LLM call → citation formatting.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import TOP_K
from src.retrieval import retrieve_chunks, format_context

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are an intelligent RAG (Retrieval-Augmented Generation) chatbot.

Follow these rules strictly:

1. Answer ONLY using the provided context.

2. Do NOT make up or assume information.

3. If the answer is not clearly present in the context, say:
   "I couldn't find a clear answer in the provided documents."

4. If the user's question is vague, ambiguous, or unclear (e.g., "tell me about the system", "explain it", "what is this"):

   * DO NOT guess
   * Ask a clarification question instead

5. If multiple interpretations are possible:

   * Provide 2–4 possible meanings
   * Ask the user to choose

6. Keep answers simple, clear, and concise.

7. Prioritize accuracy over completeness.

Format:

Context:
{context}

Question:
{question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)   # [{source, page, chunk_id}]
    raw_chunks: List[Tuple[Document, float]] = field(default_factory=list)


# ── Main QA function ──────────────────────────────────────────────────────────

def answer_question(
    question: str,
    vector_store,
    llm,
    top_k: int = 9,  # Retrieve from multiple documents
) -> QAResult:
    """
    Full RAG pipeline for a single question.

    Steps:
      1. Retrieve top-k relevant chunks from the vector store.
      2. Format them into a context string with source labels.
      3. Build the prompt and call the LLM.
      4. Parse and return the answer with citations.

    Args:
        question:     User question string.
        vector_store: Loaded FAISS index.
        llm:          LangChain-compatible LLM (Ollama or Claude).
        top_k:        How many chunks to retrieve.

    Returns:
        QAResult with answer text and structured source list.
    """
    # Step 1 — Retrieve
    chunks = retrieve_chunks(question, vector_store, top_k=top_k)
    if not chunks:
        return QAResult(
            question=question,
            answer="Answer not found in the provided documents.",
            sources=[],
        )

    # Step 2 — Build context
    context = format_context(chunks)

    # Step 3 — Build prompt string
    prompt_text = RAG_PROMPT.format(context=context, question=question)

    # Step 4 — Call LLM
    logger.info("Calling LLM …")
    llm_start = time.time()
    try:
        response = llm.invoke(prompt_text)
        llm_time = time.time() - llm_start
        logger.info(f"LLM inference: {llm_time:.2f}s")
        # ChatAnthropic returns an AIMessage; Ollama returns a string
        answer_text = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return QAResult(
            question=question,
            answer="LLM error - likely Claude credits low. Switched to free Ollama (install: ollama.com, run 'ollama serve && ollama pull phi3'). Details: " + str(e)[:200],
            sources=[],
        )

    # Step 5 — Build source list
    sources = []
    seen = set()
    for doc, _score in chunks:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":   doc.metadata.get("source", "unknown"),
                "page":     doc.metadata.get("page", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "score":    round(float(_score), 4),
            })

    return QAResult(
        question=question,
        answer=answer_text.strip(),
        sources=sources,
        raw_chunks=chunks,
    )
