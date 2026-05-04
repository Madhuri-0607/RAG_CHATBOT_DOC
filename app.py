"""
app.py — Entry point for the RAG Document Q&A Bot.

Usage:
  # 1. Index documents (run once, or after adding new files to /data)
  python app.py --index

  # 2. Start interactive Q&A session
  python app.py

  # 3. Ask a single question non-interactively
  python app.py --query "What is the main topic of the documents?"

  # 4. Show retrieval scores
  python app.py --scores
"""

import argparse
import sys
import time

from src.utils import setup_logging, print_result, print_banner
from config import LLM_PROVIDER, DATA_DIR


def run_indexing() -> None:
    """Ingest → chunk → embed → save FAISS index."""
    from src.ingestion import load_documents
    from src.chunking import chunk_documents
    from src.vector_store import build_vector_store

    print(f"\n📂 Loading documents from: {DATA_DIR}")
    documents = load_documents(DATA_DIR)
    print(f"   Loaded {len(documents)} page(s)/section(s)\n")

    print("✂️  Chunking documents …")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks\n")

    print("🔢 Embedding & building FAISS index …")
    build_vector_store(chunks)
    print("\n✅ Indexing complete! You can now run queries.\n")


def run_qa_session(single_query: str = None, show_scores: bool = False) -> None:
    """Load index + LLM and run interactive or single-query mode."""
    from src.vector_store import load_vector_store
    from src.llm.llm_factory import get_llm
    from src.qa_chain import answer_question

    print(f"\n⚙️  LLM Provider: {LLM_PROVIDER.upper()}")
    print("🔄 Loading vector store …")
    vector_store = load_vector_store()

    print("🤖 Loading LLM …")
    llm = get_llm()

    print_banner()

    if single_query:
        result = answer_question(single_query, vector_store, llm)
        print_result(result, show_scores=show_scores)
        return

    # ── Interactive loop ───────────────────────────────────────────────────────
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("🔍 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        try:
            result = answer_question(question, vector_store, llm)
            print_result(result, show_scores=show_scores)
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")


# ── CLI argument parsing ───────────────────────────────────────────────────────

def main() -> None:
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="RAG Document Q&A Bot — LangChain + FAISS + HuggingFace"
    )
    parser.add_argument(
        "--index", action="store_true",
        help="Ingest documents and build the vector index (run first)."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Ask a single question and exit."
    )
    parser.add_argument(
        "--scores", action="store_true",
        help="Show retrieval similarity scores alongside sources."
    )
    args = parser.parse_args()

    if args.index:
        run_indexing()
    else:
        try:
            run_qa_session(single_query=args.query, show_scores=args.scores)
        except FileNotFoundError as e:
            print(f"\n❌ {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
