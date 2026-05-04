"""
streamlit_app.py — Web UI for the RAG Document Q&A Bot using Streamlit.

Run with:
  streamlit run streamlit_app.py
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional

from src.utils import setup_logging
from config import LLM_PROVIDER, DATA_DIR

# ── Setup logging ──────────────────────────────────────────────────────────────
setup_logging("INFO")
logger = logging.getLogger(__name__)

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("LLM Provider")
    st.metric("Current Provider", LLM_PROVIDER.upper())
    
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=9,
        step=1,
        help="More chunks = more context but slower responses"
    )
    
    st.subheader("Vector Store")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.session_state.rebuild_index = True
    
    with col2:
        if st.button("📄 View Index Stats", use_container_width=True):
            st.session_state.show_stats = True
    
    st.divider()
    st.caption("📂 Data Directory: " + str(DATA_DIR))

# ── Session state initialization ───────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rebuild_index" not in st.session_state:
    st.session_state.rebuild_index = False
if "show_stats" not in st.session_state:
    st.session_state.show_stats = False

# ── Initialize vector store and LLM ────────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    """Load FAISS index (cached)."""
    from src.vector_store import load_vector_store as _load_vector_store
    return _load_vector_store()

@st.cache_resource
def load_llm():
    """Load LLM (cached)."""
    from src.llm.llm_factory import get_llm
    return get_llm()

# ── Rebuild index function ─────────────────────────────────────────────────────
def rebuild_vector_index():
    """Rebuild FAISS index from scratch."""
    from src.ingestion import load_documents
    from src.chunking import chunk_documents
    from src.vector_store import build_vector_store
    
    try:
        with st.spinner("🔄 Loading documents..."):
            documents = load_documents(DATA_DIR)
            st.write(f"✅ Loaded {len(documents)} page(s)/section(s)")
        
        with st.spinner("✂️ Chunking documents..."):
            chunks = chunk_documents(documents)
            st.write(f"✅ Created {len(chunks)} chunks")
        
        with st.spinner("🔢 Building FAISS index..."):
            build_vector_store(chunks)
            st.write(f"✅ FAISS index built successfully")
        
        # Clear cache to reload the vector store
        st.cache_resource.clear()
        st.success("✅ Index rebuilt! Restarting app...")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error rebuilding index: {e}")

# ── Handle index rebuild ───────────────────────────────────────────────────────
if st.session_state.rebuild_index:
    st.session_state.rebuild_index = False
    st.subheader("🔧 Rebuilding Vector Index")
    rebuild_vector_index()

# ── Show index stats ───────────────────────────────────────────────────────────
if st.session_state.show_stats:
    st.session_state.show_stats = False
    try:
        from src.vector_store import load_vector_store as _load_vector_store
        vs = _load_vector_store()
        st.subheader("📊 Vector Store Stats")
        st.info(f"Vector Store Type: FAISS\nDimension: {vs.index.d if hasattr(vs.index, 'd') else 'N/A'}")
    except Exception as e:
        st.warning(f"Could not load stats: {e}")

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Document Q&A Bot")
st.markdown("Chat with your documents using LLM-powered retrieval-augmented generation.")

# ── Load models (with status indicators) ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with st.spinner("Loading vector store..."):
        try:
            st.session_state.vector_store = load_vector_store()
            st.success("✅ Vector store loaded")
        except FileNotFoundError:
            st.error("❌ Vector store not found. Please rebuild the index in Settings.")

with col2:
    with st.spinner("Loading LLM..."):
        try:
            st.session_state.llm = load_llm()
            st.success(f"✅ LLM loaded ({LLM_PROVIDER})")
        except Exception as e:
            st.error(f"❌ Failed to load LLM: {e}")

# ── Check if models are loaded ─────────────────────────────────────────────────
if st.session_state.vector_store is None or st.session_state.llm is None:
    st.warning("⚠️ Models are not loaded. Please rebuild the index or check your configuration.")
    st.stop()

# ── Chat interface ────────────────────────────────────────────────────────────
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• **{source['source']}** | Page {source['page']}")

# Input area
user_question = st.chat_input("Ask a question about the documents...")

if user_question:
    # Add user message to history and display
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })
    
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            from src.qa_chain import answer_question
            
            with st.spinner("🔍 Retrieving relevant documents..."):
                result = answer_question(
                    user_question,
                    st.session_state.vector_store,
                    st.session_state.llm,
                    top_k=top_k
                )
            
            # Display answer
            st.markdown(result.answer)
            
            # Display sources
            if result.sources:
                with st.expander("📚 Sources Retrieved"):
                    for i, source in enumerate(result.sources, 1):
                        st.write(f"**{i}. {source['source']}** — Page {source['page']}")
            
            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources
            })
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            logger.exception("QA error:")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("💡 Tip: Ask specific questions about the documents for better results. Type 'clear' to reset chat history.")

# Clear chat history button
if st.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()
