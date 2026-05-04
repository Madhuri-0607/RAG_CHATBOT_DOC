# 📚 RAG Document Q&A Bot

A production-quality **Retrieval-Augmented Generation (RAG)** system that answers questions grounded in your own documents with full source citations. Supports both **local Ollama LLM** (for development) and **Anthropic Claude API** (for production).

Upload your documents once, index them, and ask natural-language questions. Answers are always grounded in your source material with traceable citations.

---

## ✨ Features

- **Multi-format Document Support**: PDF, DOCX, TXT files
- **Efficient Chunking**: Smart text splitting with overlap to preserve context
- **Fast Semantic Search**: FAISS-based vector similarity with sub-millisecond queries
- **Dual LLM Support**: Use local Ollama (free, offline) or Anthropic Claude (production-grade)
- **Source Citations**: Every answer includes exact source document references
- **Multiple Interfaces**: 
  - CLI (command-line with interactive mode)
  - Streamlit Web UI (user-friendly dashboard)
- **Persistent Vector Store**: FAISS index saved to disk for quick re-indexing
- **Configurable**: All parameters centralized in `config.py`

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **RAG Framework** | LangChain (v0.2+) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (persisted to disk) |
| **Local LLM** | Ollama (llama3, phi3, mistral, etc.) |
| **Cloud LLM** | Anthropic Claude API |
| **Document Loaders** | pypdf (PDF), docx2txt (DOCX), built-in (TXT) |
| **CLI** | argparse (built-in) |
| **Web UI** | Streamlit v1.35+ |
| **Config Management** | python-dotenv |

---

## 🔄 How It Works

### Indexing Phase (Run Once)
```
Documents → Load → Chunk (700 chars + 120 overlap) → Embed (384-dim) → FAISS Index
```

### Query Phase (Per Question)
```
User Query → Embed → FAISS Search → Top-9 Chunks → LLM Prompt → Answer + Citations
```

### RAG Pipeline Diagram

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Documents  │────▶│  Ingestion   │────▶│   Chunking    │────▶│  Embedding   │
│ PDF/DOCX/TX │     │  (PyPDF,     │     │ (RecursiveChar│     │ (MiniLM-L6)  │
│     T       │     │  docx2txt)   │     │  Splitter)    │     └──────┬───────┘
└─────────────┘     └──────────────┘     └───────────────┘            │
                                                                       ▼
User Query ──▶ Embed ──▶ Similarity Search ──▶ Retrieve Top-K ──▶ ┌──────────┐
                                                                   │ LLM      │
                                                                   │(Ollama or│
                                                                   │ Claude) │
                                                                   └────┬─────┘
                                                                        │
                                                                        ▼
                                                              Answer + Source Citations
```

---

## 📋 Chunking Strategy

Uses **LangChain's `RecursiveCharacterTextSplitter`** with intelligent fallback order:

1. `\n\n` — Paragraph boundaries (preferred)
2. `\n` — Line breaks
3. `. ` / `! ` / `? ` — Sentence boundaries
4. ` ` — Word boundaries
5. Single characters — Last resort

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 700 chars | Captures complete ideas while remaining precise |
| `chunk_overlap` | 120 chars | Prevents context loss at chunk boundaries |

**Metadata Retained**: source filename, page number, chunk ID, character offset for accurate citations.

---

## 🔍 Embeddings & Vector Store

### Embedding Model: `all-MiniLM-L6-v2`
- **Dimensions**: 384-dimensional dense vectors
- **Size**: ~90 MB (lightweight, CPU-friendly)
- **Performance**: Excellent semantic understanding for mid-to-long text passages
- **Speed**: Sub-millisecond inference per query on CPU

### Vector Store: FAISS
- **Type**: Approximate Nearest Neighbor (ANN) index
- **Persistence**: Saved as `rag_index.faiss` + `rag_index.pkl`
- **Query Performance**: Sub-millisecond retrieval even with thousands of chunks
- **Deployment**: No server needed — fully local
- **Scalability**: Can handle millions of vectors efficiently

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Git**
- **(Optional) Ollama** — Download from [ollama.ai](https://ollama.ai)
- **(Optional) Anthropic API Key** — For Claude-based answers

### Step 1 — Install Python & Clone Repository

```bash
# Ensure Python 3.11+ is installed
python --version

# Clone the repository
git clone <repo-url>
cd rag-doc-bot
```

### Step 2 — Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider: "ollama" (local, free) or "claude" (cloud, API key required)
LLM_PROVIDER=ollama

# Ollama Configuration
OLLAMA_MODEL=phi3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Claude Configuration (only needed if LLM_PROVIDER=claude)
# ANTHROPIC_API_KEY=sk-ant-xxxxx
# CLAUDE_MODEL=claude-3-haiku-20240307

# Tuning Parameters
CHUNK_SIZE=700
CHUNK_OVERLAP=120
TOP_K=9
MAX_TOKENS=400
```

### Step 5 — Add Your Documents

Place PDF, DOCX, or TXT files in the `data/` folder:

```bash
cp /path/to/your/documents/*.pdf data/
cp /path/to/your/documents/*.docx data/
```

### Step 6 — Index Your Documents

Run the indexing process once (or after adding new documents):

```bash
python app.py --index
```

You'll see:
- ✅ Documents loaded
- ✅ Chunks created
- ✅ FAISS index built and saved

### Step 7 — Start Using the Bot

#### CLI Mode (Interactive)
```bash
python app.py
```
You'll be prompted for questions. Type `quit` or `exit` to stop.

#### CLI Mode (Single Query)
```bash
python app.py --query "What is the main topic?"
```

#### Web UI (Streamlit)
```bash
streamlit run streamlit_app.py
```
Opens interactive dashboard at `http://localhost:8501`

---

## 💻 Usage

### Command-Line Interface (CLI)

**Interactive session:**
```bash
python app.py
```

**Single query:**
```bash
python app.py --query "Your question here"
```

**Show retrieval scores (debug mode):**
```bash
python app.py --scores
```

**Rebuild the vector index:**
```bash
python app.py --index
```

### Streamlit Web UI

Start the web interface:
```bash
streamlit run streamlit_app.py
```

Features:
- Visual query interface
- Settings panel (adjust TOP_K, switch LLM)
- Rebuild index button
- Real-time response streaming

---

## ⚙️ Configuration Guide

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Choose `ollama` (local) or `claude` (cloud) |
| `OLLAMA_MODEL` | `phi3:latest` | Model name for local inference |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `CHUNK_SIZE` | `700` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Character overlap between chunks |
| `TOP_K` | `9` | Number of chunks to retrieve per query |
| `MAX_TOKENS` | `400` | Maximum tokens in LLM response |
| `DATA_DIR` | `./data` | Directory containing documents |
| `VECTOR_STORE_DIR` | `./vector_store` | Directory for FAISS index |

Set values via **environment variables** (e.g., `CHUNK_SIZE=800`) or edit `config.py` directly.

---

## 🔄 Switching LLM Providers

### Option 1: Ollama (Local, Free, Offline)

1. **Install Ollama**: [Download](https://ollama.com/download)
2. **Start the server**:
   ```bash
   ollama serve
   ```
3. **Pull a model** (in another terminal):
   ```bash
   ollama pull llama3      # or phi3, mistral, neural-chat, etc.
   ```
4. **Configure in `.env`**:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   OLLAMA_BASE_URL=http://localhost:11434
   ```

**Recommended models:**
- `llama3` — Fast, good quality
- `phi3` — Lightweight, runs on older hardware
- `mistral` — Fast and compact
- `neural-chat` — Good for conversation

### Option 2: Anthropic Claude (Cloud, Production-Ready)

1. **Get an API key**: [console.anthropic.com](https://console.anthropic.com)
2. **Configure in `.env`**:
   ```
   LLM_PROVIDER=claude
   ANTHROPIC_API_KEY=sk-ant-xxxxx
   CLAUDE_MODEL=claude-3-haiku-20240307
   ```
3. **Start querying** — no code changes needed!

**Claude models available:**
- `claude-3-haiku` — Fast, affordable
- `claude-3-sonnet` — Balanced
- `claude-3-opus` — Most powerful

---

## 📚 Example Queries

Try these questions on your documents:

```
1. "What is the main conclusion?"
2. "Summarise the methodology used."
3. "What are the key limitations?"
4. "Who are the authors and their affiliations?"
5. "What data or experiments are described?"
6. "What is the scope and objectives?"
7. "What recommendations are made?"
8. "Compare the findings with previous research."
```

Each answer includes **source citations** with document name and content snippet.

---

## 🧪 Testing & Debugging

### Debug & Development

**Test Ollama connection:**
```bash
python test_ollama.py
```

**View logs:**
Logs are printed to console by default with `INFO` level.

**Enable verbose logging:**
Edit `src/utils.py` and change logging level to `DEBUG`.

---

## 📁 Project Structure

```
rag-doc-bot/
│
├── data/                           # Your documents go here
│   ├── document1.pdf
│   ├── document2.docx
│   └── document3.txt
│
├── vector_store/                   # Auto-created by indexing
│   ├── rag_index.faiss             # FAISS index
│   └── rag_index.pkl               # Metadata & document references
│
├── src/                            # Core RAG logic
│   ├── __init__.py
│   ├── ingestion.py                # Load PDF, DOCX, TXT files
│   ├── chunking.py                 # Text splitting with overlap
│   ├── embedding.py                # Generate embeddings (MiniLM)
│   ├── vector_store.py             # Build & load FAISS index
│   ├── retrieval.py                # Similarity search & ranking
│   ├── qa_chain.py                 # RAG pipeline & prompting
│   ├── utils.py                    # Logging, formatting utilities
│   │
│   └── llm/                        # LLM Provider Abstraction
│       ├── __init__.py
│       ├── ollama_llm.py           # Ollama wrapper
│       ├── claude_llm.py           # Claude API wrapper
│       └── llm_factory.py          # Factory pattern for switching
│
├── app.py                          # CLI entry point
├── streamlit_app.py                # Web UI (Streamlit)
├── test_ollama.py                  # Simple connection test
├── config.py                       # Central configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore
└── README.md                       # This file
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'langchain'`
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Ollama connection failed
**Solution**: Ensure Ollama is running
```bash
ollama serve
# Check: curl http://localhost:11434/api/tags
```

### Issue: FAISS index not found / "No such file or directory"
**Solution**: Rebuild the index
```bash
python app.py --index
```

### Issue: Claude API returns 401 (Unauthorized)
**Solution**: Check your API key
```bash
echo $ANTHROPIC_API_KEY    # Should print your key
# If blank, update .env and restart
```

### Issue: Slow first Ollama query
**Solution**: This is normal — Ollama loads the model into memory on first use. Subsequent queries are faster.

### Issue: "No documents found" after indexing
**Solution**: Ensure documents are in `data/` folder with supported format (PDF, DOCX, TXT)
```bash
ls -la data/    # Check files exist
python app.py --index  # Re-index
```

### Issue: Out of memory (OOM) with large documents
**Solution**: 
- Reduce `CHUNK_SIZE` in `config.py`
- Use a smaller Ollama model (e.g., `phi3` instead of `llama3`)
- Reduce `TOP_K` for fewer retrieved chunks

---

## 🔧 Tuning for Your Use Case

### For High Accuracy (Production):
```env
LLM_PROVIDER=claude
TOP_K=12
CHUNK_SIZE=700
MAX_TOKENS=500
```

### For Fast Responses (Development):
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi3:latest
TOP_K=5
CHUNK_SIZE=500
MAX_TOKENS=200
```

### For Balanced Performance:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
TOP_K=9
CHUNK_SIZE=700
MAX_TOKENS=400
```

---

## 📊 Performance Benchmarks

Tested on a laptop (Intel i7, 16GB RAM, no GPU):

| Operation | Time | Notes |
|---|---|---|
| Index 100 PDFs (~500 pages total) | 2–3 min | One-time operation |
| Embed single query | 50 ms | Using MiniLM-L6-v2 |
| FAISS similarity search (Top-9) | 5 ms | On 5000 chunks |
| Ollama response (phi3) | 2–5 sec | Depends on model & query complexity |
| Claude API response | 1–3 sec | After network latency |

**Total end-to-end query time**: 3–10 seconds (dominated by LLM generation)

---

## 🚀 Advanced Usage

### Rebuilding After Document Changes
```bash
python app.py --index   # Always safe to re-run
```

### Using a Different Embedding Model
Edit `config.py`:
```python
EMBEDDING_MODEL = "all-mpnet-base-v2"  # or any HuggingFace sentence-transformers model
```
Then re-index.

### Viewing Retrieval Scores
```bash
python app.py --scores
```
Shows similarity scores for each retrieved chunk.

### Custom Prompts
Edit `src/qa_chain.py` to modify the system prompt or response format.

---

## 📋 Supported Document Formats

| Format | Status | Notes |
|---|---|---|
| **PDF** | ✅ Supported | Uses PyPDF; preserves text & metadata |
| **DOCX** | ✅ Supported | Uses docx2txt; tables extracted as text |
| **TXT** | ✅ Supported | Plain text files |
| **PPTX** | ⚠️ Partial | Extract as TXT first |
| **HTML** | ⚠️ Partial | Extract as TXT first |
| **Images (OCR)** | ❌ Not supported | Use PDF with embedded text |
| **Audio/Video** | ❌ Not supported | Transcribe first, then index |

---

## 🔐 Privacy & Security

- **Local mode**: All data stays on your machine (Ollama)
- **Claude mode**: Documents are sent to Anthropic API
- **FAISS index**: Stored unencrypted locally in `vector_store/`
- **API keys**: Use `.env` file (never commit to git)

See `.gitignore` for what's excluded from version control.

---

## 📖 Resources

- **LangChain Docs**: [python.langchain.com](https://python.langchain.com)
- **FAISS**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Ollama**: [ollama.com](https://ollama.com)
- **Anthropic Claude**: [console.anthropic.com](https://console.anthropic.com)
- **Sentence-Transformers**: [www.sbert.net](https://www.sbert.net)

---

## 🎓 Learning Resources

This project demonstrates:
- **RAG Architecture**: How to build grounded QA systems
- **Vector Embeddings**: Using semantic search instead of keyword matching
- **LLM Integration**: Switching between local and cloud models
- **LangChain**: Building production RAG pipelines
- **FAISS**: Efficient similarity search at scale

---

## 📝 License

MIT License — Free to use, modify, and distribute.

---

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
1. Open an issue
2. Submit a pull request
3. Suggest improvements

---

## ❓ FAQ

**Q: Can I use this with my own documents?**  
A: Yes! Just add PDFs/DOCX/TXT to `data/` and run `python app.py --index`.

**Q: How many documents can it handle?**  
A: FAISS can handle millions of chunks. Memory is the limiting factor.

**Q: Can I deploy this as a web service?**  
A: Yes! The code is production-ready. Use Flask/FastAPI to expose the QA API, or use Streamlit in production.

**Q: What's the cost with Claude API?**  
A: Pricing varies. Check [console.anthropic.com](https://console.anthropic.com). Ollama is free.

**Q: How do I improve answer quality?**  
A: Try Claude API instead of Ollama, increase `TOP_K`, adjust `CHUNK_SIZE`.

**Q: Can I use different embedding models?**  
A: Yes! Edit `EMBEDDING_MODEL` in `config.py` (must be a HuggingFace sentence-transformers model).

---

**Happy querying! 🚀**
