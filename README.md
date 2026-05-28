# Document Slice Contextual Retrieval

A comprehensive implementation of [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) algorithm optimized for consumer-grade hardware (8GB VRAM). This project demonstrates advanced document chunking, semantic search, and retrieval-augmented generation (RAG) capabilities.

## 📁 Project Structure

### **RnD/** - Research & Development
Core implementation of the contextual retrieval pipeline:

- **`chunking.py`** - Main implementation of document chunking, embedding, and retrieval logic
- **`README.md`** - Detailed technical documentation of the workflow and models used
- **Preprocessing Data:**
  - `preprocessed_chunks/` - Tokenized and chunked documents with metadata
  - `ablation_doc_slice_radius_*.json` - Ablation study results for different slice radius values
- **Benchmarking Results:**
  - `benchmarking_results/` - QA prediction results for different chunking strategies (Anthropic, sliding window, traditional RAG)
  - `ablation_results*.json` - Ablation study metrics and comparisons
- **Generated Data:**
  - `generated_qna.json` - Synthetic QA pairs for benchmarking

### **Showcase/** - Interactive Demonstration
A Chainlit-based web application for interactive RAG queries:

- **`app.py`** - Main application with hybrid search (semantic + full-text), reranking, and chat interface
- **`Dockerfile`** - Containerized deployment configuration
- **`chainlit.md`** - Welcome screen for the web interface
- **`.chainlit/`** - Chainlit configuration and multi-language translations
- **`db/`** - Pre-indexed vector database (LanceDB format) with embedded documents

### **Other Files**
- **`chainlit.md`** - Generic Chainlit welcome documentation
- **`.gitignore`** - Git configuration

## 🔄 Workflow Overview

The system implements a 10-step retrieval pipeline:

1. **PDF Extraction** - Extract content and metadata using Docling
2. **Tokenization** - Chunk content according to embedding model token limits
3. **Contextualization** - Summarize each chunk's role in the document using a local LLM
4. **Augmentation** - Append summaries to chunk endpoints for context preservation
5. **Dense Embedding** - Create semantic vectors for similarity search
6. **Sparse Encoding** - Generate TF-IDF vectors for full-text search
7. **Vector Storage** - Store in LanceDB for hybrid search capabilities
8. **Hybrid Retrieval** - Combined semantic and full-text search with fusion
9. **Reranking** - Reorder results using ColBERT for improved relevance
10. **RAG Output** - Return top-N chunks for language model response generation

## 🧠 Models Used

- **Document Processing:** Docling (DocLayNet, TableFormer, EasyOCR/Tesseract)
- **Tokenization:** nomic-ai/nomic-embed-text-v1.5 (AutoTokenizer)
- **Summarization:** gemma3:4b-it-qat (via Ollama)
- **Embedding:** nomic-ai/nomic-embed-text-v1.5
- **Reranking:** colbert-ir/colbertv2.0

## 💾 Memory Efficiency

- **Max VRAM Used:** 7,695 MiB (~7.7 GB)
- **Optimized for:** 8GB consumer-grade hardware
- **After initial processing:** Minimal VRAM needed for RAG inference

## 🚀 Quick Start

### RnD - Development & Benchmarking
```bash
cd RnD
# Review README.md for detailed workflow documentation
python chunking.py  # Run the chunking pipeline
```

### Showcase - Web Interface
```bash
cd Showcase
# Local deployment
chainlit run app.py

# Docker deployment
docker build -t document-retrieval .
docker run -p 8000:8000 document-retrieval
```

## 📊 Key Features

- **Hybrid Search:** Combines semantic similarity and full-text search
- **Ablation Studies:** Comprehensive comparisons of different chunking strategies
- **Benchmarked:** Tested against traditional RAG approaches
- **Production-Ready:** Containerized with LanceDB persistence
- **Interactive UI:** Chainlit-based web interface for real-time queries
