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

## Source papers excluded for licensing reasons

The benchmark corpus of the article consists of 25 open-access scientific papers. A licence check of every PDF (Crossref metadata, October 2026) showed that five of them may not be redistributed, or not in derived form: four are published under licences that permit reading but not republication, and one is CC BY-NC-ND, which permits sharing the unmodified article but not derivative works such as cleaned text chunks and machine-generated summaries. On the advice of the project supervisor, every copy of those papers' text was removed from the public repository on 2026-10-14. The other 20 papers are CC BY licensed and remain in full.

| Paper | Venue, year | DOI | Licence | Removed from the public repository |
|---|---|---|---|---|
| Machine behaviour (Rahwan et al.) | Nature, 2019 | [10.1038/s41586-019-1138-y](https://doi.org/10.1038/s41586-019-1138-y) | subscription article (Springer Nature text-and-data-mining licence only) | PDF, chunk files, generated contexts, database rows |
| Electron Paramagnetic Resonance Study on 28Si Single Crystal for the Future Realization of the Kilogram | IEEE Transactions on Instrumentation and Measurement, 2019 | [10.1109/TIM.2018.2884044](https://doi.org/10.1109/TIM.2018.2884044) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| Stock Market Prediction via Multi-Source Multiple Instance Learning | IEEE Access, 2018 | [10.1109/ACCESS.2018.2869735](https://doi.org/10.1109/ACCESS.2018.2869735) | IEEE Open Access Publishing Agreement (free to read, no redistribution) | PDF, chunk files, generated contexts, database rows |
| The Graph Database: Jack of All Trades or Just Not SQL? | IT Professional, 2017 | [10.1109/MITP.2017.4241475](https://doi.org/10.1109/MITP.2017.4241475) | subscription article (© IEEE) | PDF, chunk files, generated contexts, database rows |
| The Application of the SOFM Neural Network and Internet of Things in Rural Revitalization | IEEE Access, 2023 | [10.1109/ACCESS.2023.3327131](https://doi.org/10.1109/ACCESS.2023.3327131) | CC BY-NC-ND 4.0 (unmodified copies may be shared, derivatives may not) | chunk files, generated contexts, database rows (the unmodified PDF stays in `RnD/input/`) |

What this means in practice:

- `RnD/input/` holds 21 of the 25 PDFs; `RnD/split_documents/`, `RnD/split_documents/AutoTokenizer/` and `RnD/split_documents_end_to_end/` hold the chunk files of 20 papers; every file under `RnD/preprocessed_chunks/`, the run records under `RnD/dynamic_slice_length/router_experiments/real_runs/`, the gold labels in `RnD/dynamic_slice_length/slice_radius_gold.json`, the older question set `RnD/q_and_a/scientific.json`, and every LanceDB table under `RnD/db/` and `Showcase/db/` contain the records of those 20 papers only (292 of the original 382 chunks).
- Everything that carries no article text is complete: the question sets in `RnD/q_and_a/Gemini/` (they reference chunks by content hash), the per-question retrieval results, the routing records, the energy ledger and the notebooks.
- All numbers, tables and figures in the article were computed on the full 25-paper corpus, so the verification scripts in `RnD/verification/` reproduce the per-corpus statistics only after the missing records are restored.
- To reproduce the experiments end to end, download the five PDFs through the DOIs above (institutional access is needed for the two subscription articles), place them in `RnD/input/` under their original file names, and run `RnD/end_to_end_dynamic_doc_slice_generation.ipynb`, which parses, routes and summarises them again; summaries will not be bit-identical to the archived run. The original chunk and context records of the five papers are available from the authors on reasonable request for verification purposes.

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
