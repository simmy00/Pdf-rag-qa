# PDF RAG Q&A — Prototype

Simple Retrieval-Augmented-Generation (RAG) demo that answers user questions using uploaded PDF documents.

## Features
- Multi-PDF upload
- Text extraction per page (PyMuPDF)
- Chunking with overlap
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS
- LLM generation: `google/flan-t5-small` (local) — replaceable with hosted LLMs
- Streamlit UI showing retrieved chunks and the final answer

## Setup (Linux / macOS / WSL)
1. Clone repo:
```bash
git clone https://github.com/simmy00/Pdf-rag-qa/
cd pdf-rag-qa
