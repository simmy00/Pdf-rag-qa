# rag/extractor.py
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re

def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """Return list of dicts: [{'text':..., 'page': int, 'filename': filename}, ...]"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        text = clean_text(text)
        pages.append({"text": text, "page": i+1, "filename": filename})
    return pages

def clean_text(s: str) -> str:
    # Basic cleaning: normalize whitespace
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{2,}", "\n\n", s)
    s = s.strip()
    return s

def chunk_page_text(page_text: str, filename: str, page_no: int,
                    chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split page_text into chunks of chunk_size chars with overlap; return list of dicts with metadata."""
    chunks = []
    start = 0
    text_len = len(page_text)
    if text_len == 0:
        return []
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = page_text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_no,
                "filename": filename
            })
        if end == text_len:
            break
        start = end - overlap
    return chunks

def extract_and_chunk(pdf_files: List[Tuple[str, bytes]], chunk_size=1000, overlap=200):
    """
    pdf_files: list of tuples (filename, bytes)
    returns list of chunks: {'text', 'page', 'filename'}
    """
    all_chunks = []
    for filename, pdf_bytes in pdf_files:
        pages = extract_text_from_pdf_bytes(pdf_bytes, filename)
        for p in pages:
            page_chunks = chunk_page_text(p["text"], filename, p["page"],
                                          chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(page_chunks)
    return all_chunks
