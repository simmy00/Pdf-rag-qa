# streamlit_app.py

import streamlit as st
from dotenv import load_dotenv
from rag.extractor import extract_and_chunk
from rag.embeddings_index import FAISSIndex
from rag.generator import GeminiGenerator
from utils.helpers import highlight_terms, top_query_terms

# Load environment variables
load_dotenv()

st.set_page_config(page_title="PDF RAG Q&A", layout="wide")
st.title("RAG-based PDF Q&A — Prototype (Gemini Edition)")

# -------------------
# Build index from PDFs
# -------------------
@st.cache_data(show_spinner=False)
def build_index_from_uploads(uploaded_files, chunk_size, overlap):
    pdfs = []
    for uf in uploaded_files:
        content = uf.read()
        pdfs.append((uf.name, content))

    chunks = extract_and_chunk(pdfs, chunk_size=chunk_size, overlap=overlap)
    index = FAISSIndex()
    if len(chunks) == 0:
        return index, []
    index.build(chunks)
    return index, chunks

# Sidebar upload + build
st.sidebar.header("Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload one or more PDFs", accept_multiple_files=True, type=["pdf"])
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)

if uploaded:
    st.sidebar.success(f"{len(uploaded)} file(s) ready")
    if st.sidebar.button("Build index from uploads"):
        with st.spinner("Extracting and building index..."):
            index, all_chunks = build_index_from_uploads(uploaded, chunk_size, overlap)
        st.session_state["index_built"] = True
        st.session_state["index"] = index
        st.session_state["chunks"] = all_chunks
        st.success("Index built! You can now ask questions.")
else:
    st.sidebar.info("Upload PDF(s) and build index.")

if "index_built" not in st.session_state:
    st.session_state["index_built"] = False

# -------------------
# Initialize Gemini Generator (No caching to avoid unhashable errors)
# -------------------
if st.session_state.get("index_built", False):
    generator = GeminiGenerator(index=st.session_state["index"])
else:
    generator = None

# -------------------
# Ask a question
# -------------------
st.header("Ask a question")
query = st.text_input("Enter your question:", key="query_input")
top_k = st.slider("Number of context chunks to retrieve", min_value=1, max_value=10, value=4)

if st.button("Get Answer") and query.strip():
    if not st.session_state["index_built"]:
        st.warning("Please upload PDFs and build the index first from the sidebar.")
    else:
        index: FAISSIndex = st.session_state["index"]
        results = index.query(query, top_k=top_k)

        if not results:
            st.info("No relevant content found in the uploaded PDFs.")
        else:
            scores, metadatas = zip(*results)
            contexts = [m["text"] for m in metadatas]

            with st.spinner("Generating answer using Gemini LLM..."):
                answer = generator.generate(query, top_k=top_k)

            # Display answer
            st.subheader("Answer")
            st.write(answer)

            # Show retrieved sources
            st.subheader("Retrieved source chunks")
            query_terms = top_query_terms(query, n=6)
            for i, (score, meta) in enumerate(results, start=1):
                st.markdown(f"**{i}. File:** `{meta['filename']}` — page {meta['page']} — score: {score:.3f}")
                highlighted = highlight_terms(meta['text'][:2000], query_terms)
                st.markdown(highlighted, unsafe_allow_html=True)
                if st.button(f"Show full chunk {i}"):
                    st.text_area("Full chunk text", value=meta['text'], height=250)

st.markdown("---")
st.markdown("**Notes:** This demo uses FAISS for semantic search and Gemini for answer generation. Upload PDFs, build the index, and ask questions!")
