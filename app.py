# ==============================================================================
# ENTERPRISE PDF → KNOWLEDGE: Simplified Streamlit Application
# Simplified version using local file storage (PDFs/images) and FAISS for vectors.
# No MongoDB or Milvus to avoid crashes.
# ==============================================================================

import streamlit as st
import os
import re
import json
import time
import traceback
import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import zipfile

from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display

# ML / LLM / embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
import logging
import numpy as np
import pickle

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/pdfs", exist_ok=True)
os.makedirs(f"{DATA_DIR}/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/faiss_index", exist_ok=True)

DOCS_FILE = f"{DATA_DIR}/documents.json"
CHUNKS_FILE = f"{DATA_DIR}/chunks.json"  # Not strictly needed, as FAISS holds them
TABLES_FILE = f"{DATA_DIR}/tables.json"
IMAGES_FILE = f"{DATA_DIR}/images.json"

DIMENSION = 384  # For all-MiniLM-L6-v2

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_ingest")

# Streamlit page config
st.set_page_config(
    page_title="📚 Enterprise PDF → Knowledge",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin-bottom: 1rem; }
    .metric { text-align: center; padding: 1rem; background: white; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .status-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 1rem; font-size: 0.9rem; font-weight: bold; margin: 0.25rem; }
    .status-success { background-color: #d4edda; color: #155724; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .section-title { font-size: 1.8rem; font-weight: bold; color: #1f77b4; margin: 1.5rem 0 1rem 0; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem; }
    .citation { background-color: #e7f3ff; padding: 0.5rem; border-left: 3px solid #1f77b4; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOCAL STORAGE HELPER FUNCTIONS
# ==============================================================================

@st.cache_resource
def get_embed_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_json_file(file_path: str) -> List[Dict]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_json_file(file_path: str, data: List[Dict]):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_documents() -> List[Dict]:
    return load_json_file(DOCS_FILE)

def save_document(doc_data: Dict):
    docs = load_documents()
    docs.append(doc_data)
    save_json_file(DOCS_FILE, docs)

def update_document(doc_id: str, updates: Dict):
    docs = load_documents()
    for i, doc in enumerate(docs):
        if doc["doc_id"] == doc_id:
            docs[i].update(updates)
            save_json_file(DOCS_FILE, docs)
            break

def delete_document(doc_id: str):
    docs = load_documents()
    docs = [d for d in docs if d["doc_id"] != doc_id]
    save_json_file(DOCS_FILE, docs)
    # Cascade: remove chunks from FAISS (simplified: rebuild on next load)
    # For tables/images: filter them too
    tables = load_json_file(TABLES_FILE)
    tables = [t for t in tables if t.get("doc_id") != doc_id]
    save_json_file(TABLES_FILE, tables)
    images = load_json_file(IMAGES_FILE)
    images = [img for img in images if img.get("doc_id") != doc_id]
    save_json_file(IMAGES_FILE, images)

def save_pdf_local(doc_id: str, file_bytes: bytes, file_name: str):
    path = f"{DATA_DIR}/pdfs/{file_name}"
    with open(path, 'wb') as f:
        f.write(file_bytes)
    return path

def load_pdf_local(file_path: str) -> bytes:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    return None

def save_image_local(img_id: str, image_bytes: bytes, ext: str = "png"):
    path = f"{DATA_DIR}/images/{img_id}.{ext}"
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return path

def load_image_local(img_path: str) -> bytes:
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            return f.read()
    return None

# ==============================================================================
# FAISS HELPER FUNCTIONS
# ==============================================================================

@st.cache_resource
def get_faiss_vectorstore():
    if os.path.exists(f"{DATA_DIR}/faiss_index/index.faiss") and os.path.exists(f"{DATA_DIR}/faiss_index/index.pkl"):
        try:
            return FAISS.load_local(f"{DATA_DIR}/faiss_index", get_embed_model(), allow_dangerous_deserialization=True)
        except:
            pass
    return None

def initialize_retriever():
    vectorstore = get_faiss_vectorstore()
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )
    return retriever

# ==============================================================================
# SESSION STATE
# ==============================================================================

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# ==============================================================================
# HELPER FUNCTIONS (Kept from original, adapted)
# ==============================================================================

def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'[\ud800-\udfff]', '', s)

def extract_pdf_elements(file_path):
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=4000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=3800,
    )

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    st.image(image_data, use_column_width=True)


QA_PROMPT = """You are an accurate document question-answering assistant.

Use ONLY the provided context to answer the question. Provide bullet points and cite every claim as:

(Document ID: [ID], Page: [X], Section: [Y], Source: [Source]).

If the answer cannot be found in the context, respond EXACTLY: "Answer is not available in the context."

Context:

{context}

Question:

{question}

Answer:
"""

def ask_groq_with_docs(docs, question, model_name="llama3-70b-8192", temperature=0.2, groq_api_key=None):
    parts = []
    for d in docs:
        content = d.page_content
        meta = d.metadata
        header = f"(Document ID: {meta.get('doc_id','NA')}, Page: {meta.get('page','NA')}, Section: {meta.get('section_title','NA')}, Source: {meta.get('source','NA')})"
        snippet = content[:1900] + "..." if len(content) > 1900 else content
        parts.append(header + "\n" + snippet)
    context = "\n\n---\n\n".join(parts)
    prompt = PromptTemplate.from_template(QA_PROMPT)
    formatted = prompt.format(context=context, question=question)
    model = ChatGroq(model_name=model_name, temperature=temperature, api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
    try:
        res = model.invoke(formatted)
        return res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        return f"LLM error: {e}"

def generate_image_summary(image_b64, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": base64.b64decode(image_b64)
        }
    ]
    prompt_parts = [
        image_parts[0],
        prompt
    ]
    response = model.generate_content(prompt_parts)
    return response.text

def summarize_chain():
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.
    Table or text chunk: {element}
    """
    prompt = PromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0, model="llama-3.1-8b-instant")
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

# ==============================================================================
# SIMPLIFIED UI PAGES
# ==============================================================================

def page_upload():
    st.markdown('<h2 class="section-title">📤 Upload & Ingest PDFs</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        department = st.text_input("📁 Department / Tag (optional)", placeholder="e.g., Finance, HR")
    with col2:
        language = st.selectbox("🌐 Language", ["en", "other"])
        use_ocr = st.checkbox("Use OCR for scanned PDFs", value=True)
    uploaded_files = st.file_uploader("📎 Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("🚀 Ingest Selected Files", type="primary"):
        if not uploaded_files:
            st.warning("Please select PDFs to upload.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(uploaded_files)
            for count, f in enumerate(uploaded_files):
                start_time = time.time()
                status_text.text(f"Processing {f.name}... ({count+1}/{total})")
                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    pdf_path = save_pdf_local(doc_id, raw, fname)
                    metadata = {"department": department, "language": language}
                    doc_meta = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "pdf_path": pdf_path,
                        "uploaded_by": "user",
                        "uploaded_at": time.time(),
                        "pages": 0,
                        "toc": [],
                        "status": "processing",
                        "metadata": metadata
                    }
                    save_document(doc_meta)
                    elements = extract_pdf_elements(pdf_path)
                    texts = [el for el in elements if "CompositeElement" in str(type(el))]
                    tables = [el for el in elements if "Table" in str(type(el))]
                    images = get_images_base64(elements)
                    extraction_time = time.time() - start_time
                    chunk_start = time.time()
                    
                    # Generate summaries
                    chain = summarize_chain()
                    text_summaries = chain.batch(texts, {"max_concurrency": 5})
                    table_summaries = chain.batch(tables, {"max_concurrency": 5})
                    image_summaries = [generate_image_summary(img, "Describe the image in detail.") for img in images]

                    # Index summaries
                    retriever = initialize_retriever()
                    image_docs = [Document(page_content=img, metadata={"doc_id": doc_id}) for img in images]
                    retriever.docstore.mset(list(zip([d.metadata["doc_id"] for d in texts + tables + image_docs], texts + tables + image_docs)))
                    summary_docs = [Document(page_content=s, metadata={"doc_id": doc_id, "page_number": i}) for i, s in enumerate(text_summaries + table_summaries + image_summaries)]
                    retriever.vectorstore.add_documents(summary_docs)
                    
                    embedding_time = time.time() - chunk_start
                    # Save tables and images
                    tables_all = load_json_file(TABLES_FILE)
                    for t in tables:
                        t["doc_id"] = doc_id
                        tables_all.append(t)
                    save_json_file(TABLES_FILE, tables_all)
                    images_all = load_json_file(IMAGES_FILE)
                    for img in images:
                        img["doc_id"] = doc_id
                        images_all.append(img)
                    save_json_file(IMAGES_FILE, images_all)
                    # Update doc meta
                    doc_meta.update({
                        "pages": len(texts),
                        "tables": len(tables),
                        "images": len(images),
                        "status": "processed",
                        "extraction_time": extraction_time,
                        "embedding_time": embedding_time
                    })
                    update_document(doc_id, doc_meta)
                    st.success(f"✅ Ingested {f.name}")
                except Exception as e:
                    error_msg = str(e)
                    update_document(doc_id, {"status": "error", "error": error_msg})
                    st.error(f"❌ Failed to process {f.name}: {error_msg}")
                progress_bar.progress((count + 1) / total)
            status_text.empty()
            progress_bar.empty()
            st.balloons()

def page_search():
    st.markdown('<h2 class="section-title">🔍 Search & Q&A</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔎 Enter your search query", placeholder="Ask anything...")
    with col2:
        k = st.slider("Top K Results", 1, 15, 5)
    tab1, tab2 = st.tabs(["🔍 Search", "🤖 Q&A with RAG"])
    with tab1:
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a query.")
            else:
                try:
                    retriever = initialize_retriever()
                    docs = retriever.get_relevant_documents(query, n_results=k)
                    st.session_state.search_results = docs
                    st.success(f"Found {len(docs)} results")
                    for i, d in enumerate(docs):
                        with st.expander(f"📄 Result {i+1}: Document ID: {d.metadata.get('doc_id', 'N/A')} - Page: {d.metadata.get('page_number', 'N/A')}"):
                            try:
                                display_base64_image(d.page_content)
                            except:
                                try:
                                    st.dataframe(pd.read_html(d.page_content)[0])
                                except:
                                    st.write(d.page_content)
                except Exception as e:
                    st.error(f"Search failed: {e}")
    with tab2:
        if st.button("Get Answer (RAG)", type="primary"):
            if not query.strip():
                st.warning("Enter a question.")
            elif not os.environ.get("GROQ_API_KEY"):
                st.error("❌ Set Groq API key in sidebar.")
            else:
                try:
                    retriever = initialize_retriever()
                    docs = retriever.get_relevant_documents(query, n_results=k)
                    with st.spinner("🤔 Thinking..."):
                        ans = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                    st.markdown("### 📝 Answer")
                    st.markdown(ans)
                    citations = re.findall(r'\(Document ID: \[([^\]]+)\], Page: \[([^\]]+)\]', ans)
                    with st.expander("📚 Sources"):
                        for cit in citations:
                            st.markdown(f'<div class="citation">Doc ID: {cit[0]}, Page: {cit[1]}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"QA failed: {e}")

# ==============================================================================
# MAIN APP (Simplified)
# ==============================================================================

def main():
    st.sidebar.title("📚 PDF Knowledge Extractor")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Upload & Ingest", "Search & Q&A"]
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    groq_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="...")

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.sidebar.caption("✅ Groq Key Set")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        st.sidebar.caption("✅ Gemini Key Set")


    st.markdown('<h1 class="main-header">📚 PDF Knowledge Extractor</h1>', unsafe_allow_html=True)

    if page == "Upload & Ingest":
        page_upload()
    elif page == "Search & Q&A":
        page_search()
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #888;">PDF Knowledge Extractor with Groq and Gemini</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
