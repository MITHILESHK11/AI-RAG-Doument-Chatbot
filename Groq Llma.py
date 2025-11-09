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

# Extraction libs
import pdfplumber
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image

# ML / LLM / embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import pandas as pd
import uuid
import logging
import pickle

# DeepSeek OCR
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

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

# DeepSeek OCR Sampling Params
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
    ),
    skip_special_tokens=False,
)

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

@st.cache_resource
def get_ocr_model():
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor]
    )
    return llm

def ocr_image(llm, image: Image.Image) -> str:
    prompt = "<image>\nFree OCR."
    model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
    outputs = llm.generate(model_input, SAMPLING_PARAMS)
    return outputs[0].outputs[0].text.strip()

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

def build_or_load_faiss(documents: List[Document]):
    vectorstore = get_faiss_vectorstore()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, get_embed_model())
    else:
        vectorstore.add_documents(documents)
    vectorstore.save_local(f"{DATA_DIR}/faiss_index")
    return vectorstore

def search_faiss(query: str, k: int):
    vectorstore = get_faiss_vectorstore()
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)

# ==============================================================================
# SESSION STATE
# ==============================================================================

if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None
if 'ocr_llm' not in st.session_state:
    st.session_state.ocr_llm = get_ocr_model()

# ==============================================================================
# HELPER FUNCTIONS (Kept from original, adapted)
# ==============================================================================

def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'[\ud800-\udfff]', '', s)

def extract_text_pdf(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[str], List[Dict]]:
    pages = []
    toc = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(safe_text(txt))
        try:
            reader = PdfReader(BytesIO(file_bytes))
            outline = reader.outline
            toc = []
            for item in outline:
                if hasattr(item, 'title') and hasattr(item, 'page'):
                    # PyPDF2 pages are 0-indexed, so add 1 for user display
                    toc.append({"title": str(item.title), "page": item.page + 1})
        except:
            toc = []
    except:
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = [safe_text(page.extract_text() or "") for page in reader.pages]
        except:
            pages = []
    if use_ocr_if_empty and any(not p.strip() for p in pages):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            llm = st.session_state.ocr_llm
            for i in range(len(doc)):
                if pages[i].strip():
                    continue
                page = doc.load_page(i)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = ocr_image(llm, img)
                pages[i] = safe_text(ocr_text)
        except:
            pass
    cleaned_pages = [postprocess_extracted_text(p) for p in pages]
    return cleaned_pages, toc

def dehyphenate(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def postprocess_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\r', '\n')
    text = dehyphenate(text)
    text = re.sub(r'[\x0c\x0b]', '', text)
    return text

def extract_tables_pdf(file_bytes: bytes) -> List[Dict]:
    tables_all = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t in tables:
                    if not t or len(t) < 2:
                        continue
                    header = [safe_text(c) for c in t[0]]
                    rows = []
                    for r in t[1:]:
                        rowd = {header[j] if j < len(header) else f"col_{j}": safe_text(cell) for j, cell in enumerate(r) if cell is not None}
                        rows.append(rowd)
                    tables_all.append({"page": i, "header": header, "rows": rows})
    except:
        pass
    return tables_all

def extract_images_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    imgs = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        llm = st.session_state.ocr_llm
        for page_i in range(len(doc)):
            page = doc[page_i]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_name = f"{doc_id}_p{page_i+1}_img{img_index}"
                img_path = save_image_local(img_name, image_bytes, ext)
                if img_path:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = ocr_image(llm, pil)
                    if not caption:
                        caption = "Image (no OCR text) — visual content"
                    imgs.append({"img_path": img_path, "page": page_i + 1, "caption": caption, "id": f"img_{doc_id}_{img_index}"})
    except:
        pass
    return imgs

def detect_headings_in_page(page_text: str) -> List[Tuple[int, str]]:
    headings = []
    lines = page_text.splitlines()
    pos = 0
    for line in lines:
        s = line.strip()
        if not s:
            pos += len(line) + 1
            continue
        is_allcaps = (len(s) > 2 and s.upper() == s and sum(c.isalpha() for c in s) >= 3)
        is_short = len(s) < 60 and (s.endswith(':') or re.match(r'^\d+(\.\d+)*\s', s))
        if is_allcaps or is_short:
            headings.append((pos, s))
        pos += len(line) + 1
    return headings

def chunk_document_pages(pages: List[str], doc_meta: Dict) -> List[Document]:
    page_marker_texts = [f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)]
    full_text = "\n\n".join(page_marker_texts)
    headings = detect_headings_in_page(full_text)
    chunks = []
    if headings:
        for start, heading in headings:
            end = full_text.find('\n\n', start + len(heading) + 10)
            chunk = full_text[start:end] if end > 0 else full_text[start:]
            if chunk.strip():
                chunks.append(chunk)
    else:
        chunks = [s.strip() for s in re.split(r'\n\s*\n', full_text) if s.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    final_chunks = []
    for i, c in enumerate(chunks):
        if len(c) < 1200:
            final_chunks.append(Document(page_content=c, metadata={"doc_id": doc_meta["doc_id"], "page": None, "section_title": "", "paragraph": i+1, "source": doc_meta["file_name"]}))
            m = re.search(r'\[Page (\d+)\]', c)
            if m:
                final_chunks[-1].metadata["page"] = int(m.group(1))
        else:
            sub = splitter.split_text(c)
            for j, s in enumerate(sub):
                doc = Document(page_content=s, metadata={"doc_id": doc_meta["doc_id"], "page": None, "section_title": "", "paragraph": i+1, "source": doc_meta["file_name"]})
                m = re.search(r'\[Page (\d+)\]', s)
                if m:
                    doc.metadata["page"] = int(m.group(1))
                final_chunks.append(doc)
    return final_chunks

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

# ==============================================================================
# SIMPLIFIED UI PAGES
# ==============================================================================

def page_dashboard():
    st.markdown('<h2 class="section-title">📊 Dashboard</h2>', unsafe_allow_html=True)
    docs = load_documents()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Documents", len(docs))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        total_pages = sum(d.get('pages', 0) for d in docs)
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Pages", total_pages)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        total_tables = len(load_json_file(TABLES_FILE))
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Tables", total_tables)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        total_images = len(load_json_file(IMAGES_FILE))
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Images", total_images)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Recent Uploads")
    if docs:
        recent = sorted(docs, key=lambda x: x.get('uploaded_at', 0), reverse=True)[:5]
        for doc in recent:
            col1, col2 = st.columns([3, 1])
            with col1:
                status = doc.get('status', 'processed')
                badge_class = "status-success" if status == "processed" else "status-warning" if status == "processing" else "status-error"
                st.markdown(f'<span class="status-badge {badge_class}">{status.upper()}</span> {doc["file_name"]}', unsafe_allow_html=True)
            with col2:
                ts = datetime.fromtimestamp(doc.get('uploaded_at', 0)).strftime('%m/%d')
                st.caption(ts)
    else:
        st.info("No documents uploaded yet.")

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
                    pages, toc = extract_text_pdf(raw, use_ocr_if_empty=use_ocr)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)
                    extraction_time = time.time() - start_time
                    # Chunk and embed
                    chunk_start = time.time()
                    chunks_docs = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": fname})
                    build_or_load_faiss(chunks_docs)
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
                        "pages": len(pages),
                        "toc": toc,
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

def page_documents():
    st.markdown('<h2 class="section-title">📚 Documents</h2>', unsafe_allow_html=True)
    docs = load_documents()
    if not docs:
        st.info("No documents ingested yet.")
        return
    for doc in sorted(docs, key=lambda x: x.get('uploaded_at', 0), reverse=True):
        with st.expander(f"📄 {doc['file_name']} ({doc['doc_id']}) - Status: {doc.get('status', 'unknown')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", doc.get('pages', 'N/A'))
            with col2:
                st.metric("📊 Tables", doc.get('tables', 0))
            with col3:
                st.metric("🖼️ Images", doc.get('images', 0))
            st.caption(f"⏰ Uploaded: {datetime.fromtimestamp(doc.get('uploaded_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️ View", key=f"view_{doc['doc_id']}"):
                    st.session_state.selected_doc = doc['doc_id']
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}"):
                    delete_document(doc['doc_id'])
                    st.rerun()
            with col3:
                if doc.get('pdf_path'):
                    pdf_bytes = load_pdf_local(doc['pdf_path'])
                    if pdf_bytes:
                        st.download_button(
                            "📥 Download PDF",
                            pdf_bytes,
                            file_name=doc['file_name'],
                            key=f"dl_{doc['doc_id']}"
                        )

@st.cache_data
def load_document_view(doc_id: str):
    docs = load_documents()
    doc = next((d for d in docs if d['doc_id'] == doc_id), None)
    if not doc or not doc.get('pdf_path'):
        return None, None
    raw = load_pdf_local(doc['pdf_path'])
    if raw:
        pages, _ = extract_text_pdf(raw)
        return pages, doc
    return None, None

def page_document_viewer():
    st.markdown('<h2 class="section-title">📖 Document Viewer</h2>', unsafe_allow_html=True)
    if st.session_state.selected_doc:
        pages, doc = load_document_view(st.session_state.selected_doc)
        if pages:
            selected_page = st.slider("Select Page", 1, len(pages), 1)
            st.text_area("Page Content", pages[selected_page-1], height=400)
            query = st.text_input("Highlight Text")
            if query:
                highlighted = pages[selected_page-1].replace(query, f"<mark>{query}</mark>")
                st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.error("Document not found.")

def page_tables():
    st.markdown('<h2 class="section-title">📊 Tables</h2>', unsafe_allow_html=True)
    tables = load_json_file(TABLES_FILE)
    if not tables:
        st.info("No tables extracted yet.")
        return
    docs = load_documents()
    doc_choices = {d['file_name']: d['doc_id'] for d in docs}
    selected_doc_name = st.selectbox("Select Document", ["All"] + list(doc_choices.keys()))
    for t in tables:
        doc_id = t['doc_id']
        if selected_doc_name != "All" and doc_choices.get(selected_doc_name) != doc_id:
            continue
        with st.expander(f"📊 Table (Page {t['page']}) - Doc: {doc_id}", expanded=False):
            df = pd.DataFrame(t["rows"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, file_name=f"table_page_{t['page']}.csv")

def page_images():
    st.markdown('<h2 class="section-title">🖼️ Images Gallery</h2>', unsafe_allow_html=True)
    images = load_json_file(IMAGES_FILE)
    if not images:
        st.info("No images extracted yet.")
        return
    cols = st.columns(3)
    for i, img in enumerate(images):
        with cols[i % 3]:
            try:
                if img.get('img_path'):
                    img_bytes = load_image_local(img['img_path'])
                    if img_bytes:
                        img_pil = Image.open(BytesIO(img_bytes))
                        st.image(img_pil, use_column_width=True)
                        st.caption(img.get('caption', 'Untitled'))
                        st.caption(f"Page: {img['page']}")
                        st.download_button("⬇️", img_bytes, file_name=f"img_{img['id']}.png", key=f"dl_img_{img['id']}")
            except Exception as e:
                st.error(f"Failed to load image: {e}")

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
                    docs = search_faiss(query, k=k)
                    st.session_state.search_results = docs
                    st.success(f"Found {len(docs)} results")
                    for i, d in enumerate(docs):
                        md = d.metadata
                        snippet = postprocess_extracted_text(d.page_content)[:500]
                        with st.expander(f"📄 Result {i+1}: {md.get('source')} (Page {md.get('page')})"):
                            st.write(snippet)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption(f"Doc ID: {md.get('doc_id')}")
                            with col2:
                                st.caption(f"Section: {md.get('section_title', 'N/A')}")
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
                    docs = search_faiss(query, k=k)
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
    st.sidebar.title("📚 Enterprise PDF → Knowledge")
    st.sidebar.markdown("Simplified: Local files + FAISS")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Upload & Ingest", "Documents", "Document Viewer", "Tables", "Images", "Search & Q&A"]
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    groq_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="sk-...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.sidebar.caption("✅ Set")
    st.markdown('<h1 class="main-header">📚 Enterprise PDF → Knowledge Hub</h1>', unsafe_allow_html=True)
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload & Ingest":
        page_upload()
    elif page == "Documents":
        page_documents()
    elif page == "Document Viewer":
        page_document_viewer()
    elif page == "Tables":
        page_tables()
    elif page == "Images":
        page_images()
    elif page == "Search & Q&A":
        page_search()
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #888;">Simplified with ❤️ | Local Storage + FAISS + OCR + Tables + RAG</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
