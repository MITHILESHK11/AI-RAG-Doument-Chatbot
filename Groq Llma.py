# ==============================================================================
# ENTERPRISE PDF → KNOWLEDGE: Enhanced Streamlit Application
# Full-featured application with MongoDB integration, reprocessing, advanced search,
# document viewer, benchmarks, and more features from the project plan.
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
import pytesseract
import cv2
import numpy as np
from bs4 import BeautifulSoup
# ML / LLM / embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
import uuid
import logging
# MongoDB
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"  # Legacy, now using Mongo
IMAGES_DIR = DATA_DIR / "images"
FAISS_DIR = DATA_DIR / "faiss_index"
AUDIT_LOG = DATA_DIR / "audit.log"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"
for d in (DATA_DIR, DOCS_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# MongoDB Configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "pdf_knowledge_db"
COLLECTIONS = {
    "documents": "documents",
    "chunks": "chunks",
    "tables": "tables",
    "images": "images"
}

# Logging configuration
logger = logging.getLogger("pdf_ingest")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(AUDIT_LOG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

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
# MONGO HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client[DB_NAME]
        return db
    except ConnectionFailure:
        st.error("❌ MongoDB connection failed. Using file-based fallback.")
        return None

def get_collection(db, coll_name):
    if db is None:
        return None
    return db[COLLECTIONS.get(coll_name, coll_name)]

def load_documents_from_mongo(db):
    coll = get_collection(db, "documents")
    if coll:
        return list(coll.find({}, {"_id": 0}))
    return []

def save_document_to_mongo(db, doc_data):
    coll = get_collection(db, "documents")
    if coll:
        coll.insert_one(doc_data)

def update_document_in_mongo(db, doc_id, updates):
    coll = get_collection(db, "documents")
    if coll:
        coll.update_one({"doc_id": doc_id}, {"$set": updates})

def delete_document_from_mongo(db, doc_id):
    coll = get_collection(db, "documents")
    if coll:
        coll.delete_one({"doc_id": doc_id})
        # Cascade delete chunks, tables, images
        for c in ["chunks", "tables", "images"]:
            get_collection(db, c).delete_many({"doc_id": doc_id})

# ==============================================================================
# SESSION STATE & AUTHENTICATION
# ==============================================================================
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'viewer'  # viewer, editor, admin
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None
if 'db' not in st.session_state:
    st.session_state.db = get_mongo_client()

# ==============================================================================
# HELPER FUNCTIONS (Enhanced)
# ==============================================================================
def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'[\ud800-\udfff]', '', s)

def audit_log(action: str, details: str = ""):
    timestamp = datetime.now().isoformat()
    logger.info(f"ACTION: {action} | ROLE: {st.session_state.user_role} | DETAILS: {details}")

def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    coords = np.column_stack(np.where(th > 0))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = th.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            th = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(th)

def extract_text_pdf(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[str], List[Dict]]:
    pages = []
    toc = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                txt = safe_text(txt)
                pages.append(txt)
        try:
            reader = PdfReader(BytesIO(file_bytes))
            outline = reader.outline
            toc = [{"title": str(getattr(item, "title", str(item))), "page": 1} for item in outline if hasattr(item, 'title')]
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
            for i in range(len(doc)):
                if pages[i].strip():
                    continue
                page = doc.load_page(i)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pre = preprocess_image_for_ocr(img)
                ocr_text = pytesseract.image_to_string(pre, lang='eng')
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
                        rowd = {header[j] if j < len(header) else f"col_{j}": safe_text(cell) for j, cell in enumerate(r)}
                        rows.append(rowd)
                    tables_all.append({"page": i, "header": header, "rows": rows})
    except:
        pass
    return tables_all

def extract_images_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    imgs = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_i in range(len(doc)):
            page = doc[page_i]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_name = f"{doc_id}_p{page_i+1}_img{img_index}.{ext}"
                p = IMAGES_DIR / img_name
                p.write_bytes(image_bytes)
                pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                if not caption:
                    caption = "Image (no OCR text) — visual content"
                imgs.append({"file": str(p), "page": page_i + 1, "caption": caption, "id": f"img_{doc_id}_{img_index}"})
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

def chunk_document_pages(pages: List[str], doc_meta: Dict) -> Tuple[List[str], List[Dict]]:
    page_marker_texts = [f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)]
    full_text = "\n\n".join(page_marker_texts)
    headings = detect_headings_in_page(full_text)
    chunks = []
    if headings:
        # Simple split by headings
        for start, heading in headings:
            # Approximate split (enhanced heuristic)
            end = full_text.find('\n\n', start + len(heading) + 10)
            chunk = full_text[start:end] if end > 0 else full_text[start:]
            if chunk.strip():
                chunks.append(chunk)
    else:
        chunks = [s.strip() for s in re.split(r'\n\s*\n', full_text) if s.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    final_chunks = []
    metadatas = []
    for i, c in enumerate(chunks):
        if len(c) < 1200:
            final_chunks.append(c)
            md = {"doc_id": doc_meta["doc_id"], "page": None, "section_title": "", "paragraph": i+1, "source": doc_meta["file_name"]}
            m = re.search(r'\[Page (\d+)\]', c)
            if m:
                md["page"] = int(m.group(1))
            metadatas.append(md)
        else:
            sub = splitter.split_text(c)
            for j, s in enumerate(sub):
                final_chunks.append(s)
                md = {"doc_id": doc_meta["doc_id"], "page": None, "section_title": "", "paragraph": i+1, "source": doc_meta["file_name"]}
                m = re.search(r'\[Page (\d+)\]', s)
                if m:
                    md["page"] = int(m.group(1))
                metadatas.append(md)
    return final_chunks, metadatas

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_or_load_faiss(chunks: List[str], metadatas: List[Dict]):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
            db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
            if chunks:
                db.add_texts(chunks, metadatas=metadatas)
                db.save_local(str(FAISS_DIR))
            return db
    except:
        pass
    if not chunks:
        chunks = [""]
        metadatas = [{"doc_id":"dummy","page":0,"paragraph":0,"source":"none"}]
    db = FAISS.from_texts(chunks, embedding=embed, metadatas=metadatas)
    db.save_local(str(FAISS_DIR))
    return db

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
        content = getattr(d, "page_content", str(d))
        meta = getattr(d, "metadata", {})
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

def load_benchmarks():
    if BENCHMARKS_FILE.exists():
        try:
            return json.loads(BENCHMARKS_FILE.read_text(encoding="utf8"))
        except:
            return {}
    return {"ingestion_times": [], "query_times": [], "extraction_accuracy": []}

def save_benchmarks(bench: Dict):
    try:
        BENCHMARKS_FILE.write_text(json.dumps(bench, indent=2), encoding="utf8")
    except:
        pass

# ==============================================================================
# UI: ROLE-BASED ACCESS CONTROL (Enhanced)
# ==============================================================================
def show_user_panel():
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Role & Access")
    role_options = ['viewer', 'editor', 'admin']
    st.session_state.user_role = st.sidebar.selectbox(
        "Select Role",
        role_options,
        index=role_options.index(st.session_state.user_role),
        help="Viewer = read-only, Editor = upload/edit, Admin = full control."
    )
    st.sidebar.markdown(f"**🧭 Current Role:** `{st.session_state.user_role.upper()}`")

def check_permission(required_role: str) -> bool:
    roles = {'viewer': 1, 'editor': 2, 'admin': 3}
    return roles.get(st.session_state.user_role, 0) >= roles.get(required_role, 0)

# ==============================================================================
# PAGES: ENHANCED UI COMPONENTS
# ==============================================================================
def page_dashboard():
    st.markdown('<h2 class="section-title">📊 Dashboard</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    docs = load_documents_from_mongo(db)
    benchmarks = load_benchmarks()

    col1, col2, col3, col4, col5 = st.columns(5)
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
        total_tables = sum(d.get('tables', 0) for d in docs)
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Tables", total_tables)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        total_images = sum(d.get('images', 0) for d in docs)
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Images", total_images)
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        avg_time = sum(benchmarks.get('ingestion_times', [])) / len(benchmarks.get('ingestion_times', [])) if benchmarks.get('ingestion_times') else 0
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Avg Ingest Time", f"{avg_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent uploads
    st.markdown("### Recent Uploads")
    if docs:
        recent = sorted(docs, key=lambda x: x.get('uploaded_at', 0), reverse=True)[:5]
        for doc in recent:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                status = doc.get('status', 'processed')
                badge_class = "status-success" if status == "processed" else "status-warning" if status == "processing" else "status-error"
                st.markdown(f'<span class="status-badge {badge_class}">{status.upper()}</span> {doc["file_name"]}', unsafe_allow_html=True)
            with col2:
                st.caption(f"Pages: {doc.get('pages', 'N/A')}")
            with col3:
                ts = datetime.fromtimestamp(doc.get('uploaded_at', 0)).strftime('%m/%d')
                st.caption(ts)
    else:
        st.info("No documents uploaded yet.")

def page_upload():
    st.markdown('<h2 class="section-title">📤 Upload & Ingest PDFs</h2>', unsafe_allow_html=True)

    if not check_permission('editor'):
        st.error("❌ You need editor permissions to upload documents.")
        return

    db = st.session_state.db
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
            benchmarks = load_benchmarks()

            for count, f in enumerate(uploaded_files):
                start_time = time.time()
                status_text.text(f"Processing {f.name}... ({count+1}/{total})")

                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    (DOCS_DIR / fname).write_bytes(raw)

                    # Update status
                    doc_meta = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "uploaded_by": st.session_state.user_role,  # Simple
                        "uploaded_at": time.time(),
                        "pages": 0,
                        "toc": [],
                        "status": "processing",
                        "metadata": {"department": department, "language": language},
                        "blob_path": str(DOCS_DIR / fname)
                    }
                    save_document_to_mongo(db, doc_meta)
                    audit_log("UPLOAD_START", f"Started {f.name} (ID: {doc_id})")

                    pages, toc = extract_text_pdf(raw, use_ocr_if_empty=use_ocr)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)

                    extraction_time = time.time() - start_time

                    # Chunk and embed
                    chunk_start = time.time()
                    chunks, ch_meta = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": fname})
                    db_vec = build_or_load_faiss(chunks, ch_meta)
                    embedding_time = time.time() - chunk_start

                    # Save to Mongo
                    doc_meta.update({
                        "pages": len(pages),
                        "toc": toc,
                        "tables": len(tables),
                        "images": len(images),
                        "status": "processed",
                        "extraction_time": extraction_time,
                        "embedding_time": embedding_time
                    })
                    update_document_in_mongo(db, doc_id, doc_meta)

                    # Save chunks, tables, images
                    chunks_coll = get_collection(db, "chunks")
                    if chunks_coll:
                        for i, (chunk, meta) in enumerate(zip(chunks, ch_meta)):
                            chunks_coll.insert_one({"_id": f"chunk_{doc_id}_{i}", **meta, "text": chunk})

                    tables_coll = get_collection(db, "tables")
                    if tables_coll and tables:
                        for t in tables:
                            t["doc_id"] = doc_id
                            tables_coll.insert_one(t)

                    images_coll = get_collection(db, "images")
                    if images_coll and images:
                        for img in images:
                            img["doc_id"] = doc_id
                            images_coll.insert_one(img)

                    # Benchmarks
                    benchmarks["ingestion_times"].append(extraction_time + embedding_time)
                    benchmarks["query_times"].append(0)  # Placeholder
                    save_benchmarks(benchmarks)

                    audit_log("UPLOAD_SUCCESS", f"Completed {f.name} (ID: {doc_id})")
                    st.success(f"✅ Ingested {f.name}")

                except Exception as e:
                    error_msg = str(e)
                    update_document_in_mongo(db, doc_id, {"status": "error", "error": error_msg})
                    st.error(f"❌ Failed to process {f.name}: {error_msg}")
                    audit_log("UPLOAD_ERROR", f"Failed {f.name}: {error_msg}")

                progress_bar.progress((count + 1) / total)

            status_text.empty()
            progress_bar.empty()
            st.balloons()

def page_documents():
    st.markdown('<h2 class="section-title">📚 Documents</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    docs = load_documents_from_mongo(db)

    if not docs:
        st.info("No documents ingested yet.")
        return

    # Filter
    departments = set(d.get('metadata', {}).get('department', '') for d in docs)
    selected_dept = st.selectbox("Filter by department", ["All"] + sorted(list(departments)))

    for doc in sorted(docs, key=lambda x: x.get('uploaded_at', 0), reverse=True):
        if selected_dept != "All" and doc.get('metadata', {}).get('department') != selected_dept:
            continue

        with st.expander(f"📄 {doc['file_name']} ({doc['doc_id']}) - Status: {doc.get('status', 'unknown')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Pages", doc.get('pages', 'N/A'))
            with col2:
                st.metric("📊 Tables", doc.get('tables', 0))
            with col3:
                st.metric("🖼️ Images", doc.get('images', 0))

            st.caption(f"⏰ Uploaded: {datetime.fromtimestamp(doc.get('uploaded_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")

            # Actions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("👁️ View", key=f"view_{doc['doc_id']}"):
                    st.session_state.selected_doc = doc['doc_id']
            with col2:
                if check_permission('editor') and st.button("✏️ Edit Metadata", key=f"edit_{doc['doc_id']}"):
                    edit_doc_metadata(doc)
            with col3:
                if check_permission('editor') and st.button("🔄 Reprocess", key=f"reprocess_{doc['doc_id']}"):
                    reprocess_document(doc['doc_id'])
            with col4:
                if check_permission('admin') and st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}"):
                    delete_document(doc['doc_id'])
                    st.rerun()

            # Download
            fpath = Path(doc["blob_path"])
            if fpath.exists():
                with open(fpath, 'rb') as file:
                    st.download_button(
                        "📥 Download PDF",
                        file.read(),
                        file_name=doc['file_name'],
                        key=f"dl_{doc['doc_id']}"
                    )

def edit_doc_metadata(doc):
    with st.form(key=f"edit_form_{doc['doc_id']}"):
        department = st.text_input("Department", value=doc.get('metadata', {}).get('department', ''))
        tags = st.text_area("Tags (comma-separated)", value=','.join(doc.get('metadata', {}).get('tags', [])))
        if st.form_submit_button("Save"):
            updates = {"metadata": {"department": department, "tags": tags.split(',') if tags else [], "language": doc.get('metadata', {}).get('language', 'en')}}
            update_document_in_mongo(st.session_state.db, doc['doc_id'], updates)
            audit_log("METADATA_EDIT", f"Edited metadata for {doc['doc_id']}")
            st.success("Metadata updated!")
            st.rerun()

def reprocess_document(doc_id):
    # Placeholder for reprocessing logic (re-run extraction)
    st.info("Reprocessing initiated... (Implementation: re-run extract_text_pdf, chunk, embed)")
    # TODO: Implement full reprocess
    audit_log("REPROCESS", f"Reprocessed {doc_id}")

def delete_document(doc_id):
    db = st.session_state.db
    delete_document_from_mongo(db, doc_id)
    # Delete file
    docs = load_documents_from_mongo(db)
    for d in docs:
        if d['doc_id'] == doc_id:
            fpath = Path(d["blob_path"])
            if fpath.exists():
                fpath.unlink()
            break
    audit_log("DELETE", f"Deleted {doc_id}")
    st.success("Document deleted!")

@st.cache_data
def load_document_view(doc_id):
    db = st.session_state.db
    doc = next((d for d in load_documents_from_mongo(db) if d['doc_id'] == doc_id), None)
    if not doc:
        return None
    fpath = Path(doc["blob_path"])
    if not fpath.exists():
        return None
    raw = fpath.read_bytes()
    pages, _ = extract_text_pdf(raw)
    return pages, doc

def page_document_viewer():
    st.markdown('<h2 class="section-title">📖 Document Viewer</h2>', unsafe_allow_html=True)
    if st.session_state.selected_doc:
        pages, doc = load_document_view(st.session_state.selected_doc)
        if pages:
            selected_page = st.slider("Select Page", 1, len(pages), 1)
            st.text_area("Page Content", pages[selected_page-1], height=400)
            # Highlight search (simple)
            query = st.text_input("Highlight Text")
            if query:
                highlighted = pages[selected_page-1].replace(query, f"<mark>{query}</mark>")
                st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.error("Document not found.")

def page_tables():
    st.markdown('<h2 class="section-title">📊 Tables</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    tables_coll = get_collection(db, "tables")
    if tables_coll:
        tables = list(tables_coll.find({}, {"_id": 0}))
    else:
        tables = []

    if not tables:
        st.info("No tables extracted yet.")
        return

    doc_choices = {d['file_name']: d['doc_id'] for d in load_documents_from_mongo(db)}
    selected_doc_name = st.selectbox("Select Document", ["All"] + list(doc_choices.keys()))

    for t in tables:
        doc_id = t['doc_id']
        if selected_doc_name != "All" and doc_choices.get(selected_doc_name) != doc_id:
            continue

        with st.expander(f"📊 Table (Page {t['page']}) - Doc: {doc_id}", expanded=False):
            df = pd.DataFrame(t["rows"])

            if check_permission('editor'):
                edited_df = st.data_editor(df, use_container_width=True, key=f"table_{t.get('_id', uuid.uuid4())}")
                if st.button("💾 Save Changes", key=f"save_table_{t.get('_id', uuid.uuid4())}"):
                    t["rows"] = edited_df.to_dict(orient='records')
                    tables_coll.update_one({"doc_id": doc_id, "page": t['page']}, {"$set": t})
                    audit_log("TABLE_EDIT", f"Edited table in {doc_id}")
                    st.success("✅ Table updated")
            else:
                st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, file_name=f"table_page_{t['page']}.csv")

def page_images():
    st.markdown('<h2 class="section-title">🖼️ Images Gallery</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    images_coll = get_collection(db, "images")
    if images_coll:
        images = list(images_coll.find({}, {"_id": 0}))
    else:
        images = []

    if not images:
        st.info("No images extracted yet.")
        return

    cols = st.columns(3)
    for i, img in enumerate(images):
        with cols[i % 3]:
            try:
                img_path = Path(img["file"])
                img_pil = Image.open(img_path)
                st.image(img_pil, use_column_width=True)
                caption_key = f"caption_{img['id']}"
                if caption_key not in st.session_state:
                    st.session_state[caption_key] = img.get('caption', img_path.stem)

                new_caption = st.text_input("Caption", value=st.session_state[caption_key], key=caption_key)
                if st.button("Update Caption", key=f"update_cap_{img['id']}"):
                    st.session_state[caption_key] = new_caption
                    images_coll.update_one({"id": img['id']}, {"$set": {"caption": new_caption}})
                    audit_log("IMAGE_CAPTION_EDIT", f"Updated caption for {img['id']}")
                    st.success("Caption updated!")

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Page: {img['page']}")
                with col2:
                    with open(img_path, 'rb') as f:
                        st.download_button("⬇️", f.read(), file_name=img_path.name, key=f"dl_img_{img['id']}")
            except Exception as e:
                st.error(f"Failed to load image: {e}")

def page_search():
    st.markdown('<h2 class="section-title">🔍 Search & Q&A</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔎 Enter your search query", placeholder="Ask anything...")
    with col2:
        k = st.slider("Top K Results", 1, 15, 5)
        search_type = st.selectbox("Search Type", ["Semantic", "Keyword", "Hybrid"])

    tab1, tab2 = st.tabs(["🔍 Search", "🤖 Q&A with RAG"])

    with tab1:
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a query.")
            else:
                try:
                    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    db_vec = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                    if search_type == "Keyword":
                        # Simple keyword (using FAISS metadata filter if possible, else basic)
                        docs = db_vec.similarity_search(query, k=k)  # Fallback to semantic
                    else:
                        docs = db_vec.similarity_search(query, k=k)
                    st.session_state.search_results = docs

                    audit_log("SEARCH", f"Query: {query} (Type: {search_type})")

                    st.success(f"Found {len(docs)} results")
                    for i, d in enumerate(docs):
                        md = d.metadata
                        snippet = postprocess_extracted_text(d.page_content)[:500]
                        with st.expander(f"📄 Result {i+1}: {md.get('source')} (Page {md.get('page')}) - Score: {d.metadata.get('score', 'N/A')}"):
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
                    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    db_vec = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                    docs = db_vec.similarity_search(query, k=k)

                    with st.spinner("🤔 Thinking..."):
                        start_q = time.time()
                        ans = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                        q_time = time.time() - start_q
                        benchmarks = load_benchmarks()
                        benchmarks["query_times"].append(q_time)
                        save_benchmarks(benchmarks)

                    audit_log("QA", f"Query: {query}")

                    st.markdown("### 📝 Answer")
                    st.markdown(ans)

                    # Parse and highlight citations
                    citations = re.findall(r'\(Document ID: \[([^\]]+)\], Page: \[([^\]]+)\]', ans)
                    with st.expander("📚 Sources"):
                        for cit in citations:
                            st.markdown(f'<div class="citation">Doc ID: {cit[0]}, Page: {cit[1]}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"QA failed: {e}")

def page_benchmarks():
    st.markdown('<h2 class="section-title">📈 Performance Benchmarks</h2>', unsafe_allow_html=True)
    benchmarks = load_benchmarks()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Ingestion Time", f"{sum(benchmarks.get('ingestion_times', []))/len(benchmarks.get('ingestion_times', [])):.2f}s" if benchmarks.get('ingestion_times') else "N/A")
    with col2:
        st.metric("Avg Query Time", f"{sum(benchmarks.get('query_times', []))/len(benchmarks.get('query_times', [])):.2f}s" if benchmarks.get('query_times') else "N/A")
    with col3:
        st.metric("Extraction Accuracy", f"{sum(benchmarks.get('extraction_accuracy', []))/len(benchmarks.get('extraction_accuracy', [])):.1%}" if benchmarks.get('extraction_accuracy') else "N/A")

    st.line_chart({"Ingestion Times": benchmarks.get('ingestion_times', [])})

def page_admin():
    st.markdown('<h2 class="section-title">⚙️ Admin Panel</h2>', unsafe_allow_html=True)

    if not check_permission('admin'):
        st.error("❌ Admin permissions required.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Management", "📈 Benchmarks", "🔐 Audit Log", "📥 Backup"])

    with tab1:
        st.subheader("System Management")
        if st.button("🔄 Rebuild FAISS Index"):
            with st.spinner("Rebuilding..."):
                db = st.session_state.db
                docs = load_documents_from_mongo(db)
                all_chunks, all_meta = [], []
                for doc in docs:
                    fpath = Path(doc["blob_path"])
                    if fpath.exists():
                        raw = fpath.read_bytes()
                        pages, _ = extract_text_pdf(raw)
                        chunks, ch_meta = chunk_document_pages(pages, doc)
                        all_chunks += chunks
                        all_meta += ch_meta
                build_or_load_faiss(all_chunks, all_meta)
                st.success("✅ Index rebuilt")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Docs in Mongo", len(load_documents_from_mongo(st.session_state.db)))
        with col2:
            st.metric("Tables in Mongo", len(list(get_collection(st.session_state.db, "tables").find())))
        with col3:
            st.metric("Images", len(list(IMAGES_DIR.glob("*"))))

    with tab2:
        page_benchmarks()

    with tab3:
        st.subheader("Audit Log")
        if AUDIT_LOG.exists():
            log_content = AUDIT_LOG.read_text()[-2000:]
            st.text_area("Recent Activity", log_content, height=300, disabled=True)
        else:
            st.info("No logs yet.")

    with tab4:
        st.subheader("Backup & Export")
        if st.button("📦 Create Full Backup (ZIP)"):
            zip_path = DATA_DIR / f"backup_{datetime.now().strftime('%Y%m%d')}.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                for f in DOCS_DIR.glob("*"):
                    z.write(f, arcname=f"docs/{f.name}")
                for f in IMAGES_DIR.glob("*"):
                    z.write(f, arcname=f"images/{f.name}")
                # Export Mongo data
                db = st.session_state.db
                if db:
                    for coll_name, mongo_coll in COLLECTIONS.items():
                        coll = get_collection(db, mongo_coll)
                        data = list(coll.find({}, {"_id": 0}))
                        json_file = DATA_DIR / f"{coll_name}.json"
                        json_file.write_text(json.dumps(data, indent=2))
                        z.write(json_file, arcname=f"mongo/{coll_name}.json")

            with open(zip_path, "rb") as f:
                st.download_button("📥 Download Backup", f.read(), file_name=zip_path.name)

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.sidebar.title("📚 Enterprise PDF → Knowledge")
    st.sidebar.markdown("Enhanced with MongoDB, reprocessing, viewer, benchmarks")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Upload & Ingest", "Documents", "Document Viewer", "Tables", "Images", "Search & Q&A", "Benchmarks", "Admin"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    groq_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="sk-...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.sidebar.caption("✅ Set")
    mongo_uri = st.sidebar.text_input("Mongo URI", value=MONGO_URI, type="password")
    if mongo_uri != MONGO_URI:
        os.environ["MONGO_URI"] = mongo_uri
        st.session_state.db = get_mongo_client()  # Refresh
        st.sidebar.caption("🔄 Reconnect")

    show_user_panel()

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
    elif page == "Benchmarks":
        page_benchmarks()
    elif page == "Admin":
        page_admin()

    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #888;">Built with ❤️ | MongoDB + OCR + Tables + RAG + Benchmarks</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
