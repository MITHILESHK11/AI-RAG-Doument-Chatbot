# ==============================================================================
# ENTERPRISE PDF → KNOWLEDGE: Enhanced Streamlit Application
# Full-featured application with interactive UI, analytics, and advanced features
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

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.json"
FAISS_DIR = DATA_DIR / "faiss_index"
AUDIT_LOG = DATA_DIR / "audit.log"
METRICS_FILE = DATA_DIR / "metrics.json"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .status-success { background-color: #d4edda; color: #155724; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE & AUTHENTICATION
# ==============================================================================
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'viewer'  # viewer, editor, admin

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_metadata() -> Dict:
    """Load metadata from JSON file."""
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text(encoding="utf8"))
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    return {}

def save_metadata(meta: Dict):
    """Save metadata to JSON file."""
    try:
        METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf8")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

def load_metrics() -> Dict:
    """Load performance metrics."""
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text(encoding="utf8"))
        except Exception:
            return {"total_docs": 0, "total_tables": 0, "total_images": 0, "avg_extraction_time": 0}
    return {"total_docs": 0, "total_tables": 0, "total_images": 0, "avg_extraction_time": 0}

def save_metrics(metrics: Dict):
    """Save performance metrics."""
    try:
        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def make_doc_id() -> str:
    """Generate unique document ID."""
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    """Sanitize text by removing invalid Unicode characters."""
    if not s:
        return ""
    return re.sub(r'[\ud800-\udfff]', '', s)

def audit_log(action: str, details: str = ""):
    """Log user actions for audit trail."""
    timestamp = datetime.now().isoformat()
    logger.info(f"ACTION: {action} | ROLE: {st.session_state.user_role} | DETAILS: {details}")

def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Preprocess image for improved OCR accuracy."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
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
            th = cv2.warpAffine(th, M, (w, h),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    pil_res = Image.fromarray(th)
    return pil_res

def extract_text_pdf(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[str], List[Dict]]:
    """Extract text from PDF with OCR fallback for scanned pages."""
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
            try:
                outline = reader.outline
                toc = []
                for item in outline:
                    try:
                        title = getattr(item, "title", str(item))
                        toc.append(str(title))
                    except Exception:
                        continue
            except Exception:
                toc = []
        except Exception:
            toc = []
    except Exception:
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                pages.append(safe_text(txt))
        except Exception:
            pages = []

    # OCR fallback for empty pages
    if use_ocr_if_empty and any(not p.strip() for p in pages):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i in range(len(doc)):
                if pages and pages[i].strip():
                    continue
                page = doc.load_page(i)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pre = preprocess_image_for_ocr(img)
                try:
                    ocr_text = pytesseract.image_to_string(pre, lang='eng')
                except pytesseract.pytesseract.TesseractNotFoundError:
                    ocr_text = ""
                pages[i] = safe_text(ocr_text)
        except Exception:
            pass

    cleaned_pages = [postprocess_extracted_text(p) for p in pages]
    return cleaned_pages, toc

def dehyphenate(text: str) -> str:
    """Remove hyphenation artifacts from extracted text."""
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def postprocess_extracted_text(text: str) -> str:
    """Clean and format extracted text."""
    if not text:
        return ""
    text = text.replace('\r', '\n')
    text = dehyphenate(text)
    text = re.sub(r'[\x0c\x0b]', '', text)
    return text

def extract_tables_pdf(file_bytes: bytes) -> List[Dict]:
    """Extract tables from PDF using pdfplumber."""
    tables_all = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for t in tables:
                        if not t or len(t) < 2:
                            continue
                        header = [safe_text(c) for c in t[0]]
                        rows = []
                        for r in t[1:]:
                            rowd = {}
                            for j, cell in enumerate(r):
                                key = header[j] if j < len(header) else f"col_{j}"
                                rowd[str(key or j)] = safe_text(cell)
                            rows.append(rowd)
                        tables_all.append({"page": i, "header": header, "rows": rows})
                except Exception:
                    continue
    except Exception:
        pass
    return tables_all

def extract_images_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    """Extract images from PDF with OCR captions."""
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
                try:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                    if not caption:
                        caption = "Image (no OCR text) — visual content"
                except Exception:
                    caption = "Image (unable to OCR)"
                imgs.append({"file": str(p), "page": page_i + 1, "caption": caption, "id": f"img_{doc_id}_{img_index}"})
    except Exception:
        pass
    return imgs

def detect_headings_in_page(page_text: str) -> List[Tuple[int, str]]:
    """Detect headings in extracted text."""
    headings = []
    lines = page_text.splitlines()
    pos = 0
    for i, line in enumerate(lines):
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
    """Chunk document pages with metadata."""
    page_marker_texts = [f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)]
    full_text = "\n\n".join(page_marker_texts)
    headings = detect_headings_in_page(full_text)
    chunks = []
    if headings:
        sections = re.split(r'\n\s*\n', full_text)
        for sec in sections:
            sec = sec.strip()
            if sec:
                chunks.append(sec)
    else:
        chunks = [s.strip() for s in re.split(r'\n\s*\n', full_text) if s.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    final_chunks = []
    metadatas = []
    for i, c in enumerate(chunks):
        if len(c) < 1200:
            final_chunks.append(c)
            md = {"doc_id": doc_meta["doc_id"], "page": None, "paragraph": i+1, "source": doc_meta["file_name"]}
            m = re.search(r'\[Page (\d+)\]', c)
            if m:
                md["page"] = int(m.group(1))
            metadatas.append(md)
        else:
            sub = splitter.split_text(c)
            for j, s in enumerate(sub):
                final_chunks.append(s)
                md = {"doc_id": doc_meta["doc_id"], "page": None, "paragraph": i+1, "source": doc_meta["file_name"]}
                m = re.search(r'\[Page (\d+)\]', s)
                if m:
                    md["page"] = int(m.group(1))
                metadatas.append(md)
    return final_chunks, metadatas

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_or_load_faiss(chunks: List[str], metadatas: List[Dict]):
    """Build or load FAISS vector index."""
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
            db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
            if chunks:
                db.add_texts(chunks, metadatas=metadatas)
                try:
                    db.save_local(str(FAISS_DIR))
                except Exception:
                    pass
            return db
    except Exception:
        pass
    if not chunks:
        chunks = [""]
        metadatas = [{"doc_id":"dummy","page":0,"paragraph":0,"source":"none"}]
    db = FAISS.from_texts(chunks, embedding=embed, metadatas=metadatas)
    try:
        db.save_local(str(FAISS_DIR))
    except Exception:
        pass
    return db

QA_PROMPT = """You are an accurate document question-answering assistant.
Use ONLY the provided context to answer the question. Provide bullet points and cite every claim as:
(Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source]).
If the answer cannot be found in the context, respond EXACTLY: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

def ask_groq_with_docs(docs, question, model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=None):
    """Get answer from Groq LLM with document context."""
    parts = []
    for d in docs:
        content = getattr(d, "page_content", d.get("page_content") if isinstance(d, dict) else str(d))
        meta = getattr(d, "metadata", d.get("metadata") if isinstance(d, dict) else {})
        header = f"(Document ID: {meta.get('doc_id','NA')}, Page: {meta.get('page','NA')}, Paragraph: {meta.get('paragraph','NA')}, Source: {meta.get('source','NA')})"
        snippet = content if len(content) < 2000 else content[:1900] + "..."
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
# UI: ROLE-BASED ACCESS CONTROL
# ==============================================================================

def show_user_panel():
    """Display user info and role in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Role & Access")

    # Let developer switch freely (for now)
    role_options = ['viewer', 'editor', 'admin']
    st.session_state.user_role = st.sidebar.selectbox(
        "Select Role",
        role_options,
        index=role_options.index(st.session_state.user_role) if 'user_role' in st.session_state else 0,
        help="Switch between roles (for testing). Viewer = read-only, Editor = can upload/edit, Admin = full control."
    )

    st.sidebar.markdown(f"**🧭 Current Role:** `{st.session_state.user_role.upper()}`")

def check_permission(required_role: str) -> bool:
    """Check if user has required permission."""
    roles = {'viewer': 1, 'editor': 2, 'admin': 3}
    return roles.get(st.session_state.user_role, 0) >= roles.get(required_role, 0)

# ==============================================================================
# PAGES: ENHANCED UI COMPONENTS
# ==============================================================================

def page_dashboard():
    """Dashboard with analytics and metrics."""
    st.markdown('<h2 class="section-title">📊 Dashboard</h2>', unsafe_allow_html=True)
    
    metadata = load_metadata()
    metrics = load_metrics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Documents", len(metadata))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Tables", metrics.get('total_tables', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Images", metrics.get('total_images', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Avg Extract Time", f"{metrics.get('avg_extraction_time', 0):.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent uploads
    st.markdown("### Recent Uploads")
    if metadata:
        recent = sorted(metadata.items(), key=lambda x: x[1].get('uploaded_at', 0), reverse=True)[:5]
        for doc_id, info in recent:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📄 {info['file_name']}")
            with col2:
                st.caption(f"Pages: {info.get('pages', 'N/A')}")
            with col3:
                ts = datetime.fromtimestamp(info.get('uploaded_at', 0)).strftime('%m/%d')
                st.caption(ts)
    else:
        st.info("No documents uploaded yet.")

def page_upload():
    """Upload and ingest PDFs."""
    st.markdown('<h2 class="section-title">📤 Upload & Ingest PDFs</h2>', unsafe_allow_html=True)
    
    if not check_permission('editor'):
        st.error("❌ You need editor permissions to upload documents.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        department = st.text_input("📁 Department / Tag (optional)", placeholder="e.g., Finance, HR, Engineering")
    with col2:
        use_ocr = st.checkbox("Use OCR for scanned PDFs", value=True)
    
    uploaded_files = st.file_uploader("📎 Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🚀 Ingest Selected Files", type="primary"):
        if not uploaded_files:
            st.warning("Please select PDFs to upload.")
        else:
            metadata = load_metadata()
            metrics = load_metrics()
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
                    (DOCS_DIR / fname).write_bytes(raw)
                    
                    pages, toc = extract_text_pdf(raw, use_ocr_if_empty=use_ocr)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)
                    
                    extraction_time = time.time() - start_time
                    
                    doc_meta = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "uploaded_at": time.time(),
                        "toc": toc,
                        "department": department,
                        "pages_count": len(pages),
                        "extraction_time": extraction_time
                    }
                    
                    chunks, ch_meta = chunk_document_pages(pages, doc_meta)
                    db = build_or_load_faiss(chunks, ch_meta)
                    
                    if tables:
                        tbl_file = TABLES_DIR / f"{doc_id}_tables.json"
                        tbl_file.write_text(json.dumps(tables, indent=2), encoding="utf8")
                    
                    metadata[doc_id] = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "pages": len(pages),
                        "toc": toc,
                        "tables": len(tables),
                        "images": len(images),
                        "uploaded_at": time.time(),
                        "extraction_time": extraction_time
                    }
                    save_metadata(metadata)
                    
                    # Update metrics
                    metrics['total_docs'] = len(metadata)
                    metrics['total_tables'] = metrics.get('total_tables', 0) + len(tables)
                    metrics['total_images'] = metrics.get('total_images', 0) + len(images)
                    times = [m.get('extraction_time', 0) for m in metadata.values()]
                    metrics['avg_extraction_time'] = sum(times) / len(times) if times else 0
                    save_metrics(metrics)
                    
                    audit_log("UPLOAD", f"Uploaded {f.name} (ID: {doc_id})")
                    
                    st.success(f"✅ Ingested {f.name}")
                    
                except pytesseract.pytesseract.TesseractNotFoundError:
                    st.error(f"⚠️ Tesseract not installed for {f.name}")
                except Exception as e:
                    st.error(f"❌ Failed to process {f.name}: {str(e)}")
                    audit_log("UPLOAD_ERROR", f"Failed to upload {f.name}: {str(e)}")
                
                progress_bar.progress((count + 1) / total)
            
            status_text.empty()
            progress_bar.empty()
            st.balloons()

def page_documents():
    """Browse and manage documents."""
    st.markdown('<h2 class="section-title">📚 Documents</h2>', unsafe_allow_html=True)
    
    metadata = load_metadata()
    
    if not metadata:
        st.info("No documents ingested yet.")
        return
    
    # Filter by department
    departments = set()
    for info in metadata.values():
        if info.get('department'):
            departments.add(info['department'])
    
    if departments:
        selected_dept = st.selectbox("Filter by department", ["All"] + sorted(list(departments)))
    else:
        selected_dept = "All"
    
    # Display documents
    for doc_id, info in sorted(metadata.items(), key=lambda x: x[1].get('uploaded_at', 0), reverse=True):
        if selected_dept != "All" and info.get('department') != selected_dept:
            continue
        
        with st.expander(f"📄 {info['file_name']} ({doc_id})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Pages", info.get('pages', 'N/A'))
            with col2:
                st.metric("📊 Tables", info.get('tables', 0))
            with col3:
                st.metric("🖼️ Images", info.get('images', 0))
            
            st.caption(f"⏰ Uploaded: {datetime.fromtimestamp(info.get('uploaded_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"👁️ View", key=f"view_{doc_id}"):
                    st.session_state.selected_doc = doc_id
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{doc_id}") and check_permission('admin'):
                    fpath = DOCS_DIR / info["file_name"]
                    if fpath.exists():
                        fpath.unlink()
                    del metadata[doc_id]
                    save_metadata(metadata)
                    audit_log("DELETE", f"Deleted document {doc_id}")
                    st.success("Document deleted")
            
            with col3:
                if st.button(f"⬇️ Export", key=f"export_{doc_id}"):
                    fpath = DOCS_DIR / info["file_name"]
                    if fpath.exists():
                        with open(fpath, 'rb') as f:
                            st.download_button(
                                "📥 Download PDF",
                                f.read(),
                                file_name=info['file_name'],
                                key=f"dl_{doc_id}"
                            )

def page_tables():
    """View and edit extracted tables."""
    st.markdown('<h2 class="section-title">📊 Tables</h2>', unsafe_allow_html=True)
    
    metadata = load_metadata()
    tables_files = list(TABLES_DIR.glob("*.json"))
    
    if not tables_files:
        st.info("No tables extracted yet.")
        return
    
    # Select document
    doc_choices = {info['file_name']: doc_id for doc_id, info in metadata.items()}
    selected_doc_name = st.selectbox("Select Document", ["All"] + list(doc_choices.keys()))
    
    for tbl_file in tables_files:
        doc_id = tbl_file.name.split('_')[0]
        if selected_doc_name != "All" and doc_choices.get(selected_doc_name) != doc_id:
            continue
        
        try:
            tlist = json.loads(tbl_file.read_text(encoding="utf8"))
            for i, t in enumerate(tlist):
                with st.expander(f"📊 Table {i+1} (Page {t['page']})", expanded=False):
                    df = pd.DataFrame(t["rows"])
                    
                    # Editable table
                    if check_permission('editor'):
                        edited_df = st.data_editor(df, key=f"table_{doc_id}_{i}", use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Changes", key=f"save_{doc_id}_{i}"):
                                t["rows"] = edited_df.to_dict(orient='records')
                                tbl_file.write_text(json.dumps(tlist, indent=2), encoding="utf8")
                                audit_log("TABLE_EDIT", f"Edited table {i} in {doc_id}")
                                st.success("✅ Table updated")
                        
                        with col2:
                            csv = edited_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV",
                                csv,
                                file_name=f"table_{i}.csv",
                                key=f"csv_{doc_id}_{i}"
                            )
                    else:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            file_name=f"table_{i}.csv",
                            key=f"csv_view_{doc_id}_{i}"
                        )
        except Exception as e:
            st.error(f"Failed to load table: {e}")

def page_images():
    """View and annotate extracted images."""
    st.markdown('<h2 class="section-title">🖼️ Images Gallery</h2>', unsafe_allow_html=True)
    
    image_files = list(IMAGES_DIR.glob("*"))
    
    if not image_files:
        st.info("No images extracted yet.")
        return
    
    # Create image gallery
    cols = st.columns(3)
    
    for i, img_file in enumerate(sorted(image_files)):
        with cols[i % 3]:
            try:
                img = Image.open(img_file)
                st.image(img, use_column_width=True)
                
                # Editable caption
                caption_key = f"caption_{img_file.name}"
                if caption_key not in st.session_state:
                    st.session_state[caption_key] = img_file.stem
                
                caption = st.text_input(
                    "Caption",
                    value=st.session_state.get(caption_key, img_file.stem),
                    key=f"cap_{img_file.name}",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"📄 {img_file.name}")
                with col2:
                    with open(img_file, 'rb') as f:
                        st.download_button(
                            "⬇️",
                            f.read(),
                            file_name=img_file.name,
                            key=f"dl_img_{img_file.name}"
                        )
            except Exception as e:
                st.error(f"Failed to load image: {e}")

def page_search():
    """Semantic search and Q&A interface."""
    st.markdown('<h2 class="section-title">🔍 Search & Q&A</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔎 Enter your search query", placeholder="Ask anything about your documents...")
    with col2:
        k = st.slider("Results", 1, 15, 5)
    
    tab1, tab2 = st.tabs(["🔍 Semantic Search", "🤖 Q&A with RAG"])
    
    with tab1:
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                try:
                    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(query, k=k)
                    
                    audit_log("SEARCH", f"Query: {query}")
                    
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
                                st.caption(f"Paragraph: {md.get('paragraph')}")
                except Exception as e:
                    st.error(f"Search failed: {e}")
    
    with tab2:
        if st.button("Get Answer (RAG)", type="primary"):
            if not query.strip():
                st.warning("Enter a question.")
            elif not os.environ.get("GROQ_API_KEY"):
                st.error("❌ Set Groq API key in sidebar first.")
            else:
                try:
                    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(query, k=k)
                    
                    with st.spinner("🤔 Thinking..."):
                        ans = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                    
                    audit_log("QA", f"Query: {query}")
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(ans)
                    
                    with st.expander("📚 Source Documents"):
                        for d in docs:
                            md = d.metadata
                            st.write(f"**{md.get('source')}** - Page {md.get('page')}, Paragraph {md.get('paragraph')}")
                except Exception as e:
                    st.error(f"QA failed: {e}")

def page_admin():
    """Admin panel for system management."""
    st.markdown('<h2 class="section-title">⚙️ Admin Panel</h2>', unsafe_allow_html=True)
    
    if not check_permission('admin'):
        st.error("❌ You need admin permissions to access this page.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Management", "📈 Statistics", "🔐 Audit Log", "📥 Export"])
    
    with tab1:
        st.subheader("System Management")
        
        if st.button("🔄 Rebuild FAISS Index"):
            try:
                metadata = load_metadata()
                all_chunks, all_meta = [], []
                
                with st.spinner("Rebuilding..."):
                    for doc_id, info in metadata.items():
                        fpath = DOCS_DIR / info["file_name"]
                        if not fpath.exists():
                            continue
                        raw = fpath.read_bytes()
                        pages, _ = extract_text_pdf(raw, use_ocr_if_empty=True)
                        chunks, ch_meta = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": info["file_name"]})
                        all_chunks += chunks
                        all_meta += ch_meta
                
                build_or_load_faiss(all_chunks, all_meta)
                st.success("✅ FAISS index rebuilt successfully")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
        
        st.write("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Docs Storage", len(list(DOCS_DIR.glob("*"))))
        with col2:
            st.metric("Tables Files", len(list(TABLES_DIR.glob("*.json"))))
        with col3:
            st.metric("Images", len(list(IMAGES_DIR.glob("*"))))
    
    with tab2:
        st.subheader("Document Statistics")
        metadata = load_metadata()
        metrics = load_metrics()
        
        if metadata:
            pages = [info.get('pages', 0) for info in metadata.values()]
            times = [info.get('extraction_time', 0) for info in metadata.values()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pages", sum(pages))
            with col2:
                st.metric("Avg Pages/Doc", round(sum(pages) / len(pages), 1) if pages else 0)
            with col3:
                st.metric("Total Images", metrics.get('total_images', 0))
    
    with tab3:
        st.subheader("Audit Log")
        if AUDIT_LOG.exists():
            log_content = AUDIT_LOG.read_text()
            st.text_area("Recent Activity", log_content[-2000:], height=300, disabled=True)
        else:
            st.info("No audit log yet.")
    
    with tab4:
        st.subheader("Export & Backup")
        
        metadata = load_metadata()
        
        if st.button("📦 Create Backup (ZIP)"):
            import zipfile
            zip_path = DATA_DIR / "backup.zip"
            
            with zipfile.ZipFile(zip_path, "w") as z:
                for f in DOCS_DIR.glob("*"):
                    z.write(f, arcname=f"docs/{f.name}")
                for f in TABLES_DIR.glob("*"):
                    z.write(f, arcname=f"tables/{f.name}")
                for f in IMAGES_DIR.glob("*"):
                    z.write(f, arcname=f"images/{f.name}")
                z.write(METADATA_FILE, arcname="metadata.json")
            
            with open(zip_path, "rb") as f:
                st.download_button(
                    "📥 Download Backup",
                    f.read(),
                    file_name=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                )

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar header
    st.sidebar.title("📚 Enterprise PDF → Knowledge")
    st.sidebar.markdown("Transform PDFs into searchable, structured knowledge")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Upload & Ingest", "Documents", "Tables", "Images", "Search & Q&A", "Admin"]
    )
    
    # LLM Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")
    groq_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="sk-...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.sidebar.caption("✅ API Key set")
    else:
        st.sidebar.caption("⚠️ No API key provided")
    
    # User panel
    show_user_panel()
    
    # Main content
    st.markdown('<h1 class="main-header">📚 Enterprise PDF → Knowledge Hub</h1>', unsafe_allow_html=True)
    
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload & Ingest":
        page_upload()
    elif page == "Documents":
        page_documents()
    elif page == "Tables":
        page_tables()
    elif page == "Images":
        page_images()
    elif page == "Search & Q&A":
        page_search()
    elif page == "Admin":
        page_admin()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
    Built with ❤️ | OCR + Table Extraction + Vector Search + RAG Q&A
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

