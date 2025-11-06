# ==============================================================================
# ENTERPRISE PDF → KNOWLEDGE: Enhanced Streamlit Application
# Full-featured application using only MongoDB (for all data incl. blobs) and Milvus Vector DB.
# No local file storage; PDFs/images stored as binary in MongoDB.
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
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
import uuid
import logging
import numpy as np
# MongoDB
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.gridfs import GridFSBucket
# Milvus
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# MongoDB Configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "pdf_knowledge_db"
COLLECTIONS = {
    "documents": "documents",
    "chunks": "chunks",
    "tables": "tables",
    "images": "images"
}
BUCKETS = {
    "pdfs": "pdfs_bucket",
    "images": "images_bucket"
}

# Milvus Configuration
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "chunks_collection"
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
# MONGO HELPER FUNCTIONS (Enhanced with GridFS for Blobs)
# ==============================================================================
@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client[DB_NAME]
        return db
    except ConnectionFailure:
        st.error("❌ MongoDB connection failed.")
        return None

def get_collection(db, coll_name):
    if db is None:
        return None
    return db[COLLECTIONS.get(coll_name, coll_name)]

def get_bucket(db, bucket_name):
    if db is None:
        return None
    return GridFSBucket(db, bucket_name)

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
        # Delete blobs
        pdf_bucket = get_bucket(db, BUCKETS["pdfs"])
        img_bucket = get_bucket(db, BUCKETS["images"])
        if pdf_bucket:
            pdf_bucket.delete(doc_id)  # Assuming filename is doc_id
        if img_bucket:
            img_coll = get_collection(db, "images")
            img_ids = [img.get('blob_id') for img in img_coll.find({"doc_id": doc_id}, {"_id": 0, "blob_id": 1})]
            for img_id in img_ids:
                img_bucket.delete(img_id)

def save_blob(db, bucket, file_name, file_bytes, metadata=None):
    bucket_instance = get_bucket(db, bucket)
    if bucket_instance:
        file_id = bucket_instance.upload_from_stream(file_name, BytesIO(file_bytes), metadata=metadata)
        return str(file_id)
    return None

def load_blob(db, bucket, file_id):
    bucket_instance = get_bucket(db, bucket)
    if bucket_instance:
        file_obj = bucket_instance.open_download_stream(ObjectId(file_id))
        return file_obj.read()
    return None

# ==============================================================================
# MILVUS HELPER FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_milvus_connection():
    try:
        connections.connect(alias="default", uri=MILVUS_URI)
        if not utility.has_collection(COLLECTION_NAME):
            create_milvus_collection()
        return True
    except:
        st.error("❌ Milvus connection failed.")
        return False

def create_milvus_collection():
    if not get_milvus_connection():
        return None
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="page", dtype=DataType.INT32, default_value=0),
        FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="paragraph", dtype=DataType.INT32, default_value=0),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields=fields, description="PDF chunks embeddings")
    collection = Collection(COLLECTION_NAME, schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    return collection

def build_or_load_milvus(chunks: List[str], metadatas: List[Dict], collection=None):
    if collection is None:
        collection = create_milvus_collection()
        if collection is None:
            return None
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.embed_documents(chunks)
    ids = list(range(len(chunks)))
    data = [
        ids,
        [m["doc_id"] for m in metadatas],
        [m.get("page", 0) for m in metadatas],
        [m.get("section_title", "") for m in metadatas],
        [m.get("paragraph", 0) for m in metadatas],
        [m["source"] for m in metadatas],
        embeddings
    ]
    collection.insert(data)
    collection.flush()
    return collection

def search_milvus(query: str, k: int, collection=None):
    if collection is None:
        collection = create_milvus_collection()
        if collection is None:
            return []
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embed_model.embed_query(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["doc_id", "page", "section_title", "paragraph", "source"]
    )
    # Fetch full texts from Mongo for results
    db = st.session_state.db
    chunks_coll = get_collection(db, "chunks")
    docs = []
    for hit in results[0]:
        md = hit.entity
        chunk_doc = chunks_coll.find_one({"doc_id": md["doc_id"], "page": md["page"], "paragraph": md["paragraph"]})
        if chunk_doc:
            docs.append({"page_content": chunk_doc["text"], "metadata": md})
    return docs

# ==============================================================================
# SESSION STATE & AUTHENTICATION
# ==============================================================================
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'viewer'
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None
if 'db' not in st.session_state:
    st.session_state.db = get_mongo_client()
if 'milvus_coll' not in st.session_state:
    st.session_state.milvus_coll = create_milvus_collection()

# ==============================================================================
# HELPER FUNCTIONS
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
    if len(coords) > 0:
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
                pages.append(safe_text(txt))
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
                        rowd = {header[j] if j < len(header) else f"col_{j}": safe_text(cell) for j, cell in enumerate(r) if cell is not None}
                        rows.append(rowd)
                    tables_all.append({"page": i, "header": header, "rows": rows})
    except:
        pass
    return tables_all

def extract_images_pdf(file_bytes: bytes, doc_id: str, db) -> List[Dict]:
    imgs = []
    img_bucket = get_bucket(db, BUCKETS["images"])
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
                metadata = {"doc_id": doc_id, "page": page_i + 1}
                blob_id = save_blob(db, BUCKETS["images"], img_name, image_bytes, metadata)
                if blob_id:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                    if not caption:
                        caption = "Image (no OCR text) — visual content"
                    imgs.append({"blob_id": blob_id, "page": page_i + 1, "caption": caption, "id": f"img_{doc_id}_{img_index}"})
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
        for start, heading in headings:
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
        content = d.get("page_content", "")
        meta = d.get("metadata", {})
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

def load_benchmarks(db):
    coll = get_collection(db, "benchmarks")
    if coll and coll.count_documents({}) > 0:
        return coll.find_one({}, {"_id": 0}) or {}
    return {"ingestion_times": [], "query_times": [], "extraction_accuracy": []}

def save_benchmarks(db, bench: Dict):
    coll = get_collection(db, "benchmarks")
    if coll:
        coll.replace_one({}, bench, upsert=True)

# ==============================================================================
# UI: ROLE-BASED ACCESS CONTROL
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
# PAGES
# ==============================================================================
def page_dashboard():
    st.markdown('<h2 class="section-title">📊 Dashboard</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    docs = load_documents_from_mongo(db)
    benchmarks = load_benchmarks(db)

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
    pdf_bucket = get_bucket(db, BUCKETS["pdfs"])
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
            benchmarks = load_benchmarks(db)

            for count, f in enumerate(uploaded_files):
                start_time = time.time()
                status_text.text(f"Processing {f.name}... ({count+1}/{total})")

                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    metadata = {"doc_id": doc_id, "file_name": fname}
                    blob_id = save_blob(db, BUCKETS["pdfs"], fname, raw, metadata)

                    # Update status
                    doc_meta = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "blob_id": blob_id,
                        "uploaded_by": st.session_state.user_role,
                        "uploaded_at": time.time(),
                        "pages": 0,
                        "toc": [],
                        "status": "processing",
                        "metadata": {"department": department, "language": language}
                    }
                    save_document_to_mongo(db, doc_meta)
                    audit_log("UPLOAD_START", f"Started {f.name} (ID: {doc_id})")

                    pages, toc = extract_text_pdf(raw, use_ocr_if_empty=use_ocr)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id, db)

                    extraction_time = time.time() - start_time

                    # Chunk and embed with Milvus
                    chunk_start = time.time()
                    chunks, ch_meta = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": fname})
                    milvus_coll = st.session_state.milvus_coll
                    if milvus_coll:
                        build_or_load_milvus(chunks, ch_meta, milvus_coll)
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

                    # Save chunks, tables, images metadata
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
                    save_benchmarks(db, benchmarks)

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

            # Download PDF from Mongo
            pdf_bucket = get_bucket(db, BUCKETS["pdfs"])
            if pdf_bucket and doc.get('blob_id'):
                pdf_bytes = load_blob(db, BUCKETS["pdfs"], doc['blob_id'])
                if pdf_bytes:
                    st.download_button(
                        "📥 Download PDF",
                        pdf_bytes,
                        file_name=doc['file_name'],
                        key=f"dl_{doc['doc_id']}"
                    )

def edit_doc_metadata(doc):
    with st.form(key=f"edit_form_{doc['doc_id']}"):
        department = st.text_input("Department", value=doc.get('metadata', {}).get('department', ''))
        tags = st.text_area("Tags (comma-separated)", value=','.join(doc.get('metadata', {}).get('tags', [])))
        if st.form_submit_button("Save"):
            updates = {"metadata": {"department": department, "tags": [t.strip() for t in tags.split(',') if t.strip()], "language": doc.get('metadata', {}).get('language', 'en')}}
            update_document_in_mongo(st.session_state.db, doc['doc_id'], updates)
            audit_log("METADATA_EDIT", f"Edited metadata for {doc['doc_id']}")
            st.success("Metadata updated!")
            st.rerun()

def reprocess_document(doc_id):
    st.info("Reprocessing initiated... (Re-run extract, chunk, embed to Milvus)")
    audit_log("REPROCESS", f"Reprocessed {doc_id}")

def delete_document(doc_id):
    db = st.session_state.db
    delete_document_from_mongo(db, doc_id)
    audit_log("DELETE", f"Deleted {doc_id}")
    st.success("Document deleted!")

@st.cache_data
def load_document_view(doc_id):
    db = st.session_state.db
    doc = next((d for d in load_documents_from_mongo(db) if d['doc_id'] == doc_id), None)
    if not doc:
        return None
    pdf_bucket = get_bucket(db, BUCKETS["pdfs"])
    if pdf_bucket and doc.get('blob_id'):
        raw = load_blob(db, BUCKETS["pdfs"], doc['blob_id'])
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
    db = st.session_state.db
    tables_coll = get_collection(db, "tables")
    if tables_coll:
        tables = list(tables_coll.find({}, {"_id": 0}))
    else:
        tables = []

    if not tables:
        st.info("No tables extracted yet.")
        return

    docs = load_documents_from_mongo(db)
    doc_choices = {d['file_name']: d['doc_id'] for d in docs}
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
    img_bucket = get_bucket(db, BUCKETS["images"])
    for i, img in enumerate(images):
        with cols[i % 3]:
            try:
                if img_bucket and img.get('blob_id'):
                    img_bytes = load_blob(db, BUCKETS["images"], img['blob_id'])
                    if img_bytes:
                        img_pil = Image.open(BytesIO(img_bytes))
                        st.image(img_pil, use_column_width=True)
                        caption_key = f"caption_{img['id']}"
                        if caption_key not in st.session_state:
                            st.session_state[caption_key] = img.get('caption', 'Untitled')

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
                    milvus_coll = st.session_state.milvus_coll
                    if milvus_coll:
                        docs = search_milvus(query, k, milvus_coll)
                    else:
                        docs = []
                    st.session_state.search_results = docs

                    audit_log("SEARCH", f"Query: {query}")

                    st.success(f"Found {len(docs)} results")
                    for i, d in enumerate(docs):
                        md = d.get("metadata", {})
                        snippet = postprocess_extracted_text(d.get("page_content", ""))[:500]
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
                    milvus_coll = st.session_state.milvus_coll
                    if milvus_coll:
                        docs = search_milvus(query, k, milvus_coll)
                    else:
                        docs = []

                    with st.spinner("🤔 Thinking..."):
                        start_q = time.time()
                        ans = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                        q_time = time.time() - start_q
                        benchmarks = load_benchmarks(st.session_state.db)
                        benchmarks["query_times"].append(q_time)
                        save_benchmarks(st.session_state.db, benchmarks)

                    audit_log("QA", f"Query: {query}")

                    st.markdown("### 📝 Answer")
                    st.markdown(ans)

                    citations = re.findall(r'\(Document ID: \[([^\]]+)\], Page: \[([^\]]+)\]', ans)
                    with st.expander("📚 Sources"):
                        for cit in citations:
                            st.markdown(f'<div class="citation">Doc ID: {cit[0]}, Page: {cit[1]}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"QA failed: {e}")

def page_benchmarks():
    st.markdown('<h2 class="section-title">📈 Performance Benchmarks</h2>', unsafe_allow_html=True)
    db = st.session_state.db
    benchmarks = load_benchmarks(db)

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_ingest = sum(benchmarks.get('ingestion_times', [])) / len(benchmarks.get('ingestion_times', [])) if benchmarks.get('ingestion_times') else 0
        st.metric("Avg Ingestion Time", f"{avg_ingest:.2f}s")
    with col2:
        avg_query = sum(benchmarks.get('query_times', [])) / len(benchmarks.get('query_times', [])) if benchmarks.get('query_times') else 0
        st.metric("Avg Query Time", f"{avg_query:.2f}s")
    with col3:
        avg_acc = sum(benchmarks.get('extraction_accuracy', [])) / len(benchmarks.get('extraction_accuracy', [])) if benchmarks.get('extraction_accuracy') else 0
        st.metric("Extraction Accuracy", f"{avg_acc:.1%}")

    if benchmarks.get('ingestion_times'):
        st.line_chart({"Ingestion Times": benchmarks['ingestion_times']})

def page_admin():
    st.markdown('<h2 class="section-title">⚙️ Admin Panel</h2>', unsafe_allow_html=True)

    if not check_permission('admin'):
        st.error("❌ Admin permissions required.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Management", "📈 Benchmarks", "🔐 Audit Log", "📥 Backup"])

    with tab1:
        st.subheader("System Management")
        if st.button("🔄 Rebuild Milvus Index"):
            with st.spinner("Rebuilding..."):
                db = st.session_state.db
                docs = load_documents_from_mongo(db)
                all_chunks, all_meta = [], []
                for doc in docs:
                    if doc.get('blob_id'):
                        raw = load_blob(db, BUCKETS["pdfs"], doc['blob_id'])
                        if raw:
                            pages, _ = extract_text_pdf(raw)
                            chunks, ch_meta = chunk_document_pages(pages, doc)
                            all_chunks += chunks
                            all_meta += ch_meta
                milvus_coll = create_milvus_collection()
                if milvus_coll:
                    milvus_coll.drop()
                    build_or_load_milvus(all_chunks, all_meta, milvus_coll)
                    st.session_state.milvus_coll = milvus_coll
                st.success("✅ Milvus index rebuilt")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Docs in Mongo", len(load_documents_from_mongo(st.session_state.db)))
        with col2:
            st.metric("Tables in Mongo", get_collection(st.session_state.db, "tables").count_documents({}) if get_collection(st.session_state.db, "tables") else 0)
        with col3:
            st.metric("Vectors in Milvus", st.session_state.milvus_coll.num_entities if st.session_state.milvus_coll else 0)

    with tab2:
        page_benchmarks()

    with tab3:
        st.subheader("Audit Log")
        # Simulate log display (in production, fetch from Mongo audit collection)
        st.info("Audit logs would be fetched from MongoDB collection.")

    with tab4:
        st.subheader("Backup & Export")
        if st.button("📦 Export Mongo Data (JSON)"):
            db = st.session_state.db
            if db:
                export_data = {}
                for coll_name in COLLECTIONS.values():
                    coll = db[coll_name]
                    export_data[coll_name] = list(coll.find({}, {"_id": 0}))
                st.download_button("📥 Download JSON Export", json.dumps(export_data, indent=2).encode(), "mongo_export.json")

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.sidebar.title("📚 Enterprise PDF → Knowledge")
    st.sidebar.markdown("Using MongoDB + Milvus only (no local files)")

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
        st.session_state.db = get_mongo_client()
        st.sidebar.caption("🔄 Reconnected")
    milvus_uri = st.sidebar.text_input("Milvus URI", value=MILVUS_URI, type="password")
    if milvus_uri != MILVUS_URI:
        os.environ["MILVUS_URI"] = milvus_uri
        st.session_state.milvus_coll = create_milvus_collection()
        st.sidebar.caption("🔄 Reconnected")

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
    st.markdown('<div style="text-align: center; color: #888;">Built with ❤️ | MongoDB (Blobs + Metadata) + Milvus + OCR + Tables + RAG</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
