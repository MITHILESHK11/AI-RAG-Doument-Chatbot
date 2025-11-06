# ==============================================================================
# ENTERPRISE PDF → KNOWLEDGE: Advanced Edition with Enhanced Features
# - Document-specific search/RAG selection
# - Advanced image extraction with LLM labeling & context awareness
# - Deep detailed extraction for both scanned & digital PDFs
# - Improved OCR and context understanding
# ==============================================================================

import streamlit as st
import os
import re
import json
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import zipfile

# Extraction
import pdfplumber
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np
from bs4 import BeautifulSoup

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import pandas as pd
import uuid
import logging

# ==== CONFIG & PATHS ====
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.json"
FAISS_DIR = DATA_DIR / "faiss_index"
AUDIT_LOG = DATA_DIR / "audit.log"
METRICS_FILE = DATA_DIR / "metrics.json"
IMAGE_METADATA_FILE = DATA_DIR / "image_metadata.json"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pdf_ingest")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(AUDIT_LOG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

st.set_page_config(page_title="Enterprise PDF → Knowledge Pro", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# ==== ROLE SELECTION (CORRECTED) ====
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'viewer'
role_options = ["viewer", "editor", "admin"]
selected_role = st.sidebar.selectbox("Role:", role_options, index=role_options.index(st.session_state.user_role))
st.session_state.user_role = selected_role
st.sidebar.write(f"Status: {st.session_state.user_role.upper()}")

def check_permission(required_role: str) -> bool:
    roles = {'viewer': 1, 'editor': 2, 'admin': 3}
    return roles.get(st.session_state.user_role, 0) >= roles.get(required_role, 0)

# ==== HELPER FUNCTIONS ====
def load_metadata() -> Dict:
    if METADATA_FILE.exists():
        try: return json.loads(METADATA_FILE.read_text(encoding="utf8"))
        except: return {}
    return {}

def save_metadata(meta: Dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf8")

def load_image_metadata() -> Dict:
    if IMAGE_METADATA_FILE.exists():
        try: return json.loads(IMAGE_METADATA_FILE.read_text(encoding="utf8"))
        except: return {}
    return {}

def save_image_metadata(meta: Dict):
    IMAGE_METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf8")

def audit_log(action: str, details: str = ""):
    timestamp = datetime.now().isoformat()
    logger.info(f"ACTION: {action} | ROLE: {st.session_state.user_role} | DETAILS: {details}")

def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    return re.sub(r'[\ud800-\udfff]', '', s) if s else ""

# ==== ADVANCED IMAGE PREPROCESSING ====
def preprocess_image_for_ocr(pil_img: Image.Image, enhancement_level: int = 2) -> Image.Image:
    """Advanced preprocessing for scanned & low-quality images"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Multi-level noise reduction
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.medianBlur(gray, 5)
    
    # Advanced thresholding
    if enhancement_level >= 2:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Deskewing
    coords = np.column_stack(np.where(th > 0))
    if coords.size > 100:
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
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return Image.fromarray(th)

def is_scanned_pdf(file_bytes: bytes) -> bool:
    """Detect if PDF is scanned (image-based) vs digital text"""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            text_pages = 0
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if len(txt.strip()) > 50:
                    text_pages += 1
            scanned_ratio = (len(pdf.pages) - text_pages) / len(pdf.pages)
            return scanned_ratio > 0.3
    except:
        return True

# ==== ADVANCED TEXT EXTRACTION WITH CONTEXT ====
def extract_text_pdf_advanced(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Advanced extraction returning detailed page data with:
    - Page type (digital/scanned)
    - Text position
    - Surrounding text for images
    """
    pages_data = []
    toc = []
    is_scanned = is_scanned_pdf(file_bytes)
    
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                txt = safe_text(txt)
                
                # Extract detailed page info
                page_info = {
                    "page_num": page_num,
                    "text": txt,
                    "is_scanned": is_scanned if not txt.strip() else False,
                    "text_objects": page.extract_text_lines() if hasattr(page, 'extract_text_lines') else [],
                    "char_count": len(txt),
                    "word_count": len(txt.split()),
                }
                pages_data.append(page_info)
        
        # Extract TOC
        try:
            reader = PdfReader(BytesIO(file_bytes))
            toc = []
            for item in getattr(reader, "outline", []):
                try:
                    title = str(getattr(item, "title", item))
                    toc.append(title)
                except:
                    continue
        except:
            toc = []
    except Exception as e:
        logger.error(f"Primary PDF extraction failed: {e}")
    
    # OCR fallback for scanned pages
    if use_ocr_if_empty and any(not p["text"].strip() for p in pages_data):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page_data in enumerate(pages_data):
                if page_data["text"].strip():
                    continue
                
                page = doc.load_page(i)
                zoom = 3  # Higher resolution for better OCR
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pre = preprocess_image_for_ocr(img, enhancement_level=2)
                
                try:
                    ocr_text = pytesseract.image_to_string(pre, lang='eng')
                    page_data["text"] = safe_text(ocr_text)
                    page_data["is_scanned"] = True
                except pytesseract.pytesseract.TesseractNotFoundError:
                    page_data["text"] = ""
        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}")
    
    # Clean and postprocess all text
    cleaned_data = []
    for page_data in pages_data:
        page_data["text"] = postprocess_extracted_text(page_data["text"])
        cleaned_data.append(page_data)
    
    return cleaned_data, toc

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
                except:
                    continue
    except:
        pass
    return tables_all

# ==== ADVANCED IMAGE EXTRACTION WITH CONTEXT & LLM LABELING ====
def extract_images_pdf_with_context(file_bytes: bytes, doc_id: str, groq_api_key: str = None) -> List[Dict]:
    """
    Extract images with:
    - Surrounding text context
    - LLM-generated intelligent labels
    - Page content analysis
    """
    imgs = []
    page_texts = {}
    
    try:
        # First, get all page texts for context
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_texts[page_num] = page.extract_text() or ""
    except:
        pass
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_i in range(len(doc)):
            page = doc[page_i]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    img_name = f"{doc_id}_p{page_i+1}_img{img_index}.{ext}"
                    img_path = IMAGES_DIR / img_name
                    img_path.write_bytes(image_bytes)
                    
                    # Get surrounding context text
                    page_context = page_texts.get(page_i, "")
                    
                    # Generate OCR-based caption
                    try:
                        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                        ocr_caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                        if not ocr_caption:
                            ocr_caption = "(Image - no OCR text detected)"
                    except:
                        ocr_caption = "(OCR unavailable)"
                    
                    # Generate LLM label (intelligent description)
                    llm_label = ""
                    if groq_api_key and ocr_caption and ocr_caption != "(Image - no OCR text detected)":
                        try:
                            llm_label = generate_image_label(ocr_caption, page_context, groq_api_key)
                        except Exception as e:
                            logger.warning(f"LLM labeling failed: {e}")
                            llm_label = f"Image context: {page_context[:100]}..."
                    
                    imgs.append({
                        "file": str(img_path),
                        "page": page_i + 1,
                        "ocr_caption": ocr_caption,
                        "llm_label": llm_label,
                        "context": page_context[:200] if page_context else "(No surrounding text)",
                        "id": f"img_{doc_id}_{img_index}"
                    })
                except Exception as e:
                    logger.error(f"Error processing image {img_index}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
    
    return imgs

def generate_image_label(ocr_text: str, context: str, groq_api_key: str) -> str:
    """Use LLM to generate intelligent image labels"""
    prompt = f"""Based on the OCR text from an image and its surrounding context, 
    provide a concise 1-2 sentence label describing what this image likely depicts:
    
    OCR Text: {ocr_text[:300]}
    Context: {context[:300]}
    
    Label:"""
    
    try:
        model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, api_key=groq_api_key)
        res = model.invoke(prompt)
        return res.content[:200] if hasattr(res, 'content') else str(res)[:200]
    except:
        return f"Image with text: {ocr_text[:50]}..."

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

def chunk_document_pages_advanced(pages_data: List[Dict], doc_meta: Dict) -> Tuple[List[str], List[Dict]]:
    """Advanced chunking preserving page context and scanned vs digital info"""
    page_marker_texts = []
    for page_data in pages_data:
        page_num = page_data["page_num"]
        text = page_data["text"]
        is_scanned = page_data.get("is_scanned", False)
        marker = f"[Page {page_num}{'_SCANNED' if is_scanned else ''}]\n{text}"
        page_marker_texts.append(marker)
    
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
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = []
    metadatas = []
    
    for i, c in enumerate(chunks):
        if len(c) < 1500:
            final_chunks.append(c)
            md = {
                "doc_id": doc_meta["doc_id"],
                "page": None,
                "paragraph": i+1,
                "source": doc_meta["file_name"],
                "is_scanned": "SCANNED" in c
            }
            m = re.search(r'\[Page (\d+)', c)
            if m:
                md["page"] = int(m.group(1))
            metadatas.append(md)
        else:
            sub = splitter.split_text(c)
            for j, s in enumerate(sub):
                final_chunks.append(s)
                md = {
                    "doc_id": doc_meta["doc_id"],
                    "page": None,
                    "paragraph": i+1,
                    "source": doc_meta["file_name"],
                    "is_scanned": "SCANNED" in s
                }
                m = re.search(r'\[Page (\d+)', s)
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
                try:
                    db.save_local(str(FAISS_DIR))
                except:
                    pass
            return db
    except:
        pass
    
    if not chunks:
        chunks = [""]
        metadatas = [{"doc_id":"dummy","page":0,"paragraph":0,"source":"none","is_scanned":False}]
    
    db = FAISS.from_texts(chunks, embedding=embed, metadatas=metadatas)
    try:
        db.save_local(str(FAISS_DIR))
    except:
        pass
    return db

QA_PROMPT = """You are an accurate document question-answering assistant.
Use ONLY the provided context to answer the question. Provide clear, detailed answers with citations:
(Document: [SOURCE], Page: [PAGE], Context: [TYPE])

Context:
{context}

Question:
{question}

Answer:
"""

def ask_groq_with_docs(docs, question, doc_filter=None, model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=None):
    """Get answer with optional document filtering"""
    parts = []
    filtered_docs = []
    
    # Filter by document if specified
    if doc_filter and doc_filter != "All Documents":
        filtered_docs = [d for d in docs if d.metadata.get('source', '') == doc_filter]
        if not filtered_docs:
            filtered_docs = docs
    else:
        filtered_docs = docs
    
    for d in filtered_docs:
        content = getattr(d, "page_content", d.get("page_content") if isinstance(d, dict) else str(d))
        meta = getattr(d, "metadata", d.get("metadata") if isinstance(d, dict) else {})
        doc_type = "Scanned" if meta.get("is_scanned") else "Digital"
        header = f"(Document: {meta.get('source','NA')}, Page: {meta.get('page','NA')}, Type: {doc_type})"
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

# ==== PAGE FUNCTIONS ====

def page_upload():
    st.header("📤 Advanced PDF Upload & Ingest")
    if not check_permission('editor'):
        st.error("Need editor permissions to upload."); return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        department = st.text_input("Department/Tag (optional)")
    with col2:
        use_ocr = st.checkbox("Enable OCR", value=True)
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    use_llm_labels = st.checkbox("Use LLM for intelligent image labels (requires API key)", value=False)
    
    if st.button("🚀 Ingest Files", type="primary"):
        if not uploaded_files:
            st.warning("Select PDFs to upload."); return
        
        metadata = load_metadata()
        progress = st.progress(0)
        status_text = st.empty()
        total = len(uploaded_files)
        
        for count, f in enumerate(uploaded_files):
            status_text.text(f"Processing {f.name}... ({count+1}/{total})")
            try:
                raw = f.read()
                doc_id = make_doc_id()
                fname = f"{doc_id}_{f.name}"
                (DOCS_DIR / fname).write_bytes(raw)
                
                # Advanced extraction
                pages_data, toc = extract_text_pdf_advanced(raw, use_ocr_if_empty=use_ocr)
                tables = extract_tables_pdf(raw)
                
                # Extract images with LLM labels
                groq_key = os.environ.get("GROQ_API_KEY") if use_llm_labels else None
                images = extract_images_pdf_with_context(raw, doc_id, groq_api_key=groq_key)
                
                doc_meta = {
                    "doc_id": doc_id,
                    "file_name": fname,
                    "uploaded_at": time.time(),
                    "toc": toc,
                    "department": department,
                    "is_scanned": is_scanned_pdf(raw),
                    "pages_count": len(pages_data)
                }
                
                chunks, ch_meta = chunk_document_pages_advanced(pages_data, doc_meta)
                db = build_or_load_faiss(chunks, ch_meta)
                
                if tables:
                    (TABLES_DIR / f"{doc_id}_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf8")
                
                if images:
                    img_meta = load_image_metadata()
                    img_meta[doc_id] = images
                    save_image_metadata(img_meta)
                
                metadata[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": fname,
                    "pages": len(pages_data),
                    "toc": toc,
                    "tables": len(tables),
                    "images": len(images),
                    "uploaded_at": time.time(),
                    "is_scanned": is_scanned_pdf(raw)
                }
                save_metadata(metadata)
                audit_log("UPLOAD", f"Uploaded {f.name} (ID: {doc_id})")
                st.success(f"✅ Ingested {f.name}")
            except Exception as e:
                st.error(f"❌ Failed {f.name}: {e}")
                st.exception(traceback.format_exc())
            
            progress.progress((count+1)/total)
        
        st.balloons()

def page_images():
    st.header("🖼️ Advanced Image Gallery")
    img_meta = load_image_metadata()
    
    if not img_meta:
        st.info("No images extracted yet."); return
    
    for doc_id, images in img_meta.items():
        st.subheader(f"Images from Document: {doc_id}")
        
        cols = st.columns(3)
        for i, img_info in enumerate(images):
            with cols[i % 3]:
                try:
                    if Path(img_info["file"]).exists():
                        img = Image.open(img_info["file"])
                        st.image(img, use_column_width=True)
                        
                        with st.expander("📝 Details"):
                            st.write(f"**Page:** {img_info['page']}")
                            st.write(f"**OCR Text:** {img_info.get('ocr_caption', 'N/A')[:100]}")
                            st.write(f"**LLM Label:** {img_info.get('llm_label', 'N/A')[:150]}")
                            st.write(f"**Context:** {img_info.get('context', 'N/A')[:100]}")
                        
                        with open(img_info["file"], 'rb') as f:
                            st.download_button("⬇️ Download", f.read(), file_name=Path(img_info["file"]).name)
                except Exception as e:
                    st.error(f"Image error: {e}")

def page_search():
    st.header("🔍 Advanced Search & Q&A")
    
    metadata = load_metadata()
    doc_options = ["All Documents"] + [info.get("file_name", doc_id) for doc_id, info in metadata.items()]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("Search query")
    with col2:
        selected_doc = st.selectbox("Filter by document", doc_options)
    with col3:
        k = st.slider("Results", 1, 15, 5)
    
    tab1, tab2 = st.tabs(["Semantic Search", "Q&A (RAG)"])
    
    with tab1:
        if st.button("Search"):
            if not query.strip():
                st.warning("Enter query."); return
            try:
                embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                docs = db.similarity_search(query, k=k)
                
                # Filter if needed
                if selected_doc != "All Documents":
                    docs = [d for d in docs if d.metadata.get("source", "") == selected_doc]
                
                st.success(f"Found {len(docs)} results")
                for i, d in enumerate(docs):
                    md = d.metadata
                    snippet = d.page_content[:500]
                    doc_type = "Scanned" if md.get("is_scanned") else "Digital"
                    
                    with st.expander(f"📄 Result {i+1}: {md.get('source')} | Page {md.get('page')} ({doc_type})"):
                        st.write(snippet)
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    with tab2:
        if st.button("Get Answer (RAG)"):
            if not query.strip():
                st.warning("Enter question."); return
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Set Groq API key."); return
            
            try:
                embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                docs = db.similarity_search(query, k=k)
                
                with st.spinner("🤔 Thinking..."):
                    ans = ask_groq_with_docs(docs, query, doc_filter=selected_doc, groq_api_key=os.environ.get("GROQ_API_KEY"))
                
                st.markdown("### 📝 Answer")
                st.markdown(ans)
                
                with st.expander("📚 Source Documents"):
                    for d in docs:
                        md = d.metadata
                        doc_type = "Scanned" if md.get("is_scanned") else "Digital"
                        st.write(f"**{md.get('source')}** - Page {md.get('page')}, Type: {doc_type}")
            except Exception as e:
                st.error(f"Q&A failed: {e}")

def page_documents():
    st.header("📚 Documents")
    metadata = load_metadata()
    
    if not metadata:
        st.info("No documents yet."); return
    
    for doc_id, info in sorted(metadata.items(), key=lambda x: x[1].get('uploaded_at', 0), reverse=True):
        doc_type = "🔍 Scanned" if info.get("is_scanned") else "📄 Digital"
        with st.expander(f"{doc_type} {info.get('file_name')} ({doc_id})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pages", info.get("pages", "N/A"))
            with col2:
                st.metric("Tables", info.get("tables", 0))
            with col3:
                st.metric("Images", info.get("images", 0))
            
            st.caption(f"⏰ {datetime.fromtimestamp(info.get('uploaded_at', 0)).strftime('%Y-%m-%d %H:%M')}")

def page_admin():
    st.header("⚙️ Admin Panel")
    if not check_permission('admin'):
        st.error("Need admin permissions."); return
    
    if st.button("🔄 Rebuild FAISS"):
        try:
            metadata = load_metadata()
            all_chunks, all_meta = [], []
            for doc_id, info in metadata.items():
                fpath = DOCS_DIR / info["file_name"]
                if not fpath.exists():
                    continue
                raw = fpath.read_bytes()
                pages_data, _ = extract_text_pdf_advanced(raw)
                chunks, ch_meta = chunk_document_pages_advanced(pages_data, {"doc_id": doc_id, "file_name": info["file_name"]})
                all_chunks += chunks
                all_meta += ch_meta
            
            build_or_load_faiss(all_chunks, all_meta)
            st.success("✅ FAISS rebuilt")
        except Exception as e:
            st.error(f"Failed: {e}")

# ==== MAIN ====
st.sidebar.title("📚 PDF Knowledge Pro")

groq_key = st.sidebar.text_input("Groq API Key", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

page = st.sidebar.radio("Page", ["Upload", "Documents", "Images", "Search", "Admin"])

if page == "Upload":
    page_upload()
elif page == "Documents":
    page_documents()
elif page == "Images":
    page_images()
elif page == "Search":
    page_search()
elif page == "Admin":
    page_admin()

st.markdown("---")
st.markdown("Built with ❤️ | Advanced OCR + Deep Extraction + LLM Image Labeling + Filtered RAG")
