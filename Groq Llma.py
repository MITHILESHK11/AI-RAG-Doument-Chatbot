# app.py  (replace your existing Groq Llma.py with this)
import streamlit as st
import os
import re
import json
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

# ----- Config paths -----
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.json"
FAISS_DIR = DATA_DIR / "faiss_index"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----- Helpers -----
logger = logging.getLogger("pdf_ingest")
logger.setLevel(logging.INFO)

def load_metadata() -> Dict:
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text(encoding="utf8"))
        except Exception:
            return {}
    return {}

def save_metadata(meta: Dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf8")

def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'[\ud800-\udfff]', '', s)

# ----- Image preprocessing utilities (improve OCR quality) -----
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    # convert PIL image to OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    # optional deskew
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
    # convert back to PIL
    pil_res = Image.fromarray(th)
    return pil_res

# ----- PDF text extraction (digital first, then OCR fallback) -----
def extract_text_pdf(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[str], List[Dict]]:
    """
    Return pages (list of strings) and TOC (best-effort).
    Strategy:
    - Try pdfplumber text extraction per page (gives better layout)
    - If a page has no text, rasterize and run OCR with preprocessing
    """
    pages = []
    toc = []
    # First attempt: pdfplumber (better for layout & tables)
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                txt = safe_text(txt)
                pages.append(txt)
            # extract simple toc/outlines from PyPDF2 if available
        # Try to get outline using PyPDF2
        try:
            reader = PdfReader(BytesIO(file_bytes))
            try:
                outline = reader.outline
                # cast to simple list of strings if possible
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
        # fallback: use PyPDF2 page.extract_text
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                pages.append(safe_text(txt))
        except Exception:
            pages = []

    # If many pages are empty and OCR fallback enabled, run OCR on those pages
    if use_ocr_if_empty and any(not p.strip() for p in pages):
        try:
            # Use PyMuPDF to render pages to images for OCR (more robust)
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i in range(len(doc)):
                if pages and pages[i].strip():
                    continue  # already have digital text
                page = doc.load_page(i)
                zoom = 2  # render at 2x resolution
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # preprocess
                pre = preprocess_image_for_ocr(img)
                try:
                    ocr_text = pytesseract.image_to_string(pre, lang='eng')
                except pytesseract.pytesseract.TesseractNotFoundError:
                    # If tesseract missing, set a notice and continue
                    ocr_text = ""
                pages[i] = safe_text(ocr_text)
        except Exception:
            # If anything fails, ignore OCR
            pass

    # final pass: clean hyphenation & join lines into paragraphs
    cleaned_pages = [postprocess_extracted_text(p) for p in pages]
    return cleaned_pages, toc

# ----- Postprocessing functions to improve readability -----
def dehyphenate(text: str) -> str:
    # join words broken by hyphen at line breaks: e.g., "exam-\nple" -> "example"
    text = re.sub(r'-\s*\n\s*', '', text)
    # join lines that were broken artificially (heuristic)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # collapse multiple newlines to paragraphs
    text = re.sub(r'\n{2,}', '\n\n', text)
    # normalize spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def postprocess_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\r', '\n')
    text = dehyphenate(text)
    # remove odd control chars
    text = re.sub(r'[\x0c\x0b]', '', text)
    return text

# ----- Table extraction -----
def extract_tables_pdf(file_bytes: bytes) -> List[Dict]:
    """
    Extract tables using pdfplumber. Returns list of tables with page number, header, rows.
    """
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

# ----- Image extraction & captioning -----
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
                # caption via OCR + small heuristic
                try:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                    if not caption:
                        caption = "image (no OCR text) — visual content"
                except Exception:
                    caption = "image (unable to OCR)"
                imgs.append({"file": str(p), "page": page_i + 1, "caption": caption})
    except Exception:
        pass
    return imgs

# ----- Chunking that respects headings and paragraphs -----
def detect_headings_in_page(page_text: str) -> List[Tuple[int, str]]:
    """
    Heuristic: find lines that are either ALL CAPS or short lines ending with ':' or numbered headings.
    Returns list of (index_in_text_chunks, heading_text) - for use in splitting.
    """
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
    """
    Strategy:
    - Build a single long text with page markers.
    - Detect heading positions; split on headings first.
    - If resulting chunks still too long, apply RecursiveCharacterTextSplitter.
    """
    page_marker_texts = [f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)]
    full_text = "\n\n".join(page_marker_texts)
    # detect headings (naive)
    headings = detect_headings_in_page(full_text)
    chunks = []
    if headings:
        # split by headings positions (convert positions into splits)
        # fallback: split by double newlines into sections first
        sections = re.split(r'\n\s*\n', full_text)
        for sec in sections:
            sec = sec.strip()
            if sec:
                chunks.append(sec)
    else:
        # if no headings, split by paragraphs (double newline)
        chunks = [s.strip() for s in re.split(r'\n\s*\n', full_text) if s.strip()]

    # Now ensure chunk size by using RecursiveCharacterTextSplitter where needed
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

# ----- Create or update FAISS index (embedding model fixed) -----
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
def build_or_load_faiss(chunks: List[str], metadatas: List[Dict]):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # if index exists, load and add
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
    # else create new
    if not chunks:
        chunks = [""]  # avoid error
        metadatas = [{"doc_id":"dummy","page":0,"paragraph":0,"source":"none"}]
    db = FAISS.from_texts(chunks, embedding=embed, metadatas=metadatas)
    try:
        db.save_local(str(FAISS_DIR))
    except Exception:
        pass
    return db

# ----- RAG / LLM helpers -----
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
    # build context by concatenating top-k docs, but avoid huge length (truncate each doc to ~1000 tokens heuristic)
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

# ----- Streamlit UI (multi-page) -----
st.set_page_config(page_title="Enterprise PDFs → Knowledge", page_icon="📚", layout="wide")
st.title("📚 Enterprise PDF → Searchable Knowledge")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Ingest", "Documents", "Search & QA", "Admin & Export"])

st.sidebar.markdown("### LLM / OCR Configuration")
groq_key = st.sidebar.text_input("Groq API Key (for QA)", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

use_google_vision = st.sidebar.checkbox("Use Google Vision for OCR if Tesseract missing (requires API key)", value=False)
google_vision_key = None
if use_google_vision:
    google_vision_key = st.sidebar.text_input("Google Vision API key (optional)", type="password")
    if google_vision_key:
        os.environ["GOOGLE_VISION_KEY"] = google_vision_key

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
metadata = load_metadata()

# --- Pages ---
if page == "Upload & Ingest":
    st.header("Upload PDFs (digital + scanned)")
    st.markdown("Upload PDFs. The system extracts text, tables, images, and builds semantic index.")
    department = st.text_input("Department / Tag (optional)")
    if st.button("Ingest selected files"):
        if not uploaded_files:
            st.warning("Please select PDFs to upload.")
        else:
            progress = st.progress(0)
            total = len(uploaded_files)
            count = 0
            for f in uploaded_files:
                count += 1
                progress.progress(count/total)
                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    (DOCS_DIR / fname).write_bytes(raw)
                    pages, toc = extract_text_pdf(raw, use_ocr_if_empty=True)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)
                    doc_meta = {"doc_id": doc_id, "file_name": fname, "uploaded_at": time.time(), "toc": toc, "department": department}
                    chunks, ch_meta = chunk_document_pages(pages, doc_meta)
                    db = build_or_load_faiss(chunks, ch_meta)
                    # save tables
                    if tables:
                        tbl_file = TABLES_DIR / f"{doc_id}_tables.json"
                        tbl_file.write_text(json.dumps(tables, indent=2), encoding="utf8")
                    # save images metadata & small preview info
                    metadata[doc_id] = {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "pages": len(pages),
                        "toc": toc,
                        "tables": bool(tables),
                        "images": images,
                        "uploaded_at": time.time(),
                    }
                    save_metadata(metadata)
                    st.success(f"Ingested {f.name} → doc_id: {doc_id}")
                except pytesseract.pytesseract.TesseractNotFoundError:
                    st.error("Tesseract not installed. Please add tesseract via packages.txt or enable Google Vision.")
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
                    st.exception(traceback.format_exc())
            progress.empty()
            st.balloons()

elif page == "Documents":
    st.header("Documents")
    if not metadata:
        st.info("No documents ingested.")
    else:
        for doc_id, info in metadata.items():
            with st.expander(f"{info['file_name']} ({doc_id})"):
                st.write("Pages:", info.get("pages"))
                st.write("Uploaded:", time.ctime(info.get("uploaded_at")))
                st.write("Has tables:", info.get("tables"))
                if info.get("images"):
                    st.write("Images & captions:")
                    for im in info.get("images", []):
                        try:
                            st.image(im["file"], width=300)
                        except Exception:
                            st.write(im)
                        st.write("Caption:", im.get("caption"))
                # show first table preview
                tbl_path = TABLES_DIR / f"{doc_id}_tables.json"
                if tbl_path.exists():
                    st.write("Sample Tables:")
                    try:
                        tlist = json.loads(tbl_path.read_text(encoding="utf8"))
                        for t in tlist[:2]:
                            df = pd.DataFrame(t["rows"])
                            st.dataframe(df)
                    except Exception:
                        st.write("Unable to render table preview.")

elif page == "Search & QA":
    st.header("Search & QA")
    query = st.text_input("Enter a query")
    k = st.slider("Top k", 1, 10, 5)
    if st.button("Search semantic"):
        if not query.strip():
            st.warning("Enter query.")
        else:
            try:
                embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                docs = db.similarity_search(query, k=k)
                st.subheader(f"Top {len(docs)} results")
                for d in docs:
                    md = d.metadata
                    snippet = postprocess_extracted_text(d.page_content)[:1000]
                    st.markdown(f"**Source:** {md.get('source')} — DocID: {md.get('doc_id')} — Page: {md.get('page')}")
                    st.write(snippet)
                    st.markdown("---")
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(traceback.format_exc())

    if st.button("Ask (RAG)"):
        if not query.strip():
            st.warning("Enter query.")
        else:
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Set Groq API key in sidebar first.")
            else:
                try:
                    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(query, k=k)
                    st.info(f"Retrieved {len(docs)} chunks. Calling LLM...")
                    ans = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                    st.subheader("Answer (RAG)")
                    st.markdown(ans)
                except Exception as e:
                    st.error(f"QA failed: {e}")
                    st.exception(traceback.format_exc())

elif page == "Admin & Export":
    st.header("Admin & Export")
    if st.button("Rebuild FAISS from all stored docs"):
        try:
            all_chunks, all_meta = [], []
            for doc_id, info in metadata.items():
                fpath = DOCS_DIR / info["file_name"]
                if not fpath.exists():
                    continue
                raw = fpath.read_bytes()
                pages, _ = extract_text_pdf(raw, use_ocr_if_empty=True)
                chunks, ch_meta = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": info["file_name"]})
                all_chunks += chunks
                all_meta += ch_meta
            db = build_or_load_faiss(all_chunks, all_meta)
            st.success("Rebuilt FAISS.")
        except Exception as e:
            st.error(f"Failed to rebuild: {e}")
            st.exception(traceback.format_exc())

    if st.button("Export tables (zip)"):
        import zipfile
        zip_path = DATA_DIR / "tables_export.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for f in TABLES_DIR.glob("*.json"):
                z.write(f, arcname=f.name)
        with open(zip_path, "rb") as rf:
            st.download_button("Download tables zip", rf.read(), file_name="tables_export.zip")

    st.markdown("Local storage stats:")
    st.write("Docs:", len(list(DOCS_DIR.glob("*"))))
    st.write("Tables files:", len(list(TABLES_DIR.glob("*.json"))))
    st.write("Images:", len(list(IMAGES_DIR.glob("*"))))
    st.write("FAISS exists:", FAISS_DIR.exists() and any(FAISS_DIR.iterdir()))

# footer
st.markdown("---")
st.markdown("Built to convert enterprise PDFs into structured, searchable knowledge. OCR for scanned pages + table extraction included.")
