# app.py - Enterprise PDF Knowledge System (Enhanced with Tables, Diagrams & Examples)
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

# ML / LLM / embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import pandas as pd
import uuid

# ========================================
# SETUP PATHS
# ========================================
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
FAISS_DIR = DATA_DIR / "faiss_index"
METADATA_FILE = DATA_DIR / "metadata.json"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ========================================
# HELPER FUNCTIONS
# ========================================
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
    return re.sub(r'[\ud800-\udfff]', '', s or "")


# ========================================
# OCR PREPROCESSING (IMPROVE SCANNED READABILITY)
# ========================================
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    return Image.fromarray(th)


# ========================================
# TEXT EXTRACTION (PDF + OCR FALLBACK)
# ========================================
def extract_text_pdf(file_bytes: bytes) -> List[str]:
    pages = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = safe_text(page.extract_text() or "")
                pages.append(text)
    except Exception:
        pass

    # OCR fallback if text missing
    if any(not p.strip() for p in pages):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                if pages and pages[i].strip():
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pre = preprocess_image_for_ocr(img)
                try:
                    ocr = pytesseract.image_to_string(pre, lang='eng')
                except pytesseract.TesseractNotFoundError:
                    ocr = ""
                pages[i] = safe_text(ocr)
        except Exception:
            pass
    return [postprocess_text(p) for p in pages]


def postprocess_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return re.sub(r'\n{2,}', '\n\n', text.strip())


# ========================================
# TABLE EXTRACTION
# ========================================
def extract_tables_pdf(file_bytes: bytes) -> List[Dict]:
    tables_all = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue
                    header = [safe_text(x) for x in table[0]]
                    rows = [
                        {header[j] if j < len(header) else f"col_{j}": safe_text(c) for j, c in enumerate(r)}
                        for r in table[1:]
                    ]
                    tables_all.append({"page": i, "header": header, "rows": rows})
    except Exception:
        pass
    return tables_all


# ========================================
# IMAGE & DIAGRAM EXTRACTION
# ========================================
def classify_image_description(caption: str) -> str:
    c = caption.lower()
    if any(x in c for x in ["decision tree", "flowchart", "diagram", "architecture", "model"]):
        return "Diagram"
    elif any(x in c for x in ["graph", "chart", "plot", "bar", "line", "histogram"]):
        return "Chart"
    elif any(x in c for x in ["dataset", "table", "matrix"]):
        return "Data Image"
    return "Other"


def extract_images_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    imgs = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            for j, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                path = IMAGES_DIR / f"{doc_id}_p{i+1}_{j}.{ext}"
                path.write_bytes(image_bytes)
                caption = ""
                try:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang="eng").strip()
                except Exception:
                    pass
                imgs.append({
                    "file": str(path),
                    "page": i + 1,
                    "caption": caption or "No visible text",
                    "type": classify_image_description(caption)
                })
        if imgs:
            with open(IMAGES_DIR / f"{doc_id}_diagrams.json", "w", encoding="utf8") as f:
                json.dump(imgs, f, indent=2)
    except Exception as e:
        st.warning(f"Image extraction failed: {e}")
    return imgs


# ========================================
# EXAMPLES / CASE STUDIES
# ========================================
def extract_examples_from_text(pages: List[str], doc_id: str) -> List[Dict]:
    examples = []
    pattern = re.compile(r'(Example|Case Study|Dataset)\s*\d*[:.\-]?\s*', re.IGNORECASE)
    for page_i, text in enumerate(pages, 1):
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if len(content) > 30:
                examples.append({
                    "page": page_i,
                    "title": m.group().strip(),
                    "content": content[:1000],
                    "doc_id": doc_id
                })
    if examples:
        with open(DATA_DIR / f"{doc_id}_examples.json", "w", encoding="utf8") as f:
            json.dump(examples, f, indent=2)
    return examples


# ========================================
# CHUNKING + FAISS INDEX
# ========================================
def chunk_document(pages: List[str], meta: Dict):
    text = "\n\n".join([f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)])
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_text(text)
    metas = []
    for i, c in enumerate(chunks):
        m = re.search(r"\[Page (\d+)\]", c)
        page = int(m.group(1)) if m else None
        metas.append({"doc_id": meta["doc_id"], "page": page, "paragraph": i + 1, "source": meta["file_name"]})
    return chunks, metas


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_or_load_faiss(chunks, metas):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        db.add_texts(chunks, metas)
        db.save_local(str(FAISS_DIR))
        return db
    db = FAISS.from_texts(chunks, embed, metas)
    db.save_local(str(FAISS_DIR))
    return db


# ========================================
# QA with GROQ LLM
# ========================================
QA_PROMPT = """You are a precise document Q&A assistant.
Use ONLY the given context. Provide clear bullet points and citations.
If answer not found, respond exactly: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

def ask_groq_with_docs(docs, query, key):
    ctx = "\n\n".join(
        f"(DocID:{d.metadata.get('doc_id')}, Page:{d.metadata.get('page')})\n{d.page_content[:1500]}"
        for d in docs
    )
    prompt = PromptTemplate.from_template(QA_PROMPT)
    formatted = prompt.format(context=ctx, question=query)
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, api_key=key)
    res = model.invoke(formatted)
    return res.content if hasattr(res, "content") else str(res)


# ========================================
# STREAMLIT UI
# ========================================
st.set_page_config(page_title="Enterprise PDF Knowledge", page_icon="📚", layout="wide")
st.title("📚 Enterprise PDF → Knowledge Explorer")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Ingest", "Documents", "Search & QA"])
groq_key = st.sidebar.text_input("Groq API Key", type="password")

uploaded = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
metadata = load_metadata()

# ------------------ Upload Page ------------------
if page == "Upload & Ingest":
    st.header("Upload and Ingest PDFs")
    if st.button("Process Files"):
        if not uploaded:
            st.warning("Upload at least one file")
        else:
            progress = st.progress(0)
            for i, f in enumerate(uploaded, 1):
                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    (DOCS_DIR / fname).write_bytes(raw)

                    pages = extract_text_pdf(raw)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)
                    examples = extract_examples_from_text(pages, doc_id)
                    chunks, metas = chunk_document(pages, {"doc_id": doc_id, "file_name": fname})
                    build_or_load_faiss(chunks, metas)

                    metadata[doc_id] = {
                        "file_name": fname,
                        "pages": len(pages),
                        "tables": bool(tables),
                        "images": bool(images),
                        "examples": bool(examples)
                    }
                    save_metadata(metadata)
                    st.success(f"✅ Ingested {f.name}")
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")
                progress.progress(i / len(uploaded))
            progress.empty()

# ------------------ Documents Page ------------------
elif page == "Documents":
    st.header("Uploaded Documents")
    if not metadata:
        st.info("No documents yet.")
    else:
        for doc_id, info in metadata.items():
            with st.expander(f"{info['file_name']} ({doc_id})"):
                tabs = st.tabs(["📄 Text Overview", "📊 Tables", "🖼️ Images & Diagrams", "💡 Examples"])

                with tabs[1]:
                    tpath = TABLES_DIR / f"{doc_id}_tables.json"
                    if tpath.exists():
                        data = json.loads(tpath.read_text())
                        for t in data:
                            st.dataframe(pd.DataFrame(t["rows"]))
                    else:
                        st.info("No tables found.")

                with tabs[2]:
                    ipath = IMAGES_DIR / f"{doc_id}_diagrams.json"
                    if ipath.exists():
                        imgs = json.loads(ipath.read_text())
                        for im in imgs:
                            st.image(im["file"], width=300, caption=f"{im['type']} (Page {im['page']})")
                            st.caption(im["caption"])
                    else:
                        st.info("No diagrams/images.")

                with tabs[3]:
                    epath = DATA_DIR / f"{doc_id}_examples.json"
                    if epath.exists():
                        ex = json.loads(epath.read_text())
                        for e in ex:
                            st.markdown(f"**{e['title']} (Page {e['page']})**")
                            st.write(e["content"])
                    else:
                        st.info("No examples detected.")

# ------------------ Search & QA ------------------
elif page == "Search & QA":
    st.header("Ask Questions or Search")
    query = st.text_input("Enter question or topic")
    k = st.slider("Top K", 1, 10, 5)
    if st.button("Search"):
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        for d in docs:
            st.markdown(f"**Doc:** {d.metadata['source']} (Page {d.metadata['page']})**")
            st.write(postprocess_text(d.page_content)[:700])
            st.divider()
    if st.button("Ask (RAG)") and groq_key:
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        st.subheader("Answer")
        st.markdown(ask_groq_with_docs(docs, query, groq_key))

