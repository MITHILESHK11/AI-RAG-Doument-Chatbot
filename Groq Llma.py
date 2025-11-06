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

# ============================ SETUP PATHS ============================
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.json"
FAISS_DIR = DATA_DIR / "faiss_index"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================ HELPERS ============================
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


# ============================ IMAGE PREPROCESSING ============================
def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
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
    return Image.fromarray(th)


# ============================ TEXT EXTRACTION ============================
def postprocess_extracted_text(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return re.sub(r'[ \t]{2,}', ' ', text).strip()


def extract_text_pdf(file_bytes: bytes, use_ocr_if_empty: bool = True) -> Tuple[List[str], List[str]]:
    pages = []
    toc = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(safe_text(txt))
    except Exception:
        pass

    # OCR fallback
    if use_ocr_if_empty and any(not p.strip() for p in pages):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i in range(len(doc)):
                if pages and pages[i].strip():
                    continue
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pre = preprocess_image_for_ocr(img)
                try:
                    text = pytesseract.image_to_string(pre, lang="eng")
                except pytesseract.TesseractNotFoundError:
                    text = ""
                pages[i] = safe_text(text)
        except Exception:
            pass
    cleaned_pages = [postprocess_extracted_text(p) for p in pages]
    return cleaned_pages, toc


# ============================ TABLE & IMAGE EXTRACTION ============================
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
                        rowd = {}
                        for j, cell in enumerate(r):
                            key = header[j] if j < len(header) else f"col_{j}"
                            rowd[str(key or j)] = safe_text(cell)
                        rows.append(rowd)
                    tables_all.append({"page": i, "header": header, "rows": rows})
    except Exception:
        pass
    return tables_all


def extract_images_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    imgs = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_i in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_i)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_name = f"{doc_id}_p{page_i+1}_img{img_index}.{ext}"
                p = IMAGES_DIR / img_name
                p.write_bytes(image_bytes)
                pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                caption = ""
                try:
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                except Exception:
                    caption = ""
                imgs.append({"file": str(p), "page": page_i + 1, "caption": caption or "No text"})
    except Exception:
        pass
    return imgs


# ============================ NEW: DIAGRAM + EXAMPLES EXTRACTION ============================
def classify_image_description(caption: str) -> str:
    c = caption.lower()
    if any(x in c for x in ["decision tree", "flowchart", "architecture", "model", "diagram", "block"]):
        return "Diagram"
    elif any(x in c for x in ["graph", "plot", "chart", "bar", "line", "histogram"]):
        return "Chart"
    elif any(x in c for x in ["dataset", "table", "matrix"]):
        return "Data Table Image"
    return "Other"


def extract_diagrams_pdf(file_bytes: bytes, doc_id: str) -> List[Dict]:
    diagrams = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_i in range(len(doc)):
            page = doc[page_i]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                img_name = f"{doc_id}_diagram_p{page_i+1}_{img_index}.{ext}"
                path = IMAGES_DIR / img_name
                path.write_bytes(image_bytes)
                caption = ""
                try:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(preprocess_image_for_ocr(pil), lang='eng').strip()
                except Exception:
                    caption = ""
                tag = classify_image_description(caption)
                diagrams.append({"file": str(path), "page": page_i + 1, "caption": caption, "type": tag})
        if diagrams:
            with open(IMAGES_DIR / f"{doc_id}_diagrams.json", "w", encoding="utf8") as f:
                json.dump(diagrams, f, indent=2)
    except Exception as e:
        st.warning(f"Diagram extraction failed: {e}")
    return diagrams


def extract_examples_from_text(pages: List[str], doc_id: str) -> List[Dict]:
    examples = []
    pattern = re.compile(r'(Example|Case Study|Dataset)\s*\d*[:.\-]?\s*', re.IGNORECASE)
    for pnum, text in enumerate(pages, 1):
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            snippet = text[start:end].strip()
            if len(snippet) > 30:
                examples.append({
                    "page": pnum,
                    "title": m.group().strip(),
                    "content": snippet[:1000],
                    "doc_id": doc_id
                })
    if examples:
        with open(DATA_DIR / f"{doc_id}_examples.json", "w", encoding="utf8") as f:
            json.dump(examples, f, indent=2)
    return examples


# ============================ CHUNKING ============================
def chunk_document_pages(pages: List[str], doc_meta: Dict) -> Tuple[List[str], List[Dict]]:
    full_text = "\n\n".join([f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)])
    paragraphs = [s.strip() for s in re.split(r'\n\s*\n', full_text) if s.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks, meta = [], []
    for i, para in enumerate(paragraphs):
        sub = splitter.split_text(para)
        for j, s in enumerate(sub):
            m = re.search(r'\[Page (\d+)\]', s)
            page = int(m.group(1)) if m else None
            chunks.append(s)
            meta.append({
                "doc_id": doc_meta["doc_id"],
                "page": page,
                "paragraph": i + 1,
                "source": doc_meta["file_name"]
            })
    return chunks, meta


# ============================ FAISS INDEX ============================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_or_load_faiss(chunks: List[str], meta: List[Dict]):
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        db.add_texts(chunks, metadatas=meta)
        db.save_local(str(FAISS_DIR))
        return db
    db = FAISS.from_texts(chunks, embed, metadatas=meta)
    db.save_local(str(FAISS_DIR))
    return db


# ============================ RAG QA ============================
QA_PROMPT = """You are an accurate document Q&A assistant.
Use ONLY the given context to answer the question. Write in clear bullet points.
Cite every claim like (Document ID:[ID], Page:[X], Paragraph:[Y], Source:[S]).
If not found, reply exactly: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

def ask_groq_with_docs(docs, question, key):
    parts = []
    for d in docs:
        c = d.page_content
        m = d.metadata
        header = f"(Document ID:{m.get('doc_id')}, Page:{m.get('page')}, Paragraph:{m.get('paragraph')}, Source:{m.get('source')})"
        snippet = c[:1500] + "..." if len(c) > 1500 else c
        parts.append(header + "\n" + snippet)
    context = "\n\n---\n\n".join(parts)
    prompt = PromptTemplate.from_template(QA_PROMPT)
    formatted = prompt.format(context=context, question=question)
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, api_key=key)
    res = model.invoke(formatted)
    return res.content if hasattr(res, "content") else str(res)


# ============================ STREAMLIT UI ============================
st.set_page_config(page_title="Enterprise PDF → Knowledge", page_icon="📚", layout="wide")
st.title("📚 Enterprise PDF → Searchable Knowledge System")

st.sidebar.header("Configuration")
page = st.sidebar.radio("Navigation", ["Upload & Ingest", "Documents", "Search & QA", "Admin & Export"])
groq_key = st.sidebar.text_input("Groq API Key", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
metadata = load_metadata()

# ------------------ Upload Page ------------------
if page == "Upload & Ingest":
    st.header("Upload & Ingest PDFs")
    if st.button("Start Ingestion"):
        if not uploaded_files:
            st.warning("Upload at least one PDF.")
        else:
            progress = st.progress(0)
            for idx, f in enumerate(uploaded_files, 1):
                try:
                    raw = f.read()
                    doc_id = make_doc_id()
                    fname = f"{doc_id}_{f.name}"
                    (DOCS_DIR / fname).write_bytes(raw)
                    pages, toc = extract_text_pdf(raw)
                    tables = extract_tables_pdf(raw)
                    images = extract_images_pdf(raw, doc_id)
                    diagrams = extract_diagrams_pdf(raw, doc_id)
                    examples = extract_examples_from_text(pages, doc_id)
                    meta_doc = {"doc_id": doc_id, "file_name": fname}
                    chunks, cm = chunk_document_pages(pages, meta_doc)
                    build_or_load_faiss(chunks, cm)
                    metadata[doc_id] = {
                        "file_name": fname,
                        "pages": len(pages),
                        "tables": bool(tables),
                        "images": images,
                        "diagrams": diagrams,
                        "examples": examples
                    }
                    save_metadata(metadata)
                    st.success(f"Ingested {f.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
            progress.empty()

# ------------------ Documents Page ------------------
elif page == "Documents":
    st.header("Documents")
    if not metadata:
        st.info("No documents yet.")
    else:
        for doc_id, info in metadata.items():
            with st.expander(f"{info['file_name']} ({doc_id})"):
                tabs = st.tabs(["📄 Text", "📊 Tables", "🖼️ Diagrams", "💡 Examples"])

                with tabs[1]:
                    tbl_path = TABLES_DIR / f"{doc_id}_tables.json"
                    if tbl_path.exists():
                        tbls = json.loads(tbl_path.read_text(encoding="utf8"))
                        for t in tbls:
                            st.dataframe(pd.DataFrame(t["rows"]))
                    else:
                        st.info("No tables found.")

                with tabs[2]:
                    diag_path = IMAGES_DIR / f"{doc_id}_diagrams.json"
                    if diag_path.exists():
                        diags = json.loads(diag_path.read_text(encoding="utf8"))
                        for d in diags:
                            st.image(d["file"], width=300, caption=f"{d['type']} — Page {d['page']}")
                            st.write(d["caption"])
                    else:
                        st.info("No diagrams found.")

                with tabs[3]:
                    ex_path = DATA_DIR / f"{doc_id}_examples.json"
                    if ex_path.exists():
                        exs = json.loads(ex_path.read_text(encoding="utf8"))
                        for e in exs:
                            st.markdown(f"**{e['title']} (Page {e['page']})**")
                            st.write(e["content"])
                    else:
                        st.info("No examples detected.")

# ------------------ Search & QA ------------------
elif page == "Search & QA":
    st.header("Search / Ask Questions")
    query = st.text_input("Enter query")
    k = st.slider("Top k", 1, 10, 5)
    if st.button("Search"):
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=k)
        for d in docs:
            md = d.metadata
            st.markdown(f"**Doc:** {md['source']} (Page {md['page']})")
            st.write(postprocess_extracted_text(d.page_content)[:800])
            st.divider()
    if st.button("Ask (RAG)"):
        if not groq_key:
            st.error("Enter Groq API Key")
        else:
            embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=k)
            answer = ask_groq_with_docs(docs, query, groq_key)
            st.subheader("Answer")
            st.markdown(answer)

# ------------------ Admin ------------------
elif page == "Admin & Export":
    st.header("Admin & Export")
    if st.button("Rebuild Index"):
        all_chunks, all_meta = [], []
        for doc_id, info in metadata.items():
            fpath = DOCS_DIR / info["file_name"]
            if not fpath.exists():
                continue
            raw = fpath.read_bytes()
            pages, _ = extract_text_pdf(raw)
            ch, cm = chunk_document_pages(pages, {"doc_id": doc_id, "file_name": info["file_name"]})
            all_chunks += ch
            all_meta += cm
        build_or_load_faiss(all_chunks, all_meta)
        st.success("Rebuilt index.")

    st.write("Docs:", len(metadata))
    st.write("Images:", len(list(IMAGES_DIR.glob('*'))))
    st.write("Tables:", len(list(TABLES_DIR.glob('*'))))
