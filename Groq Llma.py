# app.py
import streamlit as st
import os
import re
import json
import time
import traceback
import requests
from io import BytesIO
from pathlib import Path
from typing import List, Dict

# extraction libs
from bs4 import BeautifulSoup
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
import pdfplumber
import fitz  # PyMuPDF

# embeddings / vectorstore / llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import pandas as pd
import uuid

# -------------------- Config & data folders --------------------
BASE = Path.cwd()
DATA_DIR = BASE / "data"
DOCS_DIR = DATA_DIR / "docs"
TABLES_DIR = DATA_DIR / "tables"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.json"
FAISS_DIR = DATA_DIR / "faiss_index"

for d in (DATA_DIR, DOCS_DIR, TABLES_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Utility helpers --------------------
def load_metadata() -> Dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf8"))
    return {}

def save_metadata(meta: Dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf8")

def make_doc_id() -> str:
    return str(uuid.uuid4())[:12]

def safe_text(s):
    return re.sub(r'[\ud800-\udfff]', '', s or '')

# -------------------- PDF processing helpers --------------------
def extract_text_pdf(file_bytes: bytes):
    """
    Extract text page-by-page using PyPDF2 + fallback to pdfplumber for tables + OCR
    Returns: pages: list[str], toc (best-effort)
    """
    pages = []
    toc = []
    # PyPDF2 extraction
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = safe_text(text)
            pages.append(text)
        # try to get outline/toc
        try:
            outline = reader.outline
            # outline is complex; we produce a simple list
            toc = [str(x.title) for x in outline] if hasattr(reader, "outline") else []
        except Exception:
            toc = []
    except Exception:
        pages = []

    # if pages appear empty, fall back to pdfplumber + OCR
    if not any(pages):
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages = []
            for p in pdf.pages:
                txt = p.extract_text() or ""
                if not txt.strip():
                    # use OCR on rasterized image of page
                    im = p.to_image(resolution=200).original
                    txt = pytesseract.image_to_string(im)
                pages.append(safe_text(txt))
    return pages, toc

def extract_tables_pdf(file_bytes: bytes):
    """
    Extract tables using pdfplumber, return a list of tables as list-of-rows dicts.
    """
    tables_all = []
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    for t in tables:
                        # convert table (list of rows) to list of dicts if header exists
                        if not t:
                            continue
                        header = t[0]
                        rows = []
                        for row in t[1:]:
                            rowd = {}
                            for j, cell in enumerate(row):
                                key = header[j] if j < len(header) else f"col_{j}"
                                rowd[str(key or j)] = cell
                            rows.append(rowd)
                        tables_all.append({"page": i, "header": header, "rows": rows})
                except Exception:
                    continue
    except Exception:
        pass
    return tables_all

def extract_images_pdf(file_bytes: bytes, doc_id: str):
    """
    Extract images from PDF using PyMuPDF, save them and return list of image metadata.
    """
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
                # caption via OCR (lightweight)
                try:
                    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    caption = pytesseract.image_to_string(pil).strip()[:200]
                    if not caption:
                        caption = "image (no textual OCR) — visual content"
                except Exception:
                    caption = "image (unable to OCR)"
                imgs.append({"file": str(p), "page": page_i + 1, "caption": caption})
    except Exception:
        pass
    return imgs

# -------------------- Chunking & Embeddings --------------------
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

def chunk_document(pages: List[str], doc_meta: Dict):
    """
    Returns chunks: list[str], chunk_metadata: list[dict]
    We chunk per page, but attempt to keep heading boundaries: naive approach uses first 200 chars as heading if short.
    """
    text = "\n\n".join([f"[Page {i+1}]\n{p}" for i, p in enumerate(pages)])
    texts = SPLITTER.split_text(text)
    metadatas = []
    for i, t in enumerate(texts):
        md = {
            "doc_id": doc_meta["doc_id"],
            "page": None,
            "paragraph": i + 1,
            "source": doc_meta["file_name"],
        }
        # attempt to extract page number from chunk header
        m = re.search(r"\[Page (\d+)\]", t)
        if m:
            md["page"] = int(m.group(1))
        metadatas.append(md)
    return texts, metadatas

def build_or_load_faiss(chunks: List[str], metadatas: List[dict]):
    """
    Create or update FAISS index stored in FAISS_DIR.
    """
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        try:
            # try loading existing index and add new vectors
            db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
            # if new chunks present, add them
            if chunks:
                db.add_texts(chunks, metadatas=metadatas)
                try:
                    db.save_local(str(FAISS_DIR))
                except Exception:
                    pass
            return db
        except Exception:
            # fallback to creating new
            pass
    # create new index
    db = FAISS.from_texts(chunks or ["dummy"], embedding=embed, metadatas=metadatas or [{"doc_id":"dummy","page":0,"paragraph":0,"source":"none"}])
    try:
        db.save_local(str(FAISS_DIR))
    except Exception:
        pass
    return db

# -------------------- RAG / LLM helpers --------------------
QA_PROMPT = """You are a precise document QA assistant. Use ONLY the provided context to answer the question. Cite every fact in the form (Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source]). If the answer cannot be found in the context, reply exactly: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer (bullet points preferred):
"""

def ask_groq_with_docs(docs, question, model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=None):
    """
    docs: list of langchain Document objects OR dicts with metadata and page_content fields.
    We'll build a single context string (careful with length).
    """
    # Build context with metadata & snippets
    parts = []
    for d in docs:
        meta = getattr(d, "metadata", d.get("metadata") if isinstance(d, dict) else {})
        content = getattr(d, "page_content", d.get("page_content") if isinstance(d, dict) else str(d))
        md_docid = meta.get("doc_id", meta.get("source", "unknown"))
        md_page = meta.get("page", meta.get("page", "NA"))
        md_para = meta.get("paragraph", meta.get("paragraph", "NA"))
        src = meta.get("source", "unknown")
        header = f"(Document ID: {md_docid}, Page: {md_page}, Paragraph: {md_para}, Source: {src})"
        parts.append(header + "\n" + content)
    # limit context length naive: join top k
    context = "\n\n---\n\n".join(parts)
    prompt = PromptTemplate.from_template(QA_PROMPT)
    formatted = prompt.format(context=context, question=question)

    # build model
    model = ChatGroq(model_name=model_name, temperature=temperature, api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
    # invoke
    try:
        result = model.invoke(formatted)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        return f"LLM error: {e}"

# -------------------- Streamlit UI (multi-page via sidebar) --------------------
st.set_page_config(page_title="Enterprise PDF → Knowledge", page_icon="📚", layout="wide")

st.title("📚 Enterprise PDF → Searchable Knowledge")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Ingest", "Documents", "Search & QA", "Admin & Export"])

# GROQ API key input
st.sidebar.markdown("### LLM Configuration")
groq_key = st.sidebar.text_input("Groq API Key (optional for QA)", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

metadata = load_metadata()

if page == "Upload & Ingest":
    st.header("Upload PDFs")
    st.markdown("Upload PDF documents (digital or scanned). We'll extract text, tables, images, chunk & embed for semantic search.")
    uploaded = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True)
    department = st.text_input("Department / Tag (optional)")
    process_btn = st.button("Ingest uploaded files")

    if process_btn and uploaded:
        progress = st.progress(0)
        total = len(uploaded)
        cur = 0
        for f in uploaded:
            cur += 1
            progress.progress(cur / total)
            try:
                raw = f.read()
                doc_id = make_doc_id()
                fname = f"{doc_id}_{f.name}"
                (DOCS_DIR / fname).write_bytes(raw)
                # extract
                pages, toc = extract_text_pdf(raw)
                tables = extract_tables_pdf(raw)
                images = extract_images_pdf(raw, doc_id)
                # chunk
                doc_meta = {"doc_id": doc_id, "file_name": fname, "uploaded_at": time.time(), "toc": toc, "department": department}
                chunks, ch_meta = chunk_document(pages, doc_meta)
                # build or update faiss
                db = build_or_load_faiss(chunks, ch_meta)
                # save tables to JSON
                if tables:
                    tbl_file = TABLES_DIR / f"{doc_id}_tables.json"
                    tbl_file.write_text(json.dumps(tables, indent=2), encoding="utf8")
                # save images metadata
                # update metadata
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
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
                st.exception(traceback.format_exc())
        progress.empty()
        st.balloons()

elif page == "Documents":
    st.header("Documents")
    st.markdown("List of ingested documents and content preview.")
    if not metadata:
        st.info("No documents ingested yet.")
    else:
        # list docs
        for doc_id, info in metadata.items():
            with st.expander(f"{info['file_name']} (id: {doc_id})"):
                st.write("Pages:", info.get("pages"))
                st.write("Uploaded at:", time.ctime(info.get("uploaded_at")))
                st.write("Has tables:", info.get("tables"))
                # show images
                if info.get("images"):
                    st.write("Images and captions:")
                    for im in info.get("images", []):
                        try:
                            st.image(im["file"], width=300)
                        except Exception:
                            st.write(im)
                        st.write("Caption:", im.get("caption"))
                # preview extracted text: load doc bytes and show first pages' text from FAISS entries
                if FAISS_DIR.exists():
                    try:
                        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                        # naive retrieval of chunks by metadata doc_id
                        docs = db.max_marginal_relevance_search_by_vector("", k=3) if hasattr(db, "max_marginal_relevance_search_by_vector") else []
                    except Exception:
                        docs = []
                # show table samples
                tbl_path = TABLES_DIR / f"{doc_id}_tables.json"
                if tbl_path.exists():
                    st.write("Extracted Tables (sample):")
                    try:
                        tables = json.loads(tbl_path.read_text(encoding="utf8"))
                        for t in tables[:2]:
                            df = pd.DataFrame(t["rows"])
                            st.dataframe(df)
                    except Exception:
                        st.write("Unable to render table preview.")

elif page == "Search & QA":
    st.header("Search & QA")
    st.markdown("Perform semantic search or ask a question across ingested documents.")

    query = st.text_input("Enter your query here")
    k = st.slider("Top k results", 1, 10, 5)
    run_search = st.button("Search")
    run_qa = st.button("Ask (RAG)")

    if run_search and query.strip():
        # load faiss
        try:
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=k)
            st.subheader("Search results (top {})".format(len(docs)))
            for d in docs:
                md = d.metadata
                st.markdown(f"**Source:** {md.get('source')} — DocID: {md.get('doc_id')} — Page: {md.get('page')}")
                st.write(d.page_content[:800])
                st.markdown("---")
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.exception(traceback.format_exc())

    if run_qa and query.strip():
        # run retrieval then LLM
        if not os.environ.get("GROQ_API_KEY"):
            st.error("No Groq API key set. Enter it in the sidebar to enable QA.")
        else:
            try:
                embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.load_local(str(FAISS_DIR), embed, allow_dangerous_deserialization=True)
                docs = db.similarity_search(query, k=k)
                st.info(f"Retrieved {len(docs)} context chunks. Calling LLM...")
                answer = ask_groq_with_docs(docs, query, groq_api_key=os.environ.get("GROQ_API_KEY"))
                st.subheader("Answer")
                st.markdown(answer)
            except Exception as e:
                st.error(f"QA failed: {e}")
                st.exception(traceback.format_exc())

elif page == "Admin & Export":
    st.header("Admin & Export")
    st.markdown("Manage local data, export indices and tables.")

    if st.button("Rebuild FAISS from scratch using all stored docs"):
        # naive rebuild: read docs directory, re-ingest all docs present in metadata
        try:
            all_chunks = []
            all_meta = []
            for doc_id, info in metadata.items():
                fpath = DOCS_DIR / info["file_name"]
                if not fpath.exists():
                    continue
                raw = fpath.read_bytes()
                pages, _ = extract_text_pdf(raw)
                chunks, ch_meta = chunk_document(pages, {"doc_id": doc_id, "file_name": info["file_name"]})
                all_chunks += chunks
                all_meta += ch_meta
            db = build_or_load_faiss(all_chunks, all_meta)
            st.success("Rebuilt FAISS index.")
        except Exception as e:
            st.error(f"Failed to rebuild FAISS: {e}")
            st.exception(traceback.format_exc())

    if st.button("Export all tables (zip)"):
        # create zip of TABLES_DIR
        import zipfile
        zip_path = DATA_DIR / "tables_export.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for f in TABLES_DIR.glob("*.json"):
                z.write(f, arcname=f.name)
        with open(zip_path, "rb") as rf:
            st.download_button("Download tables_export.zip", rf.read(), file_name="tables_export.zip")

    st.markdown("Local storage stats:")
    st.write("Documents:", len(list(DOCS_DIR.glob("*"))))
    st.write("Tables files:", len(list(TABLES_DIR.glob("*.json"))))
    st.write("Images:", len(list(IMAGES_DIR.glob("*"))))
    st.write("FAISS index exists:", FAISS_DIR.exists() and any(FAISS_DIR.iterdir()))

# Footer
st.markdown("---")
st.markdown("Built for converting enterprise PDFs into searchable knowledge — OCR, table extraction, chunking & semantic search.")
