"""
Enterprise PDF → Searchable Knowledge App (Groq + Gemini)
---------------------------------------------------------
Single-file Streamlit application that:
 - Accepts PDF uploads (including scanned PDFs)
 - Uses `unstructured` to partition PDFs into text, tables, and images (with OCR)
 - Summarizes text and tables using Groq (ChatGroq)
 - Generates image captions using Gemini (Google Generative AI)
 - Stores structured table data in MongoDB (if connected) or local JSON
 - Creates embeddings using HuggingFace (open-source)
 - Saves text + summaries to a Chroma vector database
 - Lets you input API keys directly inside the Streamlit app

"""

import os
import json
import uuid
from typing import List, Dict, Any

import streamlit as st
from unstructured.partition.pdf import partition_pdf
from PIL import Image

# Vectorstore and embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Gemini API
try:
    import google.generativeai as genai
except Exception:
    genai = None

# MongoDB
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def partition_pdf_and_extract(file_bytes: bytes, filename: str, output_dir: str = "./content") -> Dict[str, List[Any]]:
    """Extract text, tables, and images from a PDF using Unstructured."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"upload-{uuid.uuid4().hex}.pdf")
    with open(path, "wb") as f:
        f.write(file_bytes)

    chunks = partition_pdf(
        filename=path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
    )

    texts, tables, images_b64 = [], [], []

    for ch in chunks:
        ch_type = str(type(ch))
        if "CompositeElement" in ch_type:
            texts.append(ch)
        elif "Table" in ch_type:
            tables.append(ch)

    # extract base64 images
    for ch in chunks:
        if "CompositeElement" in str(type(ch)) and hasattr(ch.metadata, "orig_elements"):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)

    return {"texts": texts, "tables": tables, "images": images_b64}


def summarize_with_groq(elements: List[str], groq_api_key: str, model_name: str = "llama-3.1-8b-instant") -> List[str]:
    """Summarize PDF chunks using Groq LLM."""
    if not groq_api_key:
        return [el[:800] + "..." if len(el) > 800 else el for el in elements]

    try:
        model = ChatGroq(temperature=0.2, model=model_name, groq_api_key=groq_api_key)
        prompt = ChatPromptTemplate.from_template(
            "Summarize this content clearly and concisely:\n\n{element}"
        )
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        return summarize_chain.batch(elements, {"max_concurrency": 3})
    except Exception as e:
        return [f"[Groq Error: {e}] {el[:500]}" for el in elements]


def caption_images_with_gemini(images_b64: List[str], gemini_api_key: str) -> List[str]:
    """Generate captions for extracted images using Gemini."""
    if not gemini_api_key or genai is None:
        return [f"Image (base64) length={len(b64)}" for b64 in images_b64]

    try:
        genai.configure(api_key=gemini_api_key)
        captions = []
        for b64 in images_b64:
            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes))
            model = genai.GenerativeModel("gemini-pro-vision")
            response = model.generate_content(["Describe this image for document search:", image])
            captions.append(response.text or "[No caption generated]")
        return captions
    except Exception as e:
        return [f"[Gemini caption error: {e}]"]


def extract_table_records(tables: List[Any]) -> List[Dict]:
    """Convert tables into structured records."""
    records = []
    for t in tables:
        rec = {
            "table_id": str(uuid.uuid4()),
            "html": getattr(t.metadata, "text_as_html", None) or str(t),
            "metadata": getattr(t, "metadata", {}).__dict__ if hasattr(t, "metadata") else {},
        }
        records.append(rec)
    return records


def store_tables_nosql(records: List[Dict], mongo_uri: str):
    """Store tables in MongoDB (if URI provided) or fallback JSON."""
    if not records:
        return None
    if mongo_uri and MongoClient:
        try:
            client = MongoClient(mongo_uri)
            db = client.get_database("enterprise_pdf_knowledge")
            db.extracted_tables.insert_many(records)
            return {"status": "mongo", "count": len(records)}
        except Exception as e:
            st.warning(f"MongoDB error: {e}")
    path = os.path.join("./content", f"tables-{uuid.uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return {"status": "json", "path": path}


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def init_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vect = Chroma(collection_name="enterprise_pdf", embedding_function=embeddings, persist_directory="./chroma_db")
    return vect


# ---------------------------------------------------------------------------
# Gemini Q&A
# ---------------------------------------------------------------------------

def answer_with_gemini(question: str, context_docs: List[Document], gemini_api_key: str) -> str:
    """Answer queries based on context using Gemini."""
    context_text = "\n\n".join([f"Source ({i}): {d.page_content}" for i, d in enumerate(context_docs)])
    if not gemini_api_key or genai is None:
        return "[Gemini not configured] " + context_text[:1000]

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = (
            f"Answer the following question strictly using this context:\n\n{context_text}\n\n"
            f"Question: {question}"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini Error: {e}]"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Enterprise PDF → Searchable Knowledge", layout="wide")
    st.title("📘 Enterprise PDF → Searchable Knowledge (Groq + Gemini)")

    # --- Sidebar: API Key Inputs ---
    st.sidebar.header("🔑 API Configuration")
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    mongo_uri = st.sidebar.text_input("MongoDB URI (optional)", type="password")

    st.sidebar.markdown("---")
    st.sidebar.info("Enter your API keys above to activate the full pipeline.")

    # --- PDF Upload ---
    upload = st.file_uploader("📄 Upload PDF file", type=["pdf"])
    vectorstore = init_vectorstore()

    if upload:
        st.info("⏳ Processing uploaded PDF...")
        pdf_bytes = upload.read()

        with st.spinner("Extracting content from PDF..."):
            extracted = partition_pdf_and_extract(pdf_bytes, upload.name)

        st.success(f"✅ Extracted {len(extracted['texts'])} text blocks, {len(extracted['tables'])} tables, {len(extracted['images'])} images.")

        text_blocks = [getattr(t, "text", str(t)) for t in extracted["texts"]]

        with st.spinner("Summarizing with Groq..."):
            summaries = summarize_with_groq(text_blocks, groq_api_key)

        with st.spinner("Captioning images with Gemini..."):
            captions = caption_images_with_gemini(extracted["images"], gemini_api_key)

        table_records = extract_table_records(extracted["tables"])
        with st.spinner("Summarizing tables with Groq..."):
            table_summaries = summarize_with_groq([r["html"] for r in table_records], groq_api_key)

        docs = []
        for i, s in enumerate(summaries):
            docs.append(Document(page_content=s, metadata={"type": "text", "index": i}))
        for i, c in enumerate(captions):
            docs.append(Document(page_content=c, metadata={"type": "image", "index": i}))
        for i, t in enumerate(table_summaries):
            docs.append(Document(page_content=t, metadata={"type": "table", "table_id": table_records[i]["table_id"]}))

        vectorstore.add_documents(docs)
        vectorstore.persist()

        res = store_tables_nosql(table_records, mongo_uri)
        st.write("Table store result:", res)
        st.success("🎯 All data processed and indexed successfully!")

    # --- Search Section ---
    st.header("🔍 Search & Ask")
    query = st.text_input("Enter your question or search query:")
    k = st.slider("Number of context chunks to retrieve", 1, 10, 4)

    if query:
        with st.spinner("Searching..."):
            results = vectorstore.similarity_search(query, k=k)
        st.subheader("📄 Retrieved Contexts")
        for i, doc in enumerate(results):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:500]}...")
        with st.spinner("Generating answer with Gemini..."):
            answer = answer_with_gemini(query, results, gemini_api_key)
        st.subheader("🧠 Gemini Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
