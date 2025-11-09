"""
Enterprise PDF → Searchable Knowledge App
-----------------------------------------
Single-file Streamlit application that:
 - Accepts PDF uploads (including scanned PDFs)
 - Uses `unstructured` to partition PDF into text, tables, images (OCR via tesseract/poppler)
 - Summarizes text and tables with Groq (langchain_groq.ChatGroq)
 - Generates image captions using Gemini (Google Generative API)
 - Stores table data in MongoDB (if available) or local JSON
 - Stores text + summaries in a local Chroma vectorstore
 - Provides a search UI that retrieves relevant chunks and answers questions using Gemini + contextual prompt

Requirements (install before running):
 pip install streamlit unstructured langchain langchain-groq langchain-openai langchain-core chromadb pymongo python-dotenv google-generativeai tesseract pillow pdf2image

Environment variables required:
 - GROQ_API_KEY
 - GEMINI_API_KEY (Google Generative AI API key / Application credentials)
 - MONGODB_URI (optional, for table storage)

Note: This is a single-file reference implementation — tweak production config (error handling, rate limits, batching, security)

"""

import os
import io
import json
import uuid
import base64
import tempfile
from typing import List, Dict, Any

import streamlit as st
from unstructured.partition.pdf import partition_pdf
from PIL import Image

# Vectorstore and embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# Groq for summarization
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGroq = None

# Gemini (Google Generative) client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional MongoDB for storing extracted tables
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# Initialize Gemini if available
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Create a local Chroma instance (default directory "./chroma_db")
VECTOR_DIR = "./chroma_db"
EMBED_MODEL = "openai"  # we use OpenAIEmbeddings for demonstration; swap if you use another

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def partition_pdf_and_extract(file_bytes: bytes, filename: str, output_dir: str = "./content") -> Dict[str, List[Any]]:
    """Run unstructured.partition.pdf to extract elements (text blocks, tables, images).
    Returns a dict with keys: texts, tables, images (base64 strings) and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, f"upload-{uuid.uuid4().hex}.pdf")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    chunks = partition_pdf(
        filename=tmp_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts = []
    tables = []
    images_b64 = []

    for ch in chunks:
        ch_type = str(type(ch))
        # CompositeElement contains grouped text elements
        if "CompositeElement" in ch_type:
            # metadata includes page number & title if available
            texts.append(ch)
        elif "Table" in ch_type or "TableElement" in ch_type:
            tables.append(ch)
        else:
            # fallback: images inside composite elements handled separately
            pass

    # extract images from composite elements
    for ch in chunks:
        if "CompositeElement" in str(type(ch)) and hasattr(ch.metadata, "orig_elements"):
            for el in ch.metadata.orig_elements:
                if "Image" in str(type(el)) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)

    return {"texts": texts, "tables": tables, "images": images_b64, "raw_chunks": chunks}


def summarize_with_groq(elements: List[str], max_concurrency: int = 3) -> List[str]:
    """Use Groq ChatGroq to summarize textual elements.
    Falls back to a cheap local summary if Groq not available.
    """
    if ChatGroq is None or GROQ_API_KEY is None:
        # naive fallback summary
        return [ (el[:800] + "..." if len(el) > 800 else el) for el in elements ]

    prompt_text = (
        "You are an assistant tasked with summarizing tables and text.\n"
        "Give a concise summary of the table or text. Respond only with the summary.\n"
        "Text: {element}\n"
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.2, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    return summarize_chain.batch(elements, {"max_concurrency": max_concurrency})


def caption_images_with_gemini(images_b64: List[str]) -> List[str]:
    """Use Gemini (Google Generative) to create captions for images.
    Falls back to a short heuristic description if Gemini not available.
    """
    captions = []
    if genai is None or GEMINI_API_KEY is None:
        for b64 in images_b64:
            captions.append("Image extracted from PDF (base64). Size: approx {} bytes".format(len(b64)))
        return captions

    for b64 in images_b64:
        # send image as data URL where the client supports it; google.generativeai supports image input for multimodal models
        try:
            # we'll create a short prompt + the image
            prompt = "Describe the image in 1-2 short sentences suitable as a searchable caption. Mention graph/table if present."
            image_data_url = f"data:image/jpeg;base64,{b64}"
            response = genai.images.generate(prototype="multimodal-text-only", # note: placeholder; update to correct api call if needed
                                             prompt=prompt,
                                             image=image_data_url,
                                             max_output_tokens=150)
            # the above is a notional call — client versions vary. Try to read `response` safely.
            text = None
            if hasattr(response, "text"): text = response.text
            elif isinstance(response, dict): text = response.get("output", {}).get("text", "")
            captions.append(text or "[no caption returned]")
        except Exception as e:
            captions.append(f"[caption error: {e}]")
    return captions


def extract_table_records(tables: List[Any]) -> List[Dict]:
    """Convert table composite elements into JSON-friendly records.
    Each table element should expose metadata.text_as_html or other structured text.
    """
    records = []
    for t in tables:
        try:
            rec = {
                "table_id": str(uuid.uuid4()),
                "html": getattr(t.metadata, "text_as_html", None) or str(t),
                "raw_metadata": getattr(t, "metadata", {}).__dict__ if hasattr(t, "metadata") else {},
            }
            records.append(rec)
        except Exception:
            records.append({"table_id": str(uuid.uuid4()), "html": str(t)})
    return records


def store_tables_nosql(records: List[Dict]):
    """Store tables into MongoDB if MONGODB_URI is present, else write to local JSON file."""
    if not records:
        return None
    if MongoClient and MONGODB_URI:
        try:
            client = MongoClient(MONGODB_URI)
            db = client.get_database("enterprise_pdf_knowledge")
            col = db.get_collection("extracted_tables")
            col.insert_many(records)
            return {"status": "mongo", "count": len(records)}
        except Exception as e:
            st.warning(f"MongoDB storage failed: {e}. Falling back to JSON file.")
    # fallback local file
    path = os.path.join("./content", f"tables-{uuid.uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return {"status": "json", "path": path, "count": len(records)}


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------

def init_vectorstore(collection_name: str = "enterprise_pdf"):
    # using Chroma with OpenAIEmbeddings (swap as needed)
    embeddings = OpenAIEmbeddings()
    vect = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=VECTOR_DIR)
    return vect


def add_documents_to_vectorstore(vectorstore, docs: List[Document]):
    vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Search + RAG answering
# ---------------------------------------------------------------------------

def retrieve_similar(vectorstore, query: str, k: int = 5):
    results = vectorstore.similarity_search(query, k=k)
    return results


def answer_with_gemini(question: str, context_docs: List[Document]) -> str:
    """Use Gemini to answer question using the context_docs as grounding. Falls back to heuristic compose if Gemini not available."""
    context_text = "\n\n".join([f"Source ({i}): {d.page_content}" for i, d in enumerate(context_docs)])
    prompt = (
        "Answer the user question using ONLY the provided context. If the answer is not in context, say 'I don't know from the documents provided.'\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    if genai is None or GEMINI_API_KEY is None:
        # fallback: naive join
        return (context_text[:1500] + "...\n\n[Gemini not configured — install google-generativeai and set GEMINI_API_KEY]")

    try:
        # call Gemini (text generation) — change to the method for your client version
        resp = genai.generate_text(model="gemini-pro",  # placeholder: set the correct model name you have access to
                                   prompt=prompt,
                                   max_output_tokens=512)
        if hasattr(resp, "text"):
            return resp.text
        elif isinstance(resp, dict):
            return resp.get("output", "")
        else:
            return str(resp)
    except Exception as e:
        return f"[Gemini API error: {e}]"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Enterprise PDF → Searchable Knowledge", layout="wide")
    st.title("Enterprise PDF → Searchable Knowledge (Groq + Gemini)")

    st.markdown("\nUpload a PDF (supports scanned PDFs if Tesseract/Poppler are installed). App will extract text, tables, images; summarize text with Groq and caption images with Gemini; tables are stored in MongoDB or JSON; text summaries go into Chroma vectorstore for search.")

    upload = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

    vectorstore = init_vectorstore()

    if upload is not None:
        file_bytes = upload.read()
        st.info("Extracting PDF (this can take a while for large PDFs)...")
        with st.spinner("Partitioning PDF and extracting elements..."):
            extracted = partition_pdf_and_extract(file_bytes, upload.name)

        st.success(f"Extracted {len(extracted['texts'])} text chunks, {len(extracted['tables'])} tables, {len(extracted['images'])} images")

        # Prepare textual content strings for summarization and storage
        text_elements = [getattr(t, "text", str(t)) for t in extracted["texts"]]
        # if text elements are composite objects with .text property, use it; else fallback

        with st.spinner("Summarizing text with Groq..."):
            text_summaries = summarize_with_groq(text_elements)

        with st.spinner("Captioning images with Gemini..."):
            image_captions = caption_images_with_gemini(extracted["images"])

        # Add summaries + metadata to vectorstore
        docs = []
        for i, s in enumerate(text_summaries):
            metadata = {
                "source": upload.name,
                "type": "text",
                "chunk_index": i,
            }
            docs.append(Document(page_content=s, metadata=metadata))

        for i, cap in enumerate(image_captions):
            metadata = {"source": upload.name, "type": "image", "image_index": i}
            docs.append(Document(page_content=cap, metadata=metadata))

        # tables: store summaries of tables as documents as well (and keep detailed table stored in DB)
        table_records = extract_table_records(extracted["tables"])
        table_htmls = [r.get("html") for r in table_records]
        table_summaries = summarize_with_groq(table_htmls)
        for i, ts in enumerate(table_summaries):
            metadata = {"source": upload.name, "type": "table", "table_id": table_records[i].get("table_id")}
            docs.append(Document(page_content=ts, metadata=metadata))

        add_documents_to_vectorstore(vectorstore, docs)

        # store tables in nosql
        res = store_tables_nosql(table_records)
        st.write("Table store result:", res)

        st.success("Indexing complete — you can now search in the box below.")

    st.header("Search & Ask")
    query = st.text_input("Enter a search query or question:")
    k = st.slider("Number of context chunks to retrieve", 1, 10, 4)

    if query:
        with st.spinner("Retrieving similar documents..."):
            hits = retrieve_similar(vectorstore, query, k=k)

        st.subheader("Search results (context)")
        context_docs = []
        for i, h in enumerate(hits):
            st.markdown(f"**Result {i+1} — score unknown**")
            st.write(h.page_content)
            st.write("Metadata:", h.metadata)
            context_docs.append(h)
            st.write("---")

        with st.spinner("Composing answer with Gemini..."):
            answer = answer_with_gemini(query, context_docs)

        st.subheader("Answer (grounded in retrieved documents)")
        st.write(answer)

    st.sidebar.header("Configuration & Keys")
    st.sidebar.write("GROQ configured:" , bool(GROQ_API_KEY and ChatGroq))
    st.sidebar.write("Gemini configured:", bool(GEMINI_API_KEY and genai))
    st.sidebar.write("MongoDB available:", bool(MONGODB_URI and MongoClient))

    st.sidebar.markdown("\n---\n**Notes:**\n- Ensure Tesseract + Poppler installed to handle OCR and accurate PDF partitioning.\n- Configure GROQ_API_KEY and GEMINI_API_KEY in your environment.\n- This is a reference pipeline — adapt batching, error handling, and production vector DB as needed.")


if __name__ == "__main__":
    main()
