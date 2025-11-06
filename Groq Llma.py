import streamlit as st
import os
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
import pandas as pd
import uuid

# ✅ Modern LangChain 2025+ Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI Document Chatbot", page_icon="📚", layout="wide")

st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body{font-family:'Inter',sans-serif;background-color:#f9fafb;}
        .stButton>button{background-color:#2563eb;color:white;border-radius:8px;padding:0.5rem 1rem;font-weight:500;}
        .stButton>button:hover{background-color:#1d4ed8;}
        .answer-table{width:100%;border-collapse:collapse;margin-top:1rem;}
        .answer-table th,.answer-table td{border:1px solid #e5e7eb;padding:0.75rem;text-align:left;}
        .answer-table th{background:#f3f4f6;font-weight:600;color:#374151;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="text-center">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">📚 AI Document Chatbot</h1>
        <p class="text-gray-600">Upload documents or URLs, ask questions, and get sourced answers with themes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Required for Groq LLaMA services.")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs / images / text files", type=["pdf", "jpg", "png", "txt"], accept_multiple_files=True
)
url_input = st.sidebar.text_input("Enter URL to fetch content from:", placeholder="https://example.com")


# -------------------- Helpers --------------------
@st.cache_data
def extract_text_from_file(file, name):
    try:
        text, metadata = "", []
        if name.lower().endswith(".pdf"):
            pdf = PdfReader(file)
            for p, page in enumerate(pdf.pages, 1):
                page_text = re.sub(r"[\ud800-\udfff]", "", page.extract_text() or "")
                text += page_text
                metadata.append({"doc_id": str(uuid.uuid4()), "page": p, "text": page_text, "source": name})
        elif name.lower().endswith((".jpg", ".png", ".jpeg")):
            img = Image.open(file)
            page_text = re.sub(r"[\ud800-\udfff]", "", pytesseract.image_to_string(img))
            text += page_text
            metadata.append({"doc_id": str(uuid.uuid4()), "page": 1, "text": page_text, "source": name})
        elif name.lower().endswith(".txt"):
            raw = file.read()
            page_text = re.sub(r"[\ud800-\udfff]", "", raw.decode("utf-8", errors="ignore"))
            text += page_text
            metadata.append({"doc_id": str(uuid.uuid4()), "page": 1, "text": page_text, "source": name})
        return text, metadata
    except Exception as e:
        st.error(f"Error processing {name}: {e}")
        return "", []


@st.cache_data
def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = "\n".join(t.get_text(strip=True) for t in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5"]))
        text = re.sub(r"[\ud800-\udfff]", "", text)
        meta = [{"doc_id": str(uuid.uuid4()), "page": 1, "text": text, "source": url}]
        return text, meta
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return "", []


@st.cache_data
def get_text_chunks(text, meta):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    chunk_meta = []
    for i, ch in enumerate(chunks):
        md = meta[i % len(meta)].copy()
        md["paragraph"] = i + 1
        chunk_meta.append(md)
    return chunks, chunk_meta


@st.cache_resource
def get_vector_store(chunks, meta):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embedding=embed, metadatas=meta)
    try:
        store.save_local("faiss_index")
    except Exception:
        pass
    return store


# -------------------- Chain Builders --------------------
def get_qa_chain():
    """Groq LLaMA QA chain with custom prompt."""
    prompt_text = """
    You are a disciplined QA agent.  
    Use only the given context to answer the question in concise bullet points.  
    Cite each claim as (Document ID:[ID], Page:[X], Paragraph:[Y], Source:[S]).  
    If answer not found, reply exactly: "Answer is not available in the context."
    Context:\n{context}\nQuestion:\n{question}\nAnswer:
    """
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_theme_chain():
    """Groq LLaMA theme summarizer."""
    template = """
    Summarize recurring themes related to the question below from the provided context.  
    Use bullet points with citations (Document ID:[ID], Page:[X], Paragraph:[Y], Source:[S]).
    Context:\n{context}\nQuestion:\n{question}\nSummary:
    """
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# -------------------- UI Logic --------------------
st.header("💬 Ask a Question")
with st.form("qform"):
    user_q = st.text_area("Enter your question (press Ctrl + Enter to submit)", height=100)
    submit = st.form_submit_button("Submit Question")

if submit:
    if not api_key:
        st.warning("Please enter your Groq API key.")
    elif not (uploaded_files or url_input):
        st.warning("Please upload a file or enter a URL.")
    elif not user_q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing documents and generating answer..."):
            try:
                all_chunks, all_meta = [], []

                for f in uploaded_files or []:
                    t, m = extract_text_from_file(f, f.name)
                    if t:
                        c, cm = get_text_chunks(t, m)
                        all_chunks += c
                        all_meta += cm

                if url_input:
                    t, m = extract_text_from_url(url_input)
                    if t:
                        c, cm = get_text_chunks(t, m)
                        all_chunks += c
                        all_meta += cm

                if not all_chunks:
                    st.warning("No text found to process.")
                    st.stop()

                store = get_vector_store(all_chunks, all_meta)
                embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.load_local("faiss_index", embed, allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_q, k=5)

                qa_chain = get_qa_chain()
                ans = qa_chain({"input_documents": docs, "question": user_q}, return_only_outputs=True)
                st.subheader("Answer")
                st.markdown(ans["output_text"])

                st.subheader("Citations")
                df = pd.DataFrame(
                    [
                        {
                            "Document ID": d.metadata["doc_id"],
                            "Source": d.metadata["source"],
                            "Page": d.metadata["page"],
                            "Paragraph": d.metadata["paragraph"],
                            "Excerpt": (d.page_content[:200] + "...") if len(d.page_content) > 200 else d.page_content,
                        }
                        for d in docs
                    ]
                )
                st.markdown(df.to_html(index=False, classes="answer-table", escape=False), unsafe_allow_html=True)

                st.subheader("Recurring Themes")
                theme_chain = get_theme_chain()
                theme = theme_chain({"input_documents": docs, "question": user_q}, return_only_outputs=True)
                st.markdown(theme["output_text"])

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(traceback.format_exc())


st.markdown(
    """
    <div class="text-center text-gray-500 mt-8">
        <hr class="border-gray-200 mb-4">
        <p>Powered by LangChain Community, Groq LLaMA 3.3 & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)


