# Gemini.py (updated, 2025-compatible)
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
import traceback

# LangChain / Google Generative imports (modern)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
# ---- Streamlit UI setup ----
st.set_page_config(page_title="AI Document Chatbot", page_icon="📚", layout="wide")

st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f9fafb; }
        .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
        .stButton>button { background-color: #2563eb; color: white; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; }
        .stButton>button:hover { background-color: #1d4ed8; }
        .answer-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .answer-table th, .answer-table td { border: 1px solid #e5e7eb; padding: 0.75rem; text-align: left; }
        .answer-table th { background-color: #f3f4f6; font-weight: 600; color: #374151; }
        .answer-table td { background-color: #ffffff; color: #4b5563; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="text-center">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">📚 AI Document Chatbot</h1>
        <p class="text-gray-600">Upload documents or add URLs, ask questions, and get sourced answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: API Key, files, URL
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Required for Google Generative AI services.")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs, images (JPG/PNG), or text files", type=["pdf", "jpg", "png", "txt"], accept_multiple_files=True)

st.sidebar.header("🌐 Add URL")
url_input = st.sidebar.text_input("Enter a URL to fetch content from:", placeholder="https://example.com")

# ---------------- utility functions (cached) ----------------
@st.cache_data
def extract_text_from_file(file, file_name):
    try:
        text = ""
        metadata = []
        if file_name.lower().endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                page_text = re.sub(r'[\ud800-\udfff]', '', page_text)
                text += page_text + "\n"
                metadata.append({"doc_id": str(uuid.uuid4()), "page": page_num, "text": page_text, "source": file_name})
        elif file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img = Image.open(file)
            page_text = pytesseract.image_to_string(img)
            page_text = re.sub(r'[\ud800-\udfff]', '', page_text)
            text += page_text + "\n"
            metadata.append({"doc_id": str(uuid.uuid4()), "page": 1, "text": page_text, "source": file_name})
        elif file_name.lower().endswith(".txt"):
            raw = file.read()
            if isinstance(raw, bytes):
                page_text = raw.decode("utf-8", errors="ignore")
            else:
                page_text = str(raw)
            page_text = re.sub(r'[\ud800-\udfff]', '', page_text)
            text += page_text + "\n"
            metadata.append({"doc_id": str(uuid.uuid4()), "page": 1, "text": page_text, "source": file_name})
        return text, metadata
    except Exception as e:
        return "", []

@st.cache_data
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5']):
            t = tag.get_text(strip=True)
            t = re.sub(r'[\ud800-\udfff]', '', t)
            text += t + "\n"
        metadata = [{"doc_id": str(uuid.uuid4()), "page": 1, "text": text, "source": url}]
        return text, metadata
    except Exception:
        return "", []

@st.cache_data
def get_text_chunks(text, metadata):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        md = metadata[i % len(metadata)].copy()
        md["paragraph"] = i + 1
        chunk_metadata.append(md)
    return chunks, chunk_metadata

@st.cache_resource
def get_vector_store(chunks, chunk_metadata):
    # embeddings model name might need to match your installed langchain-google-genai package expectations
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # FAISS.from_texts usually expects list[str] and embeddings object; community implementation may vary
    vector_store = FAISS.from_texts(chunks, embedding=embeddings, metadatas=chunk_metadata)
    # persist locally for reuse
    try:
        vector_store.save_local("faiss_index")
    except Exception:
        # some FAISS wrappers have different save names; ignore if not supported
        pass
    return vector_store

# ---- small helper to call the LLM in a tolerant way ----
def call_gemini_model(model_instance, prompt_text, system_message=None):
    """
    Try a few common LangChain LLM call patterns so this works across versions.
    Returns text (str) or raises Exception.
    """
    # 1) If model_instance has a 'predict' method (common)
    try:
        return model_instance.predict(prompt_text)
    except Exception:
        pass

    # 2) If model_instance is callable (llm(prompt_text))
    try:
        out = model_instance(prompt_text)
        # if returns a dict-like, try to extract text
        if isinstance(out, str):
            return out
        try:
            return out["text"]
        except Exception:
            return str(out)
    except Exception:
        pass

    # 3) Try generate with HumanMessage
    try:
        message = HumanMessage(content=prompt_text)
        gen = model_instance.generate([[message]])
        # generation extraction varies; try common shapes
        if hasattr(gen, "generations"):
            gens = gen.generations
            # gens -> list of lists -> take first
            return gens[0][0].text
        # fallback to str
        return str(gen)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

# ---- Prompt templates (kept as strings for clarity) ----
QA_PROMPT_TEMPLATE = """
YOU ARE A HIGHLY DISCIPLINED, EVIDENCE-DRIVEN QUESTION-ANSWERING AGENT TRAINED TO EXTRACT, ANALYZE, AND REPORT INFORMATION *EXCLUSIVELY* FROM THE PROVIDED CONTEXT.
INSTRUCTIONS:
- ANSWER USING ONLY THE PROVIDED CONTEXT.
- USE BULLETS, PROVIDE CITATIONS IN FORMAT: (Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source])
- IF INFORMATION IS NOT PRESENT, RESPOND EXACTLY: "Answer is not available in the context."
Context:
{context}

Question:
{question}

Answer:
"""

THEME_PROMPT_TEMPLATE = """
You are a document synthesis expert. Given the context below and the question, extract recurring themes or insights relevant to the question. Use bullet points and provide citations in this format: (Document ID: [ID], Page: [X], Paragraph: [Y], Source: [Source]).
Context:
{context}

Question:
{question}

Summary:
"""

# ---- Functions that produce answers using Gemini directly (no load_qa_chain) ----
def build_context_from_docs(docs):
    parts = []
    for d in docs:
        md = d.metadata
        header = f"Document ID: {md.get('doc_id','NA')}, Page: {md.get('page','NA')}, Para: {md.get('paragraph','NA')}, Source: {md.get('source','NA')}\n"
        parts.append(header + d.page_content)
    # join with separator, keep size reasonable
    return "\n\n---\n\n".join(parts)

def generate_answer_from_docs(docs, question, model_name="gemini-1.5-pro", temperature=0.2):
    if not docs:
        return "Answer is not available in the context."
    context = build_context_from_docs(docs)
    prompt_text = QA_PROMPT_TEMPLATE.format(context=context, question=question)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    try:
        answer = call_gemini_model(model, prompt_text)
    except Exception as e:
        # bubble up a helpful error but keep app running
        answer = f"LLM error: {e}\n\nTrace:\n{traceback.format_exc()}"
    return answer

def generate_theme_summary(docs, question, model_name="gemini-1.5-pro", temperature=0.5):
    if not docs:
        return "No themes available."
    context = build_context_from_docs(docs)
    prompt_text = THEME_PROMPT_TEMPLATE.format(context=context, question=question)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    try:
        answer = call_gemini_model(model, prompt_text)
    except Exception as e:
        answer = f"LLM error: {e}\n\nTrace:\n{traceback.format_exc()}"
    return answer

# ---------------- Main UI: ask question ----------------
st.header("💬 Ask a Question")
with st.form(key="question_form"):
    user_question = st.text_area("Enter your question here (Press Ctrl + Enter to submit):", placeholder="E.g., What are the key findings in the uploaded documents?", height=120)
    submit_button = st.form_submit_button(label="Submit Question")

if submit_button:
    if not api_key:
        st.warning("Please provide your Google Gemini API key in the sidebar.")
    elif not (uploaded_files or url_input):
        st.warning("Please upload at least one document or provide a URL.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing documents and generating answer..."):
            try:
                all_chunks = []
                all_metadata = []

                # process uploaded files
                if uploaded_files:
                    for f in uploaded_files:
                        text, meta = extract_text_from_file(f, f.name)
                        if text:
                            chunks, chunk_md = get_text_chunks(text, meta)
                            all_chunks.extend(chunks)
                            all_metadata.extend(chunk_md)

                # process URL
                if url_input:
                    t, m = extract_text_from_url(url_input)
                    if t:
                        c, cm = get_text_chunks(t, m)
                        all_chunks.extend(c)
                        all_metadata.extend(cm)

                if not all_chunks:
                    st.warning("No text found in uploaded files or URL.")
                else:
                    # build or load vector store
                    vector_store = get_vector_store(all_chunks, all_metadata)
                    # create embeddings (used for loading local index as well)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    try:
                        # attempt to load saved index if exists (this call shape may vary by package version)
                        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    except Exception:
                        new_db = vector_store

                    # similarity search
                    docs = new_db.similarity_search(user_question, k=5)

                    # Show which docs were retrieved
                    st.subheader("Retrieved snippets")
                    citation_data = []
                    for d in docs:
                        md = d.metadata
                        snippet = (d.page_content[:300] + "...") if len(d.page_content) > 300 else d.page_content
                        citation_data.append({
                            "Document ID": md.get("doc_id"),
                            "Source": md.get("source"),
                            "Page": md.get("page"),
                            "Paragraph": md.get("paragraph"),
                            "Snippet": snippet,
                        })
                    st.table(pd.DataFrame(citation_data))

                    # Generate answer from docs using Gemini
                    st.subheader("Answer")
                    answer_text = generate_answer_from_docs(docs, user_question)
                    st.markdown(answer_text)

                    # Generate themes
                    st.subheader("Recurring Themes")
                    themes = generate_theme_summary(docs, user_question)
                    st.markdown(themes)

            except Exception as e:
                st.error(f"An error occurred while processing: {e}")
                st.exception(traceback.format_exc())

# Footer
st.markdown(
    """
    <div class="text-center text-gray-500 mt-8">
        <hr class="border-gray-200 mb-4">
        <p>Powered by LangChain, Google Gemini, and FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)

