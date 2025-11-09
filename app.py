# -*- coding: utf-8 -*-
"""
Enterprise PDF to Searchable Knowledge Base Application

This application processes enterprise PDFs (including scanned ones via OCR) to create a searchable knowledge base.
It extracts text, tables, and images while preserving document structure (e.g., chunking by titles/sections to avoid
cutting across chapters). Text is chunked into meaningful sections and stored in a vector database (Chroma) for semantic
search. Tables are extracted and stored in a simple NoSQL-like structure (JSON file for persistence, simulating a
document store like MongoDB) for precise queries. Images/charts are described using a multimodal model for searchability.

Key Features:
- PDF Partitioning: Uses Unstructured.io with 'hi_res' strategy for OCR (requires tesseract/poppler installed).
- Chunking: 'by_title' strategy to respect sections/chapters.
- Summarization: Groq (Llama-3.1) for text/tables; Gemini for images.
- Storage: Chroma (vector DB) for text summaries; JSON file (NoSQL sim) for raw tables.
- Retrieval: MultiVectorRetriever for text/images; separate lookup for tables.
- Search Interface: Simple Gradio web UI for querying.
- Strictly uses Groq and Gemini APIs (no OpenAI).

Setup Instructions:
1. Install dependencies: pip install -Uq "unstructured[all-docs]" pillow lxml chromadb tiktoken langchain langchain-community langchain-groq langchain-google-genai python_dotenv gradio
2. Install system deps: For Linux: apt-get install poppler-utils tesseract-ocr libmagic-dev; For Mac: brew install poppler tesseract libmagic
3. Set env vars: OPENAI_API_KEY -> no, use GROQ_API_KEY and GOOGLE_API_KEY.
4. Run: python this_script.py

Documentation:
- Input: PDF file path.
- Output: Chroma DB (./chroma_db), tables.json (NoSQL store).
- Performance: See benchmarks in __main__.
- Stretch: Basic multi-lang support via Gemini; low-quality scans via OCR; chart extraction via image desc.

References:
- Inspired by: https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb
- MultiVector: https://python.langchain.com/docs/how_to/multi_vector/
"""

import os
import json
import uuid
import time
import base64
from pathlib import Path

import gradio as gr
from IPython.display import Image, display  # For Jupyter fallback, optional
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Env setup (replace with your keys)
os.environ["GROQ_API_KEY"] = "gsk_..."  # Your Groq key
os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key_here"  # Your Gemini key

# Paths
OUTPUT_PATH = "./content/"
CHROMA_PATH = "./chroma_db"  # Persistent Chroma dir
TABLES_JSON = "./tables.json"  # NoSQL sim for tables

def get_images_base64(chunks):
    """Extract base64 images from CompositeElements."""
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def display_base64_image(base64_code):
    """Display base64 image (for debugging)."""
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

def summarize_text_tables_groq(elements):
    """Summarize text/tables using Groq Llama-3.1."""
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain.batch(elements, {"max_concurrency": 3})

def summarize_images_gemini(images):
    """Summarize images using Gemini (multimodal)."""
    prompt_template = """Describe the image in detail. For context,
                          the image is part of an enterprise document like reports or manuals.
                          Be specific about charts, graphs, tables, or visuals for searchability.
                          Support multiple languages if text is present."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chain = prompt | model | StrOutputParser()
    return chain.batch(images)

def parse_docs(docs):
    """Split docs into texts (incl. tables) and images."""
    b64 = []
    text = []
    for doc in docs:
        if isinstance(doc, str):  # Images are str base64
            try:
                base64.b64decode(doc)
                b64.append(doc)
            except:
                text.append(doc)
        else:  # CompositeElement or Table objects
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    """Build multimodal prompt for Gemini."""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += getattr(text_element, 'text', str(text_element))  # .text for elements

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images/charts.
    Use descriptions for visuals. Support precise table queries if applicable.
    Context: {context_text}
    Question: {user_question}
    """
    prompt_content = [{"type": "text", "text": prompt_template}]
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            )
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

def process_pdf(file_path, clear_db=False):
    """
    Main pipeline: Process PDF -> Extract -> Summarize -> Store.
    Returns retriever and table_store.
    """
    if clear_db:
        Path(CHROMA_PATH).mkdir(exist_ok=True)
        # Clear Chroma if exists
        try:
            Chroma(persist_directory=CHROMA_PATH, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")).delete_collection()
        except:
            pass
        with open(TABLES_JSON, 'w') as f:
            json.dump({}, f)

    # Partition PDF (handles OCR, tables, images; chunk by title for sections)
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",  # For OCR/layout
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",  # Respects chapters/sections
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    # Separate elements
    tables = [chunk for chunk in chunks if "Table" in str(type(chunk))]
    texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
    images = get_images_base64(chunks)

    # Summarize
    start_time = time.time()
    text_summaries = summarize_text_tables_groq([t.text for t in texts])
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_text_tables_groq(tables_html)
    image_summaries = summarize_images_gemini(images)
    process_time = time.time() - start_time

    # Vectorstore setup (text summaries)
    embedding_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        collection_name="enterprise_rag",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_func
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    # Load texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Load tables to vector (summaries) + NoSQL (raw)
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Store raw tables in NoSQL (JSON sim)
    table_store = {}
    for i, table in enumerate(tables):
        table_data = {
            "id": table_ids[i],
            "summary": table_summaries[i],
            "html": tables_html[i],
            "markdown": table.text,  # For precise queries
            "page": getattr(table.metadata, 'page_number', 'N/A')
        }
        table_store[table_ids[i]] = table_data
    with open(TABLES_JSON, 'w') as f:
        json.dump(table_store, f, indent=2)

    # Load images
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))

    # Persist Chroma
    vectorstore.persist()

    print(f"Processed in {process_time:.2f}s. Extracted: {len(texts)} texts, {len(tables)} tables, {len(images)} images.")

    # RAG Chain (Gemini multimodal)
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        | StrOutputParser()
    )

    return chain, table_store

def query_knowledge_base(query, chain, table_store):
    """Query the RAG + check for table-specific."""
    start_time = time.time()
    response = chain.invoke(query)
    query_time = time.time() - start_time

    # For precise table queries, check if response mentions tables
    if "table" in response.lower() or any(kw in query.lower() for kw in ["table", "data", "figure"]):
        # Simple lookup: return relevant table if summary matches (in prod, use vector search on tables)
        relevant_tables = [t for t in table_store.values() if any(kw in t["summary"].lower() for kw in query.lower().split())]
        if relevant_tables:
            response += f"\n\nRelevant Table Data:\n{json.dumps(relevant_tables[0], indent=2)}"

    print(f"Query time: {query_time:.2f}s")
    return response

# Performance Benchmarks
def run_benchmarks(file_path):
    """Run timing benchmarks."""
    print("Benchmarking...")
    chain, _ = process_pdf(file_path)
    queries = ["What is the main topic?", "Summarize the tables.", "Describe any charts."]
    times = []
    for q in queries:
        start = time.time()
        query_knowledge_base(q, chain, {})
        times.append(time.time() - start)
    avg_query_time = sum(times) / len(times)
    print(f"Avg Query Time: {avg_query_time:.2f}s | {len(queries)} queries")

if __name__ == "__main__":
    # Example usage
    file_path = OUTPUT_PATH + "enterprise_report.pdf"  # Replace with your PDF
    run_benchmarks(file_path)  # Optional benchmark

    # Launch Gradio Search Interface
    def gradio_query(query):
        if not hasattr(gradio_query, 'chain'):
            gradio_query.chain, gradio_query.table_store = process_pdf(file_path)
        return query_knowledge_base(query, gradio_query.chain, gradio_query.table_store)

    iface = gr.Interface(
        fn=gradio_query,
        inputs=gr.Textbox(label="Enter your search query:"),
        outputs=gr.Textbox(label="Response:"),
        title="Enterprise PDF Knowledge Base Search",
        description="Query your processed PDFs semantically, including tables and images."
    )
    iface.launch(share=True)
