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
- Multi-Language Support: OCR with configurable languages (e.g., eng+fra); Models handle multi-lang natively.
- Strictly uses Groq and Gemini APIs (no OpenAI).

Setup Instructions:
1. Install dependencies: pip install -Uq "unstructured[all-docs]" pillow lxml chromadb tiktoken langchain langchain-community langchain-groq langchain-google-genai python_dotenv gradio
2. Install system deps: For Linux: apt-get install poppler-utils tesseract-ocr libmagic-dev; For Mac: brew install poppler tesseract libmagic
   - For multi-lang OCR: Install tesseract lang packs, e.g., apt-get install tesseract-ocr-fra
3. Set env vars: GROQ_API_KEY and GOOGLE_API_KEY.
4. Run: python this_script.py

Documentation:
- Input: PDF file path + optional languages (e.g., ['eng', 'fra']).
- Output: Chroma DB (./chroma_db), tables.json (NoSQL store).
- Performance: See benchmarks in __main__.
- Stretch: Multi-lang via Gemini/tesseract; low-quality scans via OCR; chart extraction via image desc.

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

def summarize_text_tables_groq(elements, languages=['en']):
    """Summarize text/tables using Groq Llama-3.1, with multi-lang awareness."""
    lang_hint = f" (Document language(s): {', '.join(languages)})" if len(languages) > 1 else ""
    prompt_text = f"""
    You are an assistant tasked with summarizing tables and text.
    Provide the summary in English, but preserve key terms in original language if necessary{lang_hint}.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {{element}}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain.batch(elements, {"max_concurrency": 3})

def summarize_images_gemini(images, languages=['en']):
    """Summarize images using Gemini (multimodal), with multi-lang support."""
    lang_hint = f"Document language(s): {', '.join(languages)}. Translate descriptions to English if needed, but note original lang text."
    prompt_template = f"""Describe the image in detail. For context,
                          the image is part of an enterprise document like reports or manuals.{lang_hint}
                          Be specific about charts, graphs, tables, or visuals for searchability.
                          If text is in the image, transcribe and translate key parts to English."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{{image}}"},
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

def build_prompt(kwargs, languages=['en']):
    """Build multimodal prompt for Gemini, multi-lang aware."""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += getattr(text_element, 'text', str(text_element))  # .text for elements

    lang_hint = f"Document language(s): {', '.join(languages)}. Answer in English."
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images/charts.
    Use descriptions for visuals. Support precise table queries if applicable.{lang_hint}
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

def process_pdf(file_path, ocr_languages=['eng'], clear_db=False):
    """
    Main pipeline: Process PDF -> Extract -> Summarize -> Store.
    Supports multi-lang OCR via ocr_languages (e.g., ['eng', 'fra']).
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

    # Partition PDF (handles OCR with multi-lang, tables, images; chunk by title for sections)
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",  # For OCR/layout
        languages=ocr_languages,  # Multi-lang OCR (tesseract langs)
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
    text_summaries = summarize_text_tables_groq([t.text for t in texts], ocr_languages)
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_text_tables_groq(tables_html, ocr_languages)
    image_summaries = summarize_images_gemini(images, ocr_languages)
    process_time = time.time() - start_time

    # Vectorstore setup (text summaries; Google embeddings multi-lang)
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
            "page": getattr(table.metadata, 'page_number', 'N/A'),
            "languages": ocr_languages
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

    print(f"Processed in {process_time:.2f}s with languages {ocr_languages}. Extracted: {len(texts)} texts, {len(tables)} tables, {len(images)} images.")

    # RAG Chain (Gemini multimodal, multi-lang)
    def chain_invoke(query):
        return (
            {
                "context": retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda kwargs: build_prompt(kwargs, ocr_languages))
            | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            | StrOutputParser()
        ).invoke(query)

    return chain_invoke, table_store

def query_knowledge_base(query, chain, table_store, languages=['eng']):
    """Query the RAG + check for table-specific."""
    start_time = time.time()
    response = chain(query)
    query_time = time.time() - start_time

    # For precise table queries, check if response mentions tables
    if "table" in response.lower() or any(kw in query.lower() for kw in ["table", "data", "figure"]):
        # Simple lookup: return relevant table if summary matches (in prod, use vector search on tables)
        relevant_tables = [t for t in table_store.values() if any(kw in t["summary"].lower() for kw in query.lower().split())]
        if relevant_tables:
            response += f"\n\nRelevant Table Data (langs: {t.get('languages', languages)}):\n{json.dumps(relevant_tables[0], indent=2)}"

    print(f"Query time: {query_time:.2f}s")
    return response

# Performance Benchmarks
def run_benchmarks(file_path, ocr_languages=['eng']):
    """Run timing benchmarks."""
    print("Benchmarking...")
    chain, _ = process_pdf(file_path, ocr_languages)
    queries = ["What is the main topic?", "Summarize the tables.", "Describe any charts."]
    times = []
    for q in queries:
        start = time.time()
        query_knowledge_base(q, chain, {}, ocr_languages)
        times.append(time.time() - start)
    avg_query_time = sum(times) / len(times)
    print(f"Avg Query Time: {avg_query_time:.2f}s | {len(queries)} queries")

if __name__ == "__main__":
    # Example usage
    file_path = OUTPUT_PATH + "enterprise_report.pdf"  # Replace with your PDF
    run_benchmarks(file_path, ['eng', 'fra'])  # Optional benchmark with multi-lang

    # Launch Gradio Search Interface
    def gradio_process_and_query(file, languages_input):
        if file is None:
            return "Please upload a PDF."
        file_path = file.name
        ocr_langs = languages_input.split(',') if languages_input else ['eng']
        ocr_langs = [lang.strip() for lang in ocr_langs]
        gradio_process_and_query.chain, gradio_process_and_query.table_store = process_pdf(file_path, ocr_langs)
        return f"PDF processed with languages: {ocr_langs}. Ready to query!"

    def gradio_query(query):
        if not hasattr(gradio_query, 'chain'):
            return "Please process a PDF first."
        return query_knowledge_base(query, gradio_query.chain, gradio_query.table_store, ['eng'])  # Default, but use from process

    with gr.Blocks() as iface:
        gr.Markdown("# Enterprise PDF Knowledge Base")
        with gr.Row():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            lang_input = gr.Textbox(label="OCR Languages (comma-separated, e.g., eng,fra)", placeholder="eng")
        process_btn = gr.Button("Process PDF")
        process_output = gr.Textbox(label="Processing Status")

        query_input = gr.Textbox(label="Enter your search query:")
        query_btn = gr.Button("Search")
        response_output = gr.Textbox(label="Response:")

        process_btn.click(gradio_process_and_query, inputs=[file_input, lang_input], outputs=process_output)
        query_btn.click(gradio_query, inputs=query_input, outputs=response_output)

    iface.launch(share=True)
