import os
import chainlit as cl
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import shutil

shutil.rmtree("./chroma_db", ignore_errors=True)

# === Models ===
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")  # Better embedding model
LANGUAGE_MODEL = OllamaLLM(model="mistral", streaming=True)  # LLM for answering

# Vector DB
VECTOR_DB: Chroma = None
CHROMA_DIR = "./chroma_db"

# === Prompt Template ===
PROMPT_TEMPLATE = """
You are a helpful research assistant. Use only the information provided in the context below to answer the user's query. 
Do not use prior knowledge. If the context does not contain the answer, respond with "I don't know based on the provided context."

Be accurate, factual, and concise (max 3 sentences). Reference page numbers if available.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# === Helpers ===
def clean_text(text: str) -> str:
    return ' '.join(text.split())  # Remove newlines, tabs, extra spaces

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)

    for i, doc in enumerate(chunks):
        page = doc.metadata.get("page", i + 1)
        doc.page_content = f"[Page {page}]\n{clean_text(doc.page_content)}"
        doc.metadata["source"] = f"Page {page}"

    return chunks

def index_documents(chunks):
    global VECTOR_DB
    VECTOR_DB = Chroma.from_documents(chunks, EMBEDDING_MODEL, persist_directory=CHROMA_DIR)

def find_related_documents(query, threshold=0.75):
    if VECTOR_DB is None:
        return []
    results = VECTOR_DB.similarity_search_with_score(query, k=5)
    return [doc for doc, score in results if score >= threshold]

async def generate_answer_stream(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream({"user_query": user_query, "document_context": context_text}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        await msg.stream_token(token)

    await msg.update()

# === File Upload Handler ===
async def handle_file_upload(files):
    file = files[0]
    ext = os.path.splitext(file.name)[1].lower()

    if ext != ".pdf":
        await cl.Message(content="❌ Unsupported file type. Please upload a PDF.").send()
        return

    file_path = file.path
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    chunks = chunk_documents(docs)
    index_documents(chunks)

    # Optional quick summary
    preview = "\n\n".join([c.page_content for c in chunks[:3]])
    summary_prompt = "Summarize the following content in 3 bullet points:\n\n" + preview
    summary = LANGUAGE_MODEL.invoke(summary_prompt)

    await cl.Message(content=f"📌 Summary:\n{summary}").send()
    await cl.Message(content="✅ PDF processed and indexed! Now ask a question.").send()

# === Chainlit Events ===
@cl.on_chat_start
async def on_chat_start():
    if os.path.exists(CHROMA_DIR):
        global VECTOR_DB
        VECTOR_DB = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDING_MODEL)

    await cl.Message(content="📄 Upload a PDF to get started.").send()
    files = await cl.AskFileMessage(
        content="📎 Please upload your PDF file",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=1
    ).send()

    if files:
        await handle_file_upload(files)

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content=f"❓ Question: {message.content}", author="User").send()

    docs = find_related_documents(message.content)
    if not docs:
        await cl.Message(content="⚠️ No relevant content found in the document.").send()
        return

    await generate_answer_stream(message.content, docs)
