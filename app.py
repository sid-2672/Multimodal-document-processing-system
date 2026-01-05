"""
FastAPI server for document-based question answering using RAG.
Accepts file uploads, processes them, and answers questions using Gemini.
"""

import shutil
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ingest import ingest_file, query_milvus
import google.generativeai as genai


app = FastAPI(title="RAG API", version="2.0")


# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable. Please set it before running.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document, extracts text, chunks it, and stores in Milvus.
    Returns the collection name and number of chunks created.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        result = ingest_file(temp_path)
        return {
            "file": file.filename,
            "collection_created": result["collection"],
            "chunks_ingested": result["chunks"]
        }
    finally:
        shutil.rmtree(temp_dir)


class Query(BaseModel):
    collection_name: str
    question: str


@app.post("/query")
async def query_document(data: Query):
    """
    Takes a question and a collection name, retrieves relevant chunks from Milvus,
    and uses Gemini to generate an answer based on the retrieved context.
    """
    # Retrieve relevant chunks
    matches = query_milvus(data.collection_name, data.question)
    
    if not matches:
        return {"answer": "No relevant context found in this document."}
    
    # Build prompt with retrieved context
    context_block = "\n\n".join(matches)
    prompt = f"""You are a helpful assistant. Use only the context below to answer the user's question.

CONTEXT:
{context_block}

QUESTION:
{data.question}

INSTRUCTIONS:
- Answer based solely on the context provided.
- If the answer isn't in the context, say so clearly.
- Keep your response concise and accurate.
"""
    
    # Generate answer using Gemini
    response = gemini_model.generate_content(prompt)
    answer = response.text.strip()
    
    return {
        "answer": answer,
        "context_used": matches
    }

