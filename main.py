from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid
import os
import requests

# ------------------ APP ------------------
app = FastAPI(title="RAG Server (API Embeddings)")

# ------------------ ENV ------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    raise RuntimeError("Missing environment variables")

# ------------------ QDRANT ------------------
COLLECTION = "rag_docs"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# Create collection once
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=1536,   # embedding size (IMPORTANT)
            distance=Distance.COSINE
        )
    )

# ------------------ EMBEDDING VIA API ------------------
def embed_text(text: str) -> list:
    response = requests.post(
        "https://api.groq.com/openai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        },
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(500, "Embedding API error")

    return response.json()["data"][0]["embedding"]

# ------------------ SCHEMA ------------------
class Question(BaseModel):
    question: str

# ------------------ ROUTES ------------------
@app.get("/")
def health():
    return {"status": "RAG server running"}

# ------------------ INGEST PDF ------------------
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        raise HTTPException(400, "Empty PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    points = []
    for chunk in chunks:
        vector = embed_text(chunk)
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": chunk,
                "source": file.filename
            }
        })

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    return {
        "status": "PDF ingested",
        "chunks": len(points)
    }

# ------------------ QUERY ------------------
@app.post("/query")
def query_rag(q: Question):
    query_vector = embed_text(q.question)

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=5
    )

    if not results:
        return {
            "answer": "Aucun document trouvé. Uploadez un PDF.",
            "sources": []
        }

    context = "\n".join(hit.payload["text"] for hit in results)

    prompt = f"""
Réponds uniquement avec le contexte suivant.

CONTEXTE:
{context}

QUESTION:
{q.question}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(500, "LLM error")

    answer = response.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": list({hit.payload["source"] for hit in results})
    }
