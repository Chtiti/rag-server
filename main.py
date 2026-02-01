from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os

# ================== CONFIG ==================

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_URL = "https://api.groq.com/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_SIZE = 1536
COLLECTION_NAME = "rag_docs"

# ============================================

app = FastAPI()

# Qdrant en mémoire (low RAM, parfait pour Render free)
qdrant = QdrantClient(":memory:")

# créer collection
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# ================== UTILS ==================

def split_text(text, size=200):
    for i in range(0, len(text), size):
        yield text[i:i + size]

def embed_text(text: str):
    response = requests.post(
        EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erreur embeddings API")

    return response.json()["data"][0]["embedding"]

# ================== MODELS ==================

class Question(BaseModel):
    question: str

# ================== ENDPOINTS ==================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF requis")

    reader = PdfReader(file.file)
    total_chunks = 0

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for chunk in split_text(text):
            vector = embed_text(chunk)

            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "text": chunk,
                        "source": file.filename
                    }
                }]
            )

            total_chunks += 1

    return {
        "status": "success",
        "file": file.filename,
        "chunks_stored": total_chunks
    }

@app.post("/ask")
async def ask_question(data: Question):
    question_vector = embed_text(data.question)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vector,
        limit=3
    )

    if not results:
        return {
            "answer": "Aucune information trouvée dans le document."
        }

    contexts = [r.payload["text"] for r in results]

    return {
        "question": data.question,
        "contexts": contexts
    }
