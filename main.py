from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os

# ================= CONFIG =================

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")  # clé API embeddings
EMBEDDING_URL = "https://api.groq.com/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "rag_docs"
VECTOR_SIZE = 1536  # taille embedding

# ==========================================

app = FastAPI()

qdrant = QdrantClient(":memory:")  # léger RAM (OK pour test)

# créer collection si inexistante
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# -------- fonctions utilitaires --------

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

# --------------- API --------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Fichier PDF requis")

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
