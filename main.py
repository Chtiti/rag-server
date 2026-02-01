from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os

# ================== CONFIG HUGGING FACE ==================

# Variable d'environnement Render (au lieu de EMBEDDING_API_KEY)
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ À configurer dans Render
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384  # Taille fixe pour le modèle Hugging Face
COLLECTION_NAME = "rag_docs"

# =========================================================

app = FastAPI()

# Qdrant en mémoire (inchangé)
qdrant = QdrantClient(":memory:")

# créer collection (taille vecteur modifiée)
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,  # Maintenant 384 au lieu de 1536
        distance=Distance.COSINE
    )
)

# ================== UTILS ==================

def split_text(text, size=200):
    for i in range(0, len(text), size):
        yield text[i:i + size]

def embed_text(text: str):
    # Appel à Hugging Face au lieu de Groq
    response = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",  # Format Hugging Face
            "Content-Type": "application/json"
        },
        json={
            "inputs": text,  # Hugging Face utilise "inputs" (avec 's')
            "parameters": {
                "truncation": True  # Important pour éviter les erreurs
            }
        },
        timeout=30
    )

    if response.status_code != 200:
        # Message d'erreur plus précis
        error_detail = f"Hugging Face API error: {response.status_code}"
        try:
            error_detail += f" - {response.text[:100]}"
        except:
            pass
        raise HTTPException(status_code=500, detail=error_detail)

    # Format de réponse Hugging Face (différent de Groq/OpenAI)
    data = response.json()
    
    # Hugging Face peut retourner différents formats
    if isinstance(data, list):
        if isinstance(data[0], list):
            return data[0]  # Format [[embedding1, embedding2, ...]]
        return data  # Format [embedding1, embedding2, ...]
    elif isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    else:
        # Format inattendu
        raise HTTPException(
            status_code=500, 
            detail=f"Format de réponse Hugging Face inattendu: {type(data)}"
        )

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