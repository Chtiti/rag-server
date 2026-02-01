from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os
import time

# ================== CONFIG HUGGING FACE ==================

# Récupérer depuis les Environment Variables de Render
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ HF_TOKEN non configuré dans Render Environment Variables")

# Configuration Hugging Face (comme dans votre ancien code)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 1536  # Taille pour all-MiniLM-L6-v2
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

COLLECTION_NAME = "rag_docs"

# =========================================================

app = FastAPI()

# Qdrant en mémoire (comme avant)
qdrant = QdrantClient(":memory:")

# créer collection (comme avant)
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# ================== FONCTION EMBEDDING HUGGING FACE ==================

def embed_text(text: str):
    """Fonction embedding avec Hugging Face API (remplace Groq)"""
    
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_TOKEN non configuré. Ajoutez-le dans Render Environment Variables"
        )
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": text,
        "parameters": {
            "truncation": True,
            "max_length": 512
        }
    }
    
    # Essayer plusieurs fois (Hugging Face peut être lent)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60  # Hugging Face peut être lent
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Hugging Face retourne différents formats
                if isinstance(data, list):
                    if isinstance(data[0], list):
                        return data[0]  # Format [[...]]
                    return data  # Format [...]
                else:
                    return data
                    
            elif response.status_code == 503:
                # Modèle en cours de chargement
                wait_time = 10 * (attempt + 1)
                print(f"⏳ Modèle en chargement... attente {wait_time}s")
                time.sleep(wait_time)
                continue
                
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Token Hugging Face invalide"
                )
                
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur Hugging Face: {response.status_code} - {response.text}"
                )
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"⏱️ Timeout, nouvelle tentative {attempt + 2}/{max_retries}")
                continue
            raise HTTPException(
                status_code=504,
                detail="Timeout Hugging Face API"
            )
    
    raise HTTPException(
        status_code=500,
        detail=f"Échec après {max_retries} tentatives"
    )

# ================== AUTRES FONCTIONS (IDENTIQUES À AVANT) ==================

def split_text(text, size=200):
    """Découpe le texte en chunks (identique à avant)"""
    for i in range(0, len(text), size):
        yield text[i:i + size]

# ================== MODELS (IDENTIQUES À AVANT) ==================

class Question(BaseModel):
    question: str

# ================== ENDPOINTS (IDENTIQUES À AVANT) ==================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Identique à votre ancien code, juste l'embedding change"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF requis")

    reader = PdfReader(file.file)
    total_chunks = 0

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for chunk in split_text(text):
            vector = embed_text(chunk)  # Utilise Hugging Face maintenant

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
    """Identique à votre ancien code, juste l'embedding change"""
    question_vector = embed_text(data.question)  # Utilise Hugging Face maintenant

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

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "embedding": "Hugging Face",
        "model": MODEL_NAME,
        "vector_size": VECTOR_SIZE,
        "collection": COLLECTION_NAME
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant": "active",
        "embedding": "Hugging Face"
    }

# ================== MIDDLEWARE CORS ==================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DÉMARRAGE ==================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Serveur démarré sur le port {port}")
    print(f"📦 Modèle Hugging Face: {MODEL_NAME}")
    print(f"🔢 Taille vecteurs: {VECTOR_SIZE}")
    print(f"🗄️ Collection Qdrant: {COLLECTION_NAME}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)