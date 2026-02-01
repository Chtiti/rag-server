from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os
import time

# ================== CONFIG HUGGING FACE CORRIGÉ ==================

# Variable d'environnement Render
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ À configurer dans Render

# NOUVEAU ENDPOINT (Hugging Face a changé l'URL)
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

# OU alternative gratuite (Text Embeddings Inference)
# HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384  # Taille fixe pour le modèle Hugging Face
COLLECTION_NAME = "rag_docs"

# =========================================================

app = FastAPI()

# Qdrant en mémoire (inchangé)
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

def embed_text(text: str, max_retries=3):
    """Fonction embedding avec nouveau endpoint Hugging Face"""
    
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_TOKEN non configuré. Ajoutez-le dans Render Environment Variables"
        )
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # NOUVEAU FORMAT pour le nouvel endpoint
    payload = {
        "inputs": text,
        "parameters": {
            "truncation": True,
            "max_length": 512
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60  # Timeout augmenté
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Traitement des différents formats de réponse
                if isinstance(data, list):
                    if isinstance(data[0], list):
                        return data[0]  # Format [[...]]
                    return data  # Format [...]
                elif isinstance(data, dict):
                    if "embeddings" in data:
                        return data["embeddings"]
                    elif "outputs" in data:
                        return data["outputs"]
                
                # Si on arrive ici, format inattendu
                return data
                    
            elif response.status_code == 503:
                # Modèle en cours de chargement
                wait_time = 15 * (attempt + 1)
                print(f"⏳ Modèle en chargement... attente {wait_time}s")
                time.sleep(wait_time)
                continue
                
            elif response.status_code in [401, 403]:
                raise HTTPException(
                    status_code=401,
                    detail=f"Token Hugging Face invalide ou expiré. Code: {response.status_code}"
                )
                
            elif response.status_code == 429:
                # Rate limiting
                wait_time = 30
                print(f"⚠️ Rate limit, attente {wait_time}s")
                time.sleep(wait_time)
                continue
                
            else:
                error_msg = f"Hugging Face API error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:150]}"
                raise HTTPException(status_code=500, detail=error_msg)
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"⏱️ Timeout, nouvelle tentative {attempt + 2}/{max_retries}")
                time.sleep(10)
                continue
            raise HTTPException(
                status_code=504,
                detail="Timeout Hugging Face API après plusieurs tentatives"
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Erreur, nouvelle tentative: {str(e)}")
                time.sleep(5)
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Erreur Hugging Face: {str(e)}"
            )
    
    raise HTTPException(
        status_code=500,
        detail=f"Échec après {max_retries} tentatives"
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
            try:
                vector = embed_text(chunk)
                
                # Vérifier la taille du vecteur
                if len(vector) != VECTOR_SIZE:
                    print(f"⚠️ Vecteur taille {len(vector)}, ajustement à {VECTOR_SIZE}")
                    if len(vector) > VECTOR_SIZE:
                        vector = vector[:VECTOR_SIZE]
                    else:
                        vector = vector + [0.0] * (VECTOR_SIZE - len(vector))
                
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
                
                # Petite pause pour éviter le rate limiting
                if total_chunks % 10 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Erreur sur chunk: {str(e)}")
                continue

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

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "embedding": "Hugging Face (nouvel endpoint)",
        "endpoint": HF_API_URL,
        "vector_size": VECTOR_SIZE
    }

@app.get("/health")
def health():
    # Test simple de l'API Hugging Face
    test_payload = {"inputs": "test", "parameters": {"truncation": True}}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    try:
        test_resp = requests.post(HF_API_URL, headers=headers, json=test_payload, timeout=10)
        hf_status = test_resp.status_code == 200 or test_resp.status_code == 503
    except:
        hf_status = False
    
    return {
        "status": "ok" if hf_status else "degraded",
        "hugging_face": hf_status,
        "qdrant": True,
        "token_configured": bool(HF_TOKEN)
    }

# ================== MIDDLEWARE CORS ==================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Serveur démarré avec nouveau endpoint Hugging Face")
    print(f"🔗 Endpoint: {HF_API_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)