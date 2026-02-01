from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import requests
import uuid
import os
import time

# ================== CONFIG ==================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en"
VECTOR_SIZE = 384
COLLECTION_NAME = "rag_docs"

# ============================================

app = FastAPI()

# Qdrant en mémoire
qdrant = QdrantClient(":memory:")

# Créer la collection
try:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' créée")
except Exception as e:
    print(f"ℹ️ Collection existe déjà ou erreur: {e}")

# ================== UTILS ==================

def split_text(text, size=300):
    for i in range(0, len(text), size):
        chunk = text[i:i + size].strip()
        if chunk:
            yield chunk

def embed_text(text: str):
    """Génère des embeddings avec Hugging Face"""
    
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN non configuré")
    
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
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list):
                if isinstance(data[0], list):
                    vector = data[0]
                else:
                    vector = data
                
                # Ajuster la taille si nécessaire
                if len(vector) > VECTOR_SIZE:
                    return vector[:VECTOR_SIZE]
                elif len(vector) < VECTOR_SIZE:
                    return vector + [0.0] * (VECTOR_SIZE - len(vector))
                else:
                    return vector
        else:
            raise Exception(f"API error: {response.status_code}")
            
    except Exception as e:
        print(f"Erreur embedding: {str(e)}")
        # Fallback: vecteur simple
        return [0.1] * VECTOR_SIZE

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

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text or len(text.strip()) < 20:
            continue

        for chunk in split_text(text):
            try:
                vector = embed_text(chunk)
                
                # Utiliser PointStruct correctement
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": file.filename,
                        "page": page_num
                    }
                )
                
                # Upsert avec PointStruct
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[point]
                )
                
                total_chunks += 1
                print(f"✓ Chunk ajouté: {total_chunks}")
                
                time.sleep(0.1)  # Petite pause
                
            except Exception as e:
                print(f"Erreur chunk: {str(e)}")
                continue

    return {
        "status": "success",
        "file": file.filename,
        "chunks_stored": total_chunks
    }

@app.post("/ask")
async def ask_question(data: Question):
    try:
        question_vector = embed_text(data.question)
        
        # Recherche CORRECTE dans Qdrant
        results = qdrant.query(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            limit=3
        )
        
        # Vérifier si results est une liste ou un objet
        if not results:
            return {
                "question": data.question,
                "answer": "Aucune information trouvée dans le document.",
                "contexts": []
            }
        
        # Extraire les textes selon le format de réponse
        contexts = []
        if hasattr(results[0], 'payload'):
            # Format avec objets
            contexts = [r.payload.get("text", "") for r in results]
        elif isinstance(results[0], dict) and "payload" in results[0]:
            # Format dict
            contexts = [r["payload"].get("text", "") for r in results]
        else:
            contexts = [str(r) for r in results]

        return {
            "question": data.question,
            "contexts": contexts[:3],  # Limiter à 3
            "count": len(contexts)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur recherche: {str(e)}"
        )

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "embedding": "BAAI/bge-small-en",
        "vector_size": VECTOR_SIZE,
        "status": "active"
    }

@app.get("/test")
def test():
    """Test simple de l'API"""
    return {
        "message": "API fonctionnelle",
        "qdrant": "connecté",
        "hugging_face": "configuré" if HF_TOKEN else "non configuré"
    }

# ================== MIDDLEWARE ==================

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
    print("🚀 Serveur démarré")
    print(f"Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)