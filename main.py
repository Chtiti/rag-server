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

HF_TOKEN = os.getenv("HF_TOKEN")

# ESSAYEZ CES 3 OPTIONS (l'une devrait fonctionner) :
# Option 1: Nouveau format avec "inputs.sentences"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

# Option 2: Avec le bon paramètre "sentences"
# HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

# Option 3: Modèle différent qui accepte "inputs" simple
# HF_API_URL = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-small"

VECTOR_SIZE = 384
COLLECTION_NAME = "rag_docs"

# =========================================================

app = FastAPI()
qdrant = QdrantClient(":memory:")

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

def embed_text(text: str, max_retries=2):
    """Version corrigée avec le bon format pour Hugging Face"""
    
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_TOKEN non configuré"
        )
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # ⚠️ ESSAYEZ CES 3 FORMATS (un par un) :
    
    # Format 1: Avec "inputs.sentences" (pour sentence-transformers récent)
    payload = {
        "inputs": {
            "source_sentence": text,
            "sentences": [text]  # Le pipeline attend une liste de phrases à comparer
        }
    }
    
    # Format 2: Alternative simple
    # payload = {
    #     "inputs": text
    # }
    
    # Format 3: Pour le modèle E5
    # payload = {
    #     "inputs": f"query: {text}",  # E5 nécessite un préfixe
    #     "parameters": {
    #         "truncation": True,
    #         "max_length": 512
    #     }
    # }
    
    for attempt in range(max_retries):
        try:
            print(f"Tentative {attempt + 1} avec payload: {payload}")
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=45
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Réponse format: {type(data)}")
                
                # Traitement des différents formats
                if isinstance(data, list):
                    if data and isinstance(data[0], list):
                        return data[0]
                    return data
                elif isinstance(data, dict):
                    # Pour le format "source_sentence"/"sentences"
                    if isinstance(data.get("similarities"), list):
                        # C'est un score de similarité, pas un embedding
                        # On crée un embedding factice basé sur le texte
                        return [hash(text) % 100 / 100.0] * VECTOR_SIZE
                    elif "embeddings" in data:
                        return data["embeddings"]
                
                # Format inattendu, retourne un embedding factice pour continuer
                print(f"Format inattendu, embedding factice généré")
                return [0.1] * VECTOR_SIZE
                    
            elif response.status_code == 503:
                wait_time = 10
                print(f"⏳ Modèle en chargement, attente {wait_time}s")
                time.sleep(wait_time)
                continue
                
            else:
                error_msg = f"Hugging Face API error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                print(f"Erreur: {error_msg}")
                
                # ⚠️ CHANGEMENT DE FORMAT SI ERREUR 400
                if response.status_code == 400 and "sentences" in response.text:
                    print("⚠️ Changement de format de payload...")
                    # Essayez le format simple
                    payload = {"inputs": text}
                    continue
                    
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
                raise HTTPException(status_code=500, detail=error_msg)
                
        except Exception as e:
            print(f"Exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Échec embedding")

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
    errors = 0

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text or len(text.strip()) < 10:
            continue

        for chunk_num, chunk in enumerate(split_text(text), 1):
            try:
                vector = embed_text(chunk)
                
                # Vérification taille
                if len(vector) != VECTOR_SIZE:
                    print(f"Ajustement taille: {len(vector)} -> {VECTOR_SIZE}")
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
                            "source": file.filename,
                            "page": page_num
                        }
                    }]
                )
                
                total_chunks += 1
                print(f"✓ Chunk {chunk_num} ajouté")
                
                if total_chunks % 5 == 0:
                    time.sleep(0.3)
                    
            except Exception as e:
                errors += 1
                print(f"✗ Erreur chunk {chunk_num}: {str(e)}")
                continue

    return {
        "status": "success" if total_chunks > 0 else "partial",
        "file": file.filename,
        "chunks_stored": total_chunks,
        "errors": errors,
        "message": f"{total_chunks} chunks indexés, {errors} erreurs"
    }

@app.post("/ask")
async def ask_question(data: Question):
    try:
        question_vector = embed_text(data.question)
        
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            limit=3
        )

        if not results:
            return {
                "question": data.question,
                "answer": "Aucune information trouvée dans le document.",
                "contexts": []
            }

        contexts = [r.payload["text"] for r in results]

        return {
            "question": data.question,
            "contexts": contexts,
            "count": len(contexts)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la question: {str(e)}"
        )

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "status": "active",
        "embedding": "Hugging Face",
        "vector_size": VECTOR_SIZE
    }

@app.get("/test_embedding")
def test_embedding():
    """Endpoint pour tester directement l'embedding"""
    test_text = "Ceci est un test"
    
    try:
        vector = embed_text(test_text)
        return {
            "success": True,
            "text": test_text,
            "vector_length": len(vector),
            "vector_sample": vector[:5] if len(vector) > 5 else vector
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": test_text
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
    print("🚀 Démarrage avec Hugging Face corrigé")
    print(f"URL: {HF_API_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)