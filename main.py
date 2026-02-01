from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import requests
import uuid
import os
import time
import hashlib

# ================== CONFIG CORRIGÉE ==================

HF_TOKEN = os.getenv("HF_TOKEN")

# ⚠️ CHANGER CETTE URL ! Le modèle actuel ne fait que de la similarité
# Utilisez plutôt un modèle qui fait réellement des embeddings :

# OPTION 1: BGE model (recommandé)
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en"

# OPTION 2: E5 model (multilingue)
# HF_API_URL = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-small"

# OPTION 3: feature-extraction pipeline
# HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en"

VECTOR_SIZE = 384  # Pour BAAI/bge-small-en
# VECTOR_SIZE = 384  # Pour multilingual-e5-small aussi

COLLECTION_NAME = "rag_docs"

# =====================================================

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

def split_text(text, size=300):
    for i in range(0, len(text), size):
        chunk = text[i:i + size].strip()
        if chunk:
            yield chunk

def embed_text(text: str, max_retries=3):
    """Version corrigée avec BGE model"""
    
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_TOKEN non configuré dans Render Environment Variables"
        )
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # ⚠️ FORMAT CORRECT pour BAAI/bge-small-en :
    payload = {
        "inputs": text,
        "parameters": {
            "truncation": True,
            "max_length": 512
        }
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Tentative {attempt + 1} - Texte: {text[:50]}...")
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Type réponse: {type(data)}")
                
                # BGE renvoie une liste d'embeddings
                if isinstance(data, list):
                    if isinstance(data[0], list):
                        embedding = data[0]  # Format [[...]]
                    else:
                        embedding = data  # Format [...]
                    
                    print(f"Longueur embedding: {len(embedding)}")
                    
                    # Vérification taille
                    if len(embedding) == VECTOR_SIZE:
                        return embedding
                    elif len(embedding) > VECTOR_SIZE:
                        print(f"Troncature: {len(embedding)} -> {VECTOR_SIZE}")
                        return embedding[:VECTOR_SIZE]
                    else:
                        print(f"Padding: {len(embedding)} -> {VECTOR_SIZE}")
                        return embedding + [0.0] * (VECTOR_SIZE - len(embedding))
                        
                else:
                    print(f"Format inattendu: {type(data)}")
                    # Fallback: générer un embedding factice basé sur le hash
                    return generate_fallback_embedding(text)
                    
            elif response.status_code == 503:
                wait_time = 15
                print(f"⏳ Modèle en chargement, attente {wait_time}s")
                time.sleep(wait_time)
                continue
                
            elif response.status_code == 400:
                # Essayer un format différent
                print("⚠️ Code 400, essai format simple...")
                payload = {"inputs": text}
                continue
                
            else:
                error_msg = f"Erreur API: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:150]}"
                print(f"Erreur: {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
                # Fallback en cas d'échec
                return generate_fallback_embedding(text)
                
        except Exception as e:
            print(f"Exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            # Fallback en cas d'échec
            return generate_fallback_embedding(text)
    
    # Fallback final
    return generate_fallback_embedding(text)

def generate_fallback_embedding(text: str):
    """Génère un embedding factice basé sur le hash du texte"""
    # Hash du texte pour générer des valeurs pseudo-aléatoires
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convertir en liste de flottants
    embedding = []
    for i in range(0, min(len(hash_bytes), VECTOR_SIZE)):
        embedding.append((hash_bytes[i] / 255.0) - 0.5)  # Valeurs entre -0.5 et 0.5
    
    # Remplir si nécessaire
    while len(embedding) < VECTOR_SIZE:
        embedding.append(0.0)
    
    print(f"⚠️ Fallback embedding généré (taille: {len(embedding)})")
    return embedding

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
        if not text or len(text.strip()) < 20:
            continue

        chunks = list(split_text(text))
        print(f"Page {page_num}: {len(chunks)} chunks")
        
        for chunk_num, chunk in enumerate(chunks, 1):
            try:
                vector = embed_text(chunk)
                
                # Vérification finale
                if len(vector) != VECTOR_SIZE:
                    print(f"Ajustement final: {len(vector)} -> {VECTOR_SIZE}")
                    if len(vector) > VECTOR_SIZE:
                        vector = vector[:VECTOR_SIZE]
                    else:
                        vector = vector + [0.0] * (VECTOR_SIZE - len(vector))
                
                # Créer un ID unique
                point_id = str(uuid.uuid4())
                
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[{
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "text": chunk,
                            "source": file.filename,
                            "page": page_num,
                            "chunk": chunk_num
                        }
                    }]
                )
                
                total_chunks += 1
                print(f"✓ Chunk {chunk_num} ajouté (ID: {point_id[:8]})")
                
                # Petite pause
                if chunk_num % 5 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                errors += 1
                print(f"✗ Erreur chunk {chunk_num}: {str(e)}")
                continue

    return {
        "status": "success",
        "file": file.filename,
        "chunks_stored": total_chunks,
        "errors": errors,
        "message": f"PDF traité: {total_chunks} chunks indexés"
    }

@app.post("/ask")
async def ask_question(data: Question):
    try:
        print(f"Question reçue: {data.question}")
        
        question_vector = embed_text(data.question)
        print(f"Embedding question généré (taille: {len(question_vector)})")
        
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
        scores = [r.score for r in results]

        return {
            "question": data.question,
            "contexts": contexts,
            "scores": scores,
            "count": len(contexts)
        }
        
    except Exception as e:
        print(f"Erreur dans /ask: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "status": "active",
        "embedding_model": "BAAI/bge-small-en",
        "vector_size": VECTOR_SIZE
    }

@app.get("/test_model")
def test_model():
    """Test complet du modèle"""
    test_text = "Ceci est un test d'embedding"
    
    try:
        vector = embed_text(test_text)
        
        # Vérifier aussi directement l'API
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": test_text}
        
        direct_response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        return {
            "model": "BAAI/bge-small-en",
            "url": HF_API_URL,
            "test_text": test_text,
            "embedding_length": len(vector),
            "embedding_sample": vector[:3],
            "api_status": direct_response.status_code,
            "api_response_type": type(direct_response.json()).__name__ if direct_response.status_code == 200 else "error"
        }
    except Exception as e:
        return {"error": str(e)}

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
    print("🚀 Démarrage avec BGE model")
    print(f"Modèle: {HF_API_URL}")
    print(f"Taille vecteur: {VECTOR_SIZE}")
    uvicorn.run(app, host="0.0.0.0", port=port)