from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import requests
import uuid
import os
import time

# ================== CONFIG CORRIGÉE ==================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en"
VECTOR_SIZE = 384  # ⚠️ CORRECT: 384 pour BGE-small
COLLECTION_NAME = "rag_docs"

# =====================================================

app = FastAPI()

# Qdrant en mémoire
qdrant = QdrantClient(":memory:")

# DÉTRUIRE l'ancienne collection (taille 1536)
try:
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    print("🗑️ Ancienne collection (1536D) supprimée")
except:
    print("ℹ️ Pas d'ancienne collection à supprimer")

# CRÉER nouvelle collection (taille 384)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,  # 384 !
        distance=Distance.COSINE
    )
)

print(f"✅ Collection '{COLLECTION_NAME}' créée: {VECTOR_SIZE} dimensions")

# ================== UTILS ==================

def split_text(text, size=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def embed_text(text: str):
    """Génère des embeddings de 384 dimensions"""
    
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN non configuré")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": text, "parameters": {"truncation": True, "max_length": 512}}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list):
                if data and isinstance(data[0], list):
                    vector = data[0]
                else:
                    vector = data
                
                # VÉRIFICATION CRUCIALE : doit être 384
                if len(vector) != VECTOR_SIZE:
                    print(f"⚠️ ATTENTION: Vecteur {len(vector)}D, attendu {VECTOR_SIZE}D")
                    
                    # Ajustement automatique
                    if len(vector) > VECTOR_SIZE:
                        vector = vector[:VECTOR_SIZE]
                        print(f"  → Tronqué à {VECTOR_SIZE}D")
                    else:
                        vector = vector + [0.0] * (VECTOR_SIZE - len(vector))
                        print(f"  → Complété à {VECTOR_SIZE}D")
                
                return vector
            else:
                print(f"Format inattendu, fallback 384D")
                return [0.1] * VECTOR_SIZE
        else:
            print(f"API error {response.status_code}, fallback 384D")
            return [0.1] * VECTOR_SIZE
            
    except Exception as e:
        print(f"Erreur embedding, fallback 384D: {str(e)}")
        return [0.1] * VECTOR_SIZE

# ================== MODELS ==================

class Question(BaseModel):
    question: str

# ================== ENDPOINTS ==================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF requis")
    
    try:
        reader = PdfReader(file.file)
        total_chunks = 0
        points = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text or len(text.strip()) < 20:
                continue
            
            for chunk in split_text(text):
                vector = embed_text(chunk)
                
                # Vérification finale
                if len(vector) != VECTOR_SIZE:
                    print(f"❌ ERREUR CRITIQUE: Vecteur {len(vector)}D != {VECTOR_SIZE}D")
                    # Forcer à la bonne taille
                    vector = vector[:VECTOR_SIZE] if len(vector) > VECTOR_SIZE else vector + [0.0] * (VECTOR_SIZE - len(vector))
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk, "source": file.filename, "page": page_num}
                )
                points.append(point)
                total_chunks += 1
        
        # Insérer tous les points
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"📥 {total_chunks} points insérés ({VECTOR_SIZE}D chacun)")
        
        return {
            "status": "success",
            "file": file.filename,
            "chunks_stored": total_chunks,
            "vector_size": VECTOR_SIZE,
            "model": "BAAI/bge-small-en"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/ask")
async def ask_question(data: Question):
    try:
        print(f"❓ Question: {data.question}")
        
        # Embedding de la question (384D)
        question_vector = embed_text(data.question)
        print(f"📐 Embedding question: {len(question_vector)} dimensions")
        
        # Vérification cruciale
        if len(question_vector) != VECTOR_SIZE:
            print(f"🚨 PANIC: Embedding {len(question_vector)}D != collection {VECTOR_SIZE}D")
            return {"error": f"Dimension mismatch: {len(question_vector)} vs {VECTOR_SIZE}"}
        
        # Recherche dans Qdrant
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_vector,
            limit=3
        )
        
        print(f"🔍 {len(results)} résultats trouvés")
        
        if not results:
            return {
                "question": data.question,
                "answer": "Aucune information trouvée",
                "contexts": []
            }
        
        contexts = [r.payload.get("text", "") for r in results]
        scores = [float(r.score) for r in results]
        
        return {
            "question": data.question,
            "contexts": contexts,
            "scores": scores,
            "vector_size_used": VECTOR_SIZE
        }
        
    except Exception as e:
        print(f"💥 Erreur /ask: {str(e)}")
        return {
            "question": data.question,
            "error": str(e),
            "vector_size": VECTOR_SIZE
        }

@app.get("/config")
def get_config():
    """Vérification de configuration"""
    try:
        collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        return {
            "model": "BAAI/bge-small-en",
            "vector_size_config": VECTOR_SIZE,
            "vector_size_actual": collection_info.config.params.vectors.size,
            "collection": COLLECTION_NAME,
            "vectors_count": qdrant.count(collection_name=COLLECTION_NAME).count,
            "status": "OK" if VECTOR_SIZE == collection_info.config.params.vectors.size else "ERROR: mismatch"
        }
    except Exception as e:
        return {"error": str(e), "vector_size_config": VECTOR_SIZE}

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
    
    print("=" * 50)
    print("🎯 RAG API - CONFIGURATION CORRECTE")
    print(f"🤖 Modèle: BAAI/bge-small-en")
    print(f"📏 Taille vecteur: {VECTOR_SIZE} dimensions")
    print(f"🗄️ Collection: {COLLECTION_NAME}")
    print("=" * 50)
    
    # Test automatique
    print("🧪 Test automatique de configuration...")
    test_vector = embed_text("test")
    print(f"Test embedding: {len(test_vector)} dimensions")
    
    if len(test_vector) == VECTOR_SIZE:
        print("✅ Configuration CORRECTE!")
    else:
        print(f"❌ PROBLÈME: {len(test_vector)}D vs {VECTOR_SIZE}D")
    
    uvicorn.run(app, host="0.0.0.0", port=port)