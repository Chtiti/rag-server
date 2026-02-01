from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import requests
import uuid
import os
import time
from typing import List

# ================== CONFIG LÉGÈRE ==================

# Environnement Render - variables minimales
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️  HF_TOKEN non configuré dans Render Environment Variables")

# Modèle plus petit pour économiser la mémoire
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384  # Taille fixe pour ce modèle
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# ===================================================

app = FastAPI()

# Stockage SIMPLE en mémoire (léger)
documents = []  # Liste de tuples (id, vector, text, source)

# ================== FONCTIONS OPTIMISÉES ==================

def split_text_fast(text: str, chunk_size=300) -> List[str]:
    """Découpe simple sans chevauchement pour économiser la RAM"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

def get_embedding(text: str, max_retries=2) -> List[float]:
    """Embedding avec timeout court et peu de retry"""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN non configuré sur Render")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "parameters": {"truncation": True}}
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data[0] if isinstance(data[0], list) else data
                return data
            elif resp.status_code == 503 and attempt < max_retries - 1:
                time.sleep(10)
                continue
            else:
                raise Exception(f"API error: {resp.status_code}")
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise Exception("Timeout")
    
    raise Exception("Failed after retries")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calcul de similarité cosine simple"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def search_similar(query_vector: List[float], limit=3):
    """Recherche linéaire simple (assez rapide pour 500MB)"""
    results = []
    for doc_id, vector, text, source in documents:
        score = cosine_similarity(query_vector, vector)
        if score > 0.3:  # Seuil minimum
            results.append((score, text, source))
    
    # Trier par score
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:limit]

# ================== MODÈLES ==================

class Question(BaseModel):
    question: str

# ================== ENDPOINTS ==================

@app.get("/")
def root():
    return {
        "status": "ok",
        "memory": "optimized_500mb",
        "model": MODEL_NAME,
        "documents_count": len(documents)
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf_token_configured": bool(HF_TOKEN),
        "documents": len(documents)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload simple avec traitement page par page"""
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF only")
    
    try:
        # Lire PDF avec buffer limité
        reader = PdfReader(file.file)
        chunks_added = 0
        
        # Limiter à 20 pages max pour éviter OOM
        max_pages = min(20, len(reader.pages))
        
        for page_num, page in enumerate(reader.pages[:max_pages], 1):
            text = page.extract_text()
            if not text or len(text) < 50:
                continue
                
            # Découpage et traitement
            for chunk in split_text_fast(text):
                if len(chunk) < 30:
                    continue
                    
                try:
                    vector = get_embedding(chunk)
                    
                    # Vérifier taille du vecteur
                    if len(vector) != VECTOR_SIZE:
                        # Ajuster si nécessaire
                        if len(vector) > VECTOR_SIZE:
                            vector = vector[:VECTOR_SIZE]
                        else:
                            vector = vector + [0.0] * (VECTOR_SIZE - len(vector))
                    
                    # Stocker en mémoire
                    documents.append((
                        str(uuid.uuid4()),
                        vector,
                        chunk,
                        f"{file.filename}_p{page_num}"
                    ))
                    
                    chunks_added += 1
                    
                    # Limiter le nombre total de documents
                    if len(documents) > 1000:  # Hard limit pour 500MB
                        documents.pop(0)
                    
                    # Pause pour éviter de surcharger l'API Hugging Face
                    if chunks_added % 5 == 0:
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"Chunk error: {e}")
                    continue
        
        return {
            "status": "success",
            "file": file.filename,
            "chunks": chunks_added,
            "total_documents": len(documents),
            "warning": "Limited to 20 pages for memory safety" if len(reader.pages) > 20 else None
        }
        
    except Exception as e:
        raise HTTPException(500, f"Upload error: {str(e)}")

@app.post("/ask")
async def ask_question(data: Question):
    """Question simple avec recherche"""
    
    if not data.question.strip():
        raise HTTPException(400, "Question required")
    
    if not documents:
        return {
            "question": data.question,
            "answer": "No documents uploaded yet",
            "contexts": []
        }
    
    try:
        # Embedding de la question
        question_vector = get_embedding(data.question)
        
        # Recherche
        results = search_similar(question_vector, limit=3)
        
        if not results:
            return {
                "question": data.question,
                "answer": "No relevant information found",
                "contexts": []
            }
        
        # Extraire les contextes
        contexts = [text for _, text, _ in results]
        scores = [round(score, 3) for score, _, _ in results]
        
        return {
            "question": data.question,
            "contexts": contexts,
            "scores": scores,
            "top_score": scores[0] if scores else 0
        }
        
    except Exception as e:
        raise HTTPException(500, f"Ask error: {str(e)}")

@app.get("/stats")
def stats():
    """Statistiques minimales"""
    memory_usage = len(documents) * VECTOR_SIZE * 4 / 1024 / 1024  # Estimation MB
    return {
        "documents_count": len(documents),
        "estimated_memory_mb": round(memory_usage, 2),
        "vector_size": VECTOR_SIZE,
        "max_documents": 1000
    }

@app.delete("/clear")
def clear():
    """Vider la mémoire"""
    global documents
    count = len(documents)
    documents = []
    return {"status": "cleared", "documents_removed": count}

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
    
    print(f"🚀 Server starting on port {port}")
    print(f"📦 Model: {MODEL_NAME}")
    print(f"💾 Memory limit: 500MB")
    print(f"🔢 Vector size: {VECTOR_SIZE}")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")