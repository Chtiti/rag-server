from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid
import os
import requests

# ------------------ APP ------------------
app = FastAPI(title="RAG Server")

# ------------------ ENV ------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    raise RuntimeError("❌ Missing environment variables")

# ------------------ QDRANT ------------------
COLLECTION = "rag_docs"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# Create collection ONLY if not exists
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

# ------------------ MODEL (LAZY LOAD) ------------------
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# ------------------ SCHEMAS ------------------
class Question(BaseModel):
    question: str

# ------------------ ROUTES ------------------
@app.get("/")
def health():
    return {"status": "RAG server running"}

# ------------------ INGEST PDF ------------------
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    reader = PdfReader(file.file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Empty PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(full_text)
    model = get_model()

    points = []
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": chunk,
                "source": file.filename
            }
        })

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    return {
        "status": "PDF ingested successfully",
        "chunks": len(points)
    }

# ------------------ QUERY ------------------
@app.post("/query")
def query_rag(q: Question):
    model = get_model()
    query_vector = model.encode(q.question).tolist()

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=5
    )

    if not results:
        return {
            "answer": "Aucun document trouvé. Veuillez uploader un PDF.",
            "sources": []
        }

    context = "\n".join(hit.payload["text"] for hit in results)

    prompt = f"""
Réponds uniquement avec le contexte suivant.

CONTEXTE:
{context}

QUESTION:
{q.question}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="LLM error")

    answer = response.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": list({hit.payload["source"] for hit in results})
    }
