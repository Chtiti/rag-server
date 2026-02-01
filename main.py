from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os, uuid, requests

app = FastAPI(title="RAG Server (Render Free Safe)")

# ========= ENV =========
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("Missing environment variables")

COLLECTION = "rag_docs"

# ========= UTILS =========
def get_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )

def embed_text(text: str) -> list:
    r = requests.post(
        "https://api.groq.com/openai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        },
        timeout=15
    )

    if r.status_code != 200:
        raise HTTPException(500, "Embedding API error")

    return r.json()["data"][0]["embedding"]

# ========= SCHEMA =========
class Question(BaseModel):
    question: str

# ========= ROUTES =========
@app.get("/")
def health():
    return {"status": "RAG server running"}

# ========= INGEST PDF =========
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF allowed")

    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    reader = PdfReader(file.file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    client = get_qdrant()
    count = 0

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            if len(chunk) < 50:
                continue

            vector = embed_text(chunk)

            client.upsert(
                collection_name=COLLECTION,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "text": chunk,
                        "source": file.filename
                    }
                }]
            )

            count += 1

    if count == 0:
        raise HTTPException(400, "No text extracted from PDF")

    return {
        "status": "PDF ingested",
        "chunks": count
    }

# ========= QUERY =========
@app.post("/query")
def query_rag(q: Question):
    client = get_qdrant()
    query_vector = embed_text(q.question)

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=5
    )

    if not results:
        return {
            "answer": "Aucun document trouvé. Uploadez un PDF.",
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

    r = requests.post(
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
        timeout=20
    )

    if r.status_code != 200:
        raise HTTPException(500, "LLM error")

    answer = r.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": list({hit.payload["source"] for hit in results})
    }
