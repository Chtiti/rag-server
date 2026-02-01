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
import io

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    content = await file.read()
    print("PDF:", file.filename, "SIZE:", len(content))

    reader = PdfReader(io.BytesIO(content))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    total_chunks = 0

    for page in reader.pages:
        page_text = page.extract_text()
        print("PAGE TEXT:", page_text[:200] if page_text else "EMPTY")

        if not page_text:
            continue

        chunks = splitter.split_text(page_text)

        for chunk in chunks[:3]:
            response = requests.post(
                "https://api.groq.com/openai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": chunk
                },
                timeout=30
            )

            if response.status_code != 200:
                print("EMBED ERROR:", response.text)
                raise HTTPException(500, "Embedding failed")

            vector = response.json()["data"][0]["embedding"]

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

            total_chunks += 1

    if total_chunks == 0:
        raise HTTPException(400, "No text extracted")

    return {"status": "ok", "chunks": total_chunks}
    
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
