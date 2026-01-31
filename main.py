from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid, os, requests

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "rag_docs"

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config={"size": 384, "distance": "Cosine"}
)

class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "RAG server running"}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

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

    return {"status": "PDF ingested", "chunks": len(points)}


@app.post("/query")
def query_rag(q: Question):
    query_vector = model.encode(q.question).tolist()

    search = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=5
    )

    if not search:
        return {
            "answer": "Aucun document trouvé. Merci d’uploader un PDF.",
            "sources": []
        }

    context = "\n".join([hit.payload["text"] for hit in search])

    prompt = f"""
    Réponds à la question uniquement avec le contexte.

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
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    answer = response.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": list(set([hit.payload["source"] for hit in search]))
    }
