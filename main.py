from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid, os, requests

app = FastAPI()

# ENV
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# INIT
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
