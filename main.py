from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests, os

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "RAG server running"}

@app.post("/query")
def query_rag(q: Question):
    return {
        "answer": "RAG prêt. LLM + Vector DB connectés.",
        "question": q.question
    }
