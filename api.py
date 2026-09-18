"""
REST API wrapper around the IDCUBE retrieval index, for an external agent
platform (Purple Fabric) to use as a "search this website's data" tool.

This service does retrieval ONLY: question -> embed -> search Chroma ->
return the matching chunks. It does not call any LLM itself -- the calling
agent platform (which has its own LLM step) is expected to take these chunks
as context and generate the final answer.

Run:
  pip install fastapi uvicorn
  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /search   {"question": "...", "top_k": 5}  ->  {"results": [...]}
  GET  /health
"""

import os

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ask import MODEL_NAME, COLLECTION_NAME, retrieve

app = FastAPI(title="IDCUBE Search API", version="1.0")

_state = {}


@app.on_event("startup")
def load_index():
    _state["model"] = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", "chroma_db"))
    _state["collection"] = chroma_client.get_collection(COLLECTION_NAME)


class SearchRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    url: str
    title: str = ""
    heading_path: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.get("/health")
def health():
    collection = _state.get("collection")
    return {"status": "ok", "chunks_indexed": collection.count() if collection else 0}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    chunks = retrieve(req.question, _state["model"], _state["collection"], k=req.top_k)
    results = [
        SearchResult(
            text=c["text"],
            url=c["url"],
            title=c.get("title", ""),
            heading_path=c.get("heading_path", ""),
        )
        for c in chunks
    ]
    return SearchResponse(results=results)
