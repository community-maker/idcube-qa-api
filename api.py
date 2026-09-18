"""
REST API wrapper around the IDCUBE Q&A retrieval pipeline, so an external
agent platform (or anything that can make an HTTP call) can use this data
as a tool, instead of re-implementing retrieval itself.

Same retrieval logic as ask.py: question -> embed -> search Chroma -> Claude
grounded answer -> answer + sources.

Requires ANTHROPIC_API_KEY to be set (see README.md).

Run:
  pip install fastapi uvicorn
  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /ask   {"question": "..."}  ->  {"answer": "...", "sources": [...]}
  GET  /health
"""

import os
import traceback

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ask import MODEL_NAME, COLLECTION_NAME, answer_question

app = FastAPI(title="IDCUBE Q&A API", version="1.0")

_state = {}


@app.on_event("startup")
def load_index():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    import anthropic

    _state["model"] = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=os.environ.get("CHROMA_DIR", "chroma_db"))
    _state["collection"] = chroma_client.get_collection(COLLECTION_NAME)
    _state["client"] = anthropic.Anthropic()
    _state["llm_model"] = os.environ.get("LLM_MODEL", "claude-sonnet-5")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    collection = _state.get("collection")
    return {"status": "ok", "chunks_indexed": collection.count() if collection else 0}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        answer, sources = answer_question(
            req.question, _state["model"], _state["collection"], _state["client"], _state["llm_model"]
        )
    except Exception:
        # TODO: remove this debug detail once the Render deploy is confirmed working.
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    return AskResponse(answer=answer, sources=sources)
