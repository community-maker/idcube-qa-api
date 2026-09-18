"""
Ask a question, answered only from the IDCUBE website/PDF content indexed
in build_index.py. This is the actual "website Q&A agent" end of the pipeline:

  question -> embed -> search Chroma for top-k relevant chunks ->
  hand those chunks + question to Claude -> grounded answer with sources

Requires an Anthropic API key:
  set ANTHROPIC_API_KEY=sk-ant-...        (Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-...")

Usage:
  python ask.py "What controllers does IDCUBE offer?"
  python ask.py                     # interactive mode, blank line to quit
"""

import argparse
import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "idcube_india"
TOP_K = 5

SYSTEM_PROMPT = """You are a support assistant for IDCUBE Systems' India website \
(idcubesystems.com/in/en). Answer ONLY using the provided context chunks -- \
they are excerpts from the company's own website and PDFs. If the answer \
isn't in the context, say you don't know and suggest the user contact IDCUBE \
directly (contact@idcubesystems.com / +91 7676110110). Do not invent product \
names, specs, or prices that aren't in the context. Keep answers concise."""


def retrieve(question: str, model, collection, k=TOP_K):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, **meta})
    return chunks


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c['url']}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, model, collection, anthropic_client, llm_model: str):
    chunks = retrieve(question, model, collection)
    context = build_context(chunks)

    message = anthropic_client.messages.create(
        model=llm_model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
        }],
    )
    answer = message.content[0].text
    sources = sorted({c["url"] for c in chunks})
    return answer, sources


def main():
    ap = argparse.ArgumentParser(description="Ask a question about idcubesystems.com")
    ap.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    ap.add_argument("--persist-dir", default="chroma_db")
    ap.add_argument("--llm-model", default="claude-sonnet-5")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set the ANTHROPIC_API_KEY environment variable first.")
        print('  PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-..."')
        print("  bash:        export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    print("Loading embedding model and index...")
    model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Ready. Index has {collection.count()} chunks.\n")

    if args.question:
        questions = [args.question]
    else:
        questions = iter(lambda: input("Ask (blank to quit): ").strip(), "")

    for q in questions:
        if not q:
            break
        answer, sources = answer_question(q, model, collection, client, args.llm_model)
        print(f"\n{answer}\n")
        print("Sources:")
        for s in sources:
            print(f"  - {s}")
        print()


if __name__ == "__main__":
    main()
