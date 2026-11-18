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
import re
import sys
from functools import cached_property

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

# Embedding is handled by Chroma's built-in default embedding function (a
# local ONNX build of all-MiniLM-L6-v2, via onnxruntime) rather than
# sentence-transformers/torch -- functionally the same model, but without
# torch's ~400MB+ runtime footprint, which matters on a memory-capped free
# hosting tier. build_index.py (index time) uses Chroma's default; queries
# use QueryEmbedder below, the same model tuned for one short question.
COLLECTION_NAME = "idcube_india"
TOP_K = 8


class QueryEmbedder(ONNXMiniLM_L6_V2):
    """Chroma's default embedding model, tuned for embedding one question.

    Chroma pads every input to 256 tokens and lets onnxruntime start a
    spinning thread per host core. On Render's 0.1-CPU free tier that made
    each search take 7-13 s. Without padding (mean pooling ignores pad
    tokens, so the vectors are identical) and on one thread, a question
    embeds in under 10 ms locally.
    """

    @cached_property
    def tokenizer(self):
        tok = super().tokenizer
        tok.no_padding()
        return tok

    @cached_property
    def model(self):
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )


_embedder = QueryEmbedder()


def embed_question(question: str):
    # One question per call: unpadded inputs of different lengths can't batch.
    return _embedder([question])[0]

SYSTEM_PROMPT = """You are a support assistant for IDCUBE Systems' website \
(idcubesystems.com). Answer ONLY using the provided context chunks -- \
they are excerpts from the company's own website and PDFs. If the answer \
isn't in the context, say you don't know and suggest the user contact IDCUBE \
directly (contact@idcubesystems.com). Do not invent product \
names, specs, or prices that aren't in the context. Keep answers concise."""

_SITE = "https://www.idcubesystems.com"
# When a question names a product, its own page should lead the results.
# The embedding model splits "GreenID" into "green"+"id", so "tell me about
# GreenID" otherwise ranks "Green Initiative" sections and short GreenID
# teasers from other pages above the GreenID page's feature list.
PRODUCT_PAGES = [
    (re.compile(r"\bgreen ?id\b", re.I), _SITE + "/products/solutions/greenid/"),
    (re.compile(r"\bez ?visit", re.I), _SITE + "/products/solutions/visitor-management/"),
    (re.compile(r"\battendio", re.I), _SITE + "/products/solutions/attendance/"),
    (re.compile(r"\bainsights?\b", re.I), _SITE + "/products/solutions/analytical-dashboard/"),
    (re.compile(r"\baccess ?master\b", re.I), _SITE + "/products/solutions/access-control/"),
    (re.compile(r"\bhelix\b", re.I), _SITE + "/products/platforms/access360-helix/"),
    (re.compile(r"\baccess ?360[ .]?ai\b", re.I), _SITE + "/products/platforms/access360-ai/"),
    (re.compile(r"\bmercury\b", re.I), _SITE + "/products/devices/controllers/mercury-icm/"),
    (re.compile(r"\baero\b", re.I), _SITE + "/products/devices/controllers/aero-ica/"),
    (re.compile(r"\bopen ?hex\b", re.I), _SITE + "/products/devices/controllers/openhex/"),
    (re.compile(r"\baccess ?box\b", re.I), _SITE + "/products/devices/enclosure/access-box/"),
]
PINNED_PER_PRODUCT = 3


def _query(collection, embedding, n, where=None):
    # build_index.py stores two vectors per chunk (plain + title/heading
    # context), so over-fetch 2x and let the caller keep each chunk once.
    kwargs = {"query_embeddings": [embedding], "n_results": min(n * 2, collection.count())}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return [{"text": doc, **meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]


def _unique(rows, limit):
    out, seen = [], set()
    for c in rows:
        chunk_id = c.get("chunk_id") or c["text"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            out.append(c)
            if len(out) == limit:
                break
    return out


def retrieve(question: str, collection, k=TOP_K):
    embedding = embed_question(question)
    pinned = []
    for pattern, page_url in PRODUCT_PAGES:
        if pattern.search(question):
            rows = _query(collection, embedding, PINNED_PER_PRODUCT, where={"url": page_url})
            pinned += _unique(rows, PINNED_PER_PRODUCT)
    return _unique(pinned + _query(collection, embedding, k), k)


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c['url']}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, collection, anthropic_client, llm_model: str):
    chunks = retrieve(question, collection)
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

    print("Loading index...")
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
        answer, sources = answer_question(q, collection, client, args.llm_model)
        print(f"\n{answer}\n")
        print("Sources:")
        for s in sources:
            print(f"  - {s}")
        print()


if __name__ == "__main__":
    main()
