import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# Vector store on disk – each session gets its own index file.
# --------------------------------------------------------------
BASE_RAG_DIR = Path.home() / ".ollama_chat" / "rag"
BASE_RAG_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")  # ~ 384‑dim, fast & small


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _index_path(session_id: str) -> Path:
    return BASE_RAG_DIR / f"{session_id}.index"


def _metadata_path(session_id: str) -> Path:
    return BASE_RAG_DIR / f"{session_id}.metadata.json"


def _load_metadata(session_id: str) -> List[dict]:
    p = _metadata_path(session_id)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def _save_metadata(session_id: str, meta: List[dict]) -> None:
    p = _metadata_path(session_id)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def add_documents(session_id: str, docs: List[str]) -> None:
    """
    Encode docs, add them to the FAISS index for the session, and store
    a tiny JSON metadata list with original text.
    """
    if not docs:
        return

    embeddings = EMBEDDER.encode(docs, normalize_embeddings=True)
    dim = embeddings.shape[1]

    index_file = _index_path(session_id)

    if index_file.is_file():
        index = faiss.read_index(str(index_file))
    else:
        index = faiss.IndexFlatIP(dim)   # inner product = cosine after norm

    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, str(index_file))

    # Update metadata
    meta = _load_metadata(session_id)
    for doc in docs:
        meta.append({"id": _hash(doc), "text": doc})
    _save_metadata(session_id, meta)


def _search_index(session_id: str, query: str, k: int = 4) -> List[str]:
    """Return top‑k raw documents for a query."""
    index_file = _index_path(session_id)
    if not index_file.is_file():
        return []

    index = faiss.read_index(str(index_file))
    q_emb = EMBEDDER.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype=np.float32), k)

    meta = _load_metadata(session_id)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(meta):
            continue
        results.append(meta[idx]["text"])
    return results


def retrieve(session_id: str, query: str) -> str:
    """
    Retrieve relevant chunks, concatenate them and return a single string.
    The caller can feed this into the LLM as context.
    """
    docs = _search_index(session_id, query, k=4)
    return "\n---\n".join(docs)
