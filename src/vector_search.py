import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EMBED_PATH = BASE_DIR / "outputs" / "document_embeddings.jsonl"

def load_embeddings():
    docs = []
    with EMBED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["embedding"] = np.array(rec["embedding"], dtype="float32")
            docs.append(rec)
    return docs

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding, docs, top_k=5):
    scored = []
    for d in docs:
        score = cosine_similarity(query_embedding, d["embedding"])
        scored.append((score, d))
    return sorted(scored, key=lambda x: -x[0])[:top_k]