import json
import os

from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, CHUNKS_FILE, VECTOR_BACKEND
from vector_store import get_backend

# --- Load chunks ---
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"{CHUNKS_FILE} tidak ditemukan")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# --- Build embeddings ---
print(f"[INFO] Loading embedding model '{EMBED_MODEL}'...")
embedder = SentenceTransformer(EMBED_MODEL)

print(f"[INFO] Embedding {len(texts)} chunks...")
vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

# --- Build index on the configured backend (faiss/qdrant) ---
print(f"[INFO] Building index using backend='{VECTOR_BACKEND}'...")
backend = get_backend(chunks)
backend.build(chunks, vectors)

print("[SUCCESS] Vector index built and stored.")
