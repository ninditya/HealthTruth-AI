import os
import json
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
from config import (
    EMBED_MODEL, CHUNKS_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE
)

# --- Init Gemini Client dengan API key ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY belum di-set di environment atau .env")

client = genai.Client(api_key=api_key)

# --- Load chunks ---
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"{CHUNKS_FILE} tidak ditemukan")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Gunakan hanya sebagian data saat testing agar tidak melebihi kuota
texts = [c["text"] for c in chunks[:10]]  # contoh: 10 chunk pertama


# --- Build embeddings ---
emb_vectors = []
print(f"[INFO] Embedding {len(texts)} chunks...")

for t in texts:
    res = client.models.embed_content(
        model=EMBED_MODEL,   # contoh: "gemini-embedding-001"
        contents=[t],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    # Gemini v1 API → embeddings ada di .embeddings
    vector = np.array(res.embeddings[0].values, dtype="float32")
    emb_vectors.append(vector)

# Stack agar lebih stabil
emb_vectors = np.stack(emb_vectors).astype("float32")

# --- Build FAISS index ---
print("[INFO] Building FAISS index...")
index = faiss.IndexFlatL2(emb_vectors.shape[1])
index.add(emb_vectors)

# --- Save FAISS index ---
faiss.write_index(index, FAISS_INDEX_FILE)

# --- Save raw embeddings ---
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(emb_vectors, f)

print("[SUCCESS] FAISS index built and embeddings stored.")
