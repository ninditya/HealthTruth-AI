import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "env" / ".env")

# --- App settings ---
APP_TITLE = "HealthTruth-AI"
MODEL_NAME = "google/gemini-2.5-flash"   # model OpenRouter — bisa diganti sesuai kebutuhan
EMBED_MODEL = "all-MiniLM-L6-v2"         # sentence-transformers lokal

# --- Paths ---
# Override lewat env var supaya evaluation/ bisa membangun index sendiri (corpus contoh, terisolasi)
# tanpa menyentuh data produksi di data/raw/.
HOAX_SAMPLES_FILE = os.getenv("HOAX_SAMPLES_FILE", "data/raw/hoax_samples.json")
RAW_REFERENCES_DIR = os.getenv("RAW_REFERENCES_DIR", "data/raw/references/")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/processed/chunks.json")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "data/processed/embeddings.pkl")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "vectorstore/index.faiss")

# --- Chunking ---
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# --- Vector backend ---
# "faiss" untuk lokal/dev (index file di VECTOR STORE), "qdrant" untuk staging/production.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "healthtruth_chunks")

# --- OpenRouter API key ---
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY belum di-set di env/.env")
