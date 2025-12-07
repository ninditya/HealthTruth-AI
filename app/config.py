import os
from dotenv import load_dotenv
from google import genai  # gunakan import dari package terbaru

# --- Load environment variables ---
load_dotenv("env/.env")

# --- App settings ---
APP_TITLE = "HoaxShield Health AI"
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# --- Paths ---
HOAX_SAMPLES_FILE = "data/raw/hoax_samples.json"
RAW_REFERENCES_DIR = "data/raw/references/"
CHUNKS_FILE = "data/processed/chunks.json"
EMBEDDINGS_FILE = "data/processed/embeddings.pkl"
FAISS_INDEX_FILE = "vectorstore/index.faiss"

# --- Chunking ---
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# --- Init Gemini client ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY belum di-set di environment atau .env")

client = genai.Client(api_key=API_KEY)
