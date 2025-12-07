import os
import json
import numpy as np
import pickle
import faiss
from google import genai
from google.genai import types
from config import (
    MODEL_NAME,
    EMBED_MODEL,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
)
from prompts import FACT_CHECK_PROMPT  # sesuaikan nama file

class RAGEngine:
    def __init__(self):
        # --- Init Gemini Client dengan API key ---
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY belum di-set di environment atau .env")
        self.client = genai.Client(api_key=api_key)
        self.model = self.client.models

        # --- Load chunks ---
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # --- Load embeddings ---
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = None

        # --- Load FAISS index ---
        if os.path.exists(FAISS_INDEX_FILE):
            self.index = faiss.read_index(FAISS_INDEX_FILE)
        else:
            self.index = None

    # ------------------------------------
    # EMBEDDING
    # ------------------------------------
    def embed(self, text: str):
        res = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return np.array(res.embeddings[0].values, dtype="float32")

    # ------------------------------------
    # RETRIEVER
    # ------------------------------------
    def retrieve(self, query, k=3):
        if self.index is None or self.embeddings is None:
            raise RuntimeError("FAISS index atau embeddings belum tersedia. Jalankan build_index.py terlebih dahulu.")
        q_emb = self.embed(query)
        scores, idxs = self.index.search(np.array([q_emb]), k)
        return [self.chunks[i] for i in idxs[0]]

    # ------------------------------------
    # FACT CHECK USING PROMPT
    # ------------------------------------
    def fact_check(self, query: str):
        retrieved = self.retrieve(query, k=3)
        context = "\n\n".join([c["text"] for c in retrieved])
        prompt = FACT_CHECK_PROMPT.format(question=query, context=context)

        res = self.model.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return res.text

    # ------------------------------------
    # THREE MODES UI
    # ------------------------------------
    def answer(self, query, mode):
        retrieved = self.retrieve(query, k=3)
        context = "\n\n".join([x["text"] for x in retrieved])

        if mode == "ringkas":
            instruction = "Berikan jawaban ringkas, 2-3 kalimat, untuk masyarakat umum."
        elif mode == "detail":
            instruction = "Berikan jawaban lengkap, akurat, dan berbasis bukti ilmiah."
        else:
            instruction = "Sertakan daftar sumber resmi (WHO/Kemenkes/CDC)."

        prompt = f"""
KONTEKS:
{context}

PERTANYAAN:
{query}

INSTRUKSI:
{instruction}

Jawaban:
"""

        res = self.model.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return res.text
