import os
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config import (
    MODEL_NAME,
    EMBED_MODEL,
    CHUNKS_FILE,
)
from prompts import FACT_CHECK_PROMPT
from vector_store import get_backend

class RAGEngine:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY belum di-set di env/.env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = MODEL_NAME
        self.embedder = SentenceTransformer(EMBED_MODEL)

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.backend = get_backend(self.chunks)

    # ------------------------------------
    # EMBEDDING
    # ------------------------------------
    def embed(self, text: str):
        return self.embedder.encode(text, convert_to_numpy=True).astype("float32")

    # ------------------------------------
    # RETRIEVER
    # ------------------------------------
    def retrieve(self, query, k=3):
        q_emb = self.embed(query)
        return self.backend.search(q_emb, k)

    # ------------------------------------
    # FACT CHECK USING PROMPT
    # ------------------------------------
    def fact_check(self, query: str):
        retrieved = self.retrieve(query, k=3)
        context = "\n\n".join([c["text"] for c in retrieved])
        prompt = FACT_CHECK_PROMPT.format(question=query, context=context)

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content

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

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content
