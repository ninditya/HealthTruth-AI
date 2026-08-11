# HealthTruth-AI 
Sistem RAG untuk mendeteksi hoax kesehatan dari pesan WhatsApp menggunakan Gemini API + FAISS.

## Arsitektur
User (input forward WA) -> Text -> Embedding (Gemini Embedding) -> Vector DB (FAISS / Chroma) -> Retriever -> Gemini LLM (RAG Chain) -> Output

## Fitur
- Deteksi hoax kesehatan berbasis dokumen resmi WHO/Kemenkes.
- RAG pipeline: preprocessing → chunking → embedding → FAISS search.
- UI Streamlit: chatbot interface
- Dataset 21 hoax WA paling umum.

## Cara Pakai
1. Install requirements:  
   `pip install -r requirements.txt`

2. Set API Key:  
   `echo "GEMINI_API_KEY=xxxx" > env/.env`

3. Preprocess dokumen:  
   `python app/preprocess.py`

4. Bangun FAISS index:  
   `python build_index.py`

5. Jalankan app:  
   `streamlit run app/main.py`

## Vector Backend
`app/rag_engine.py` dan `app/build_index.py` mengambil backend vector store lewat `VECTOR_BACKEND`
di `env/.env` — abstraksinya ada di [`app/vector_store.py`](app/vector_store.py):

- **`faiss`** (default) — index file lokal (`vectorstore/index.faiss`), cocok untuk dev.
- **`qdrant`** — untuk staging/production. Set:
  ```
  VECTOR_BACKEND=qdrant
  QDRANT_URL=https://xxxx.qdrant.io   # atau ":memory:" untuk uji lokal tanpa server
  QDRANT_API_KEY=xxxx
  QDRANT_COLLECTION=healthtruth_chunks
  ```

Ganti backend, lalu jalankan ulang `python app/build_index.py` untuk membangun index pada backend
yang aktif — `RAGEngine` dan skrip di `evaluation/` otomatis ikut memakainya, tidak perlu diubah.

## Evaluasi
[`evaluation/`](evaluation/README.md) berisi dataset uji, corpus contoh, dan skrip evaluasi RAG
(DeepEval + RAGAS) dalam satu folder yang self-contained — punya corpus & index sendiri (terpisah
dari `data/raw/` produksi), jadi bisa langsung dijalankan setelah `pip install -r evaluation/requirements.txt`
tanpa setup data manual.

## Live
https://healthtruth-ai.streamlit.app/
