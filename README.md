# HealthTruth-AI 
Sistem RAG untuk mendeteksi hoax kesehatan dari pesan WhatsApp menggunakan Gemini API + FAISS.

## Arsitektur
User (input forward WA)
        ↓
Text/Image Preprocessor
        ↓
Embedding (Gemini Embedding)
        ↓
Vector DB (FAISS / Chroma)
        ↓
Retriever
        ↓
Gemini LLM (RAG Chain)
        ↓
Output in 3 formats (Ringkas, Detail, Sumber)

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

## Live
https://healthtruth-ai.streamlit.app/
