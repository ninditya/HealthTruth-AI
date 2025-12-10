import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="HealthTruth AI", layout="wide")

engine = RAGEngine()

st.title("🛡️ HealthTruth AI — Anti Hoax Kesehatan (RAG + Gemini)")

query = st.text_area("Tulis pesan WhatsApp / klaim kesehatan yang ingin dicek:")

mode = st.radio(
    "Mode jawaban:",
    ["ringkas", "detail", "sumber"],
    horizontal=True
)

if st.button("Cek Fakta"):
    if not query.strip():
        st.error("Masukkan teks!")
    else:
        with st.spinner("Memproses..."):
            answer = engine.answer(query, mode)

        st.subheader("Hasil")
        st.write(answer)

st.markdown("---")
st.caption("HealthTruth AI — Gemini + RAG + FAISS")