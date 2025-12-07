import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="HealthTruth AI", layout="wide")

# init engine
engine = RAGEngine()

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🛡️ HealthTruth AI — Anti Hoax Kesehatan")

# render pesan sebelumny
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input chat
user_input = st.chat_input("Tulis pertanyaan atau klaim kesehatan…")

if user_input:
    # tampilkan user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # panggil RAG
    with st.chat_message("assistant"):
        with st.spinner("Memproses..."):
            # default mode bisa apa saja, misalnya "ringkas"
            answer = engine.answer(user_input, mode="ringkas")
            st.write(answer)

    # simpan jawaban
    st.session_state.messages.append({"role": "assistant", "content": answer})

# footer
st.markdown("---")
st.caption("HealthTruth AI — Gemini + RAG + FAISS")