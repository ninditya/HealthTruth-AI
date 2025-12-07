import os, json, re
from config import RAW_REFERENCES_DIR, HOAX_SAMPLES_FILE, CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
from utils_preprocess import clean_text, chunk_text, load_pdf, load_txt

def load_documents():
    docs = []

    # references
    for f in os.listdir(RAW_REFERENCES_DIR):
        path = os.path.join(RAW_REFERENCES_DIR, f)
        if f.lower().endswith(".pdf"):
            docs.append({"source": f, "text": clean_text(load_pdf(path))})
        elif f.lower().endswith(".txt"):
            docs.append({"source": f, "text": clean_text(load_txt(path))})

    # hoax dataset
    with open(HOAX_SAMPLES_FILE, "r", encoding="utf-8") as f:
        j = json.load(f)
        text = "\n\n".join([x["text"] for x in j["hoax_examples"]])
        docs.append({"source": "hoax_samples", "text": clean_text(text)})

    return docs

def main():
    docs = load_documents()
    all_chunks = []

    for d in docs:
        chunks = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "source": d["source"],
                "chunk_id": i,
                "text": ch
            })

    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"SAVED {len(all_chunks)} chunks → {CHUNKS_FILE}")

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------