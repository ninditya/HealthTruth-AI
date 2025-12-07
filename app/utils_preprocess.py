import re
import PyPDF2

def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, size, overlap):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap

    return chunks
