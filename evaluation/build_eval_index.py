import os
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
CORPUS_DIR = EVAL_DIR / "corpus"
INDEX_DIR = EVAL_DIR / "index"

EVAL_PATH_ENV = {
    "RAW_REFERENCES_DIR": str(CORPUS_DIR / "references") + "/",
    "HOAX_SAMPLES_FILE": str(CORPUS_DIR / "hoax_samples.json"),
    "CHUNKS_FILE": str(INDEX_DIR / "chunks.json"),
    "EMBEDDINGS_FILE": str(INDEX_DIR / "embeddings.pkl"),
    "FAISS_INDEX_FILE": str(INDEX_DIR / "index.faiss"),
}


def build(env: dict = None) -> None:
    """Runs the app's own preprocess + build_index against the evaluation corpus,
    writing into evaluation/index/ (or a Qdrant collection) instead of data/processed
    or vectorstore/, so evaluation never touches production data."""
    run_env = {**os.environ, **EVAL_PATH_ENV, **(env or {})}
    run_env.setdefault("QDRANT_COLLECTION", "healthtruth_chunks_eval")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "app/preprocess.py"], cwd=ROOT_DIR, env=run_env, check=True)
    subprocess.run([sys.executable, "app/build_index.py"], cwd=ROOT_DIR, env=run_env, check=True)


if __name__ == "__main__":
    build()
