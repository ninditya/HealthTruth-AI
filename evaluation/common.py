import json
import os
import re
import sys
from pathlib import Path

import build_eval_index

EVAL_DIR = Path(__file__).resolve().parent
APP_DIR = EVAL_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

DATASET_FILE = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.json"


def load_dataset() -> list[dict]:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_status(raw_output: str) -> tuple[str, str]:
    """Extracts (status, explanation) from the RAG engine's fact_check JSON output,
    falling back to the raw text if the model didn't return valid JSON."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_output or "").strip())
    try:
        parsed = json.loads(stripped)
        return parsed.get("status", "UNKNOWN"), parsed.get("explanation", raw_output)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\b(HOAX|BENAR|TIDAK LENGKAP)\b", raw_output or "", re.IGNORECASE)
        status = match.group(1).upper() if match else "UNKNOWN"
        return status, raw_output


def generate_predictions(k: int = 4) -> list[dict]:
    # Point RAGEngine at evaluation/corpus + evaluation/index instead of data/raw,
    # so this never reads or overwrites production data.
    os.environ.update(build_eval_index.EVAL_PATH_ENV)
    if not (build_eval_index.INDEX_DIR / "chunks.json").exists():
        print("[INFO] Index evaluasi belum ada, membangun dari evaluation/corpus/ ...")
        build_eval_index.build()

    from rag_engine import RAGEngine

    engine = RAGEngine()
    dataset = load_dataset()
    predictions = []

    for item in dataset:
        retrieved = engine.retrieve(item["claim"], k=k)
        raw_output = engine.fact_check(item["claim"])
        status, explanation = _parse_status(raw_output)

        predictions.append({
            **item,
            "retrieval_context": [c["text"] for c in retrieved],
            "raw_output": raw_output,
            "actual_output": explanation,
            "predicted_status": status,
        })
        print(f"[{item['id']}] expected={item['expected_status']} predicted={status}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(predictions)} predictions -> {PREDICTIONS_FILE}")
    return predictions


def load_predictions(regenerate: bool = False) -> list[dict]:
    if regenerate or not PREDICTIONS_FILE.exists():
        return generate_predictions()
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    generate_predictions()
