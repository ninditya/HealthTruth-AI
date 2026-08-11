import argparse
import json

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from common import RESULTS_DIR, load_predictions
from llm_judge import OpenRouterJudge

REPORT_FILE = RESULTS_DIR / "deepeval_report.json"


def build_test_cases(predictions: list[dict]) -> list[LLMTestCase]:
    return [
        LLMTestCase(
            input=p["claim"],
            actual_output=p["actual_output"],
            expected_output=p["reference_answer"],
            retrieval_context=p["retrieval_context"],
            context=p["retrieval_context"],
        )
        for p in predictions
    ]


def classification_accuracy(predictions: list[dict]) -> dict:
    total = len(predictions)
    correct = sum(1 for p in predictions if p["predicted_status"] == p["expected_status"])
    mismatches = [
        {"id": p["id"], "claim": p["claim"], "expected": p["expected_status"], "predicted": p["predicted_status"]}
        for p in predictions
        if p["predicted_status"] != p["expected_status"]
    ]
    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0.0, "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description="Evaluate HealthTruth-AI RAG pipeline with DeepEval")
    parser.add_argument("--regenerate", action="store_true", help="Re-run the RAG pipeline instead of using cached predictions.json")
    args = parser.parse_args()

    predictions = load_predictions(regenerate=args.regenerate)
    judge = OpenRouterJudge()

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        ContextualPrecisionMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
        HallucinationMetric(threshold=0.3, model=judge),
    ]

    test_cases = build_test_cases(predictions)
    eval_result = evaluate(test_cases, metrics)

    acc = classification_accuracy(predictions)
    print("\n=== Klasifikasi HOAX/BENAR/TIDAK LENGKAP ===")
    print(f"Akurasi: {acc['correct']}/{acc['total']} ({acc['accuracy']:.1%})")
    for m in acc["mismatches"]:
        print(f"  MISMATCH [{m['id']}] expected={m['expected']} predicted={m['predicted']} :: {m['claim']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({"classification_accuracy": acc}, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] classification accuracy -> {REPORT_FILE}")
    print("(per-metric scores are printed above by deepeval's own evaluate() output)")


if __name__ == "__main__":
    main()
