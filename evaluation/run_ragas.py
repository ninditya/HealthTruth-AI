import argparse

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from common import RESULTS_DIR, load_predictions
from llm_judge import build_ragas_llm

REPORT_FILE = RESULTS_DIR / "ragas_report.csv"


def build_ragas_dataset(predictions: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question": [p["claim"] for p in predictions],
        "answer": [p["actual_output"] for p in predictions],
        "contexts": [p["retrieval_context"] for p in predictions],
        "ground_truth": [p["reference_answer"] for p in predictions],
    })


def main():
    parser = argparse.ArgumentParser(description="Evaluate HealthTruth-AI RAG pipeline with RAGAS")
    parser.add_argument("--regenerate", action="store_true", help="Re-run the RAG pipeline instead of using cached predictions.json")
    args = parser.parse_args()

    predictions = load_predictions(regenerate=args.regenerate)
    dataset = build_ragas_dataset(predictions)

    llm = LangchainLLMWrapper(build_ragas_llm())
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORT_FILE, index=False)

    print(df.to_string())
    print("\n=== RAGAS Summary (rata-rata semua sampel) ===")
    print(result)
    print(f"\n[SAVED] per-sample scores -> {REPORT_FILE}")


if __name__ == "__main__":
    main()
