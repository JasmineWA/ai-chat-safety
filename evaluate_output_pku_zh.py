import argparse
import json
import os
import time
from collections import defaultdict

from local_model import OUTPUT_LABEL_CONFIG, OnnxLocalModel
from logger import get_logger


logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL_PATH = os.path.join(
    BASE_DIR,
    "FineTuning",
    "datafiles",
    "output_pku_saferlhf_eval_zh.json",
)
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "FineTuning",
    "models",
    "risk_classifier_output_pku_zh",
    "model.onnx",
)


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_result_row(index, case, prediction):
    expected_label = case["label"]
    actual_label = prediction.get("risk_category", "")
    confidence = float(prediction.get("confidence", 0.0))
    return {
        "index": index,
        "text": case["text"][:120],
        "expected_label": expected_label,
        "actual_label": actual_label,
        "confidence": round(confidence, 4),
        "correct": actual_label == expected_label,
    }


def compute_metrics(results):
    total = len(results)
    correct = sum(1 for row in results if row["correct"])
    by_label = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in results:
        label = row["expected_label"]
        by_label[label]["total"] += 1
        if row["correct"]:
            by_label[label]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "label_accuracy": round(correct / total * 100, 2) if total else 0,
        "by_label": {
            label: {
                "total": info["total"],
                "correct": info["correct"],
                "label_accuracy": round(info["correct"] / info["total"] * 100, 2)
                if info["total"]
                else 0,
            }
            for label, info in sorted(by_label.items())
        },
    }


def create_model(model_path):
    model = OnnxLocalModel(
        model_path,
        default_path=DEFAULT_MODEL_PATH,
        label_config=OUTPUT_LABEL_CONFIG,
    )
    if not model.is_loaded:
        raise RuntimeError(f"ONNX model failed to load: {model.load_error}")
    return model


def run_evaluation(eval_path, model_path=None, limit=None):
    cases = load_cases(eval_path)
    if limit is not None:
        cases = cases[:limit]

    resolved_model_path = model_path or DEFAULT_MODEL_PATH
    model = create_model(resolved_model_path)
    results = []
    start = time.time()

    logger.info("Chinese PKU output model eval loaded: %d cases", len(cases))

    for idx, case in enumerate(cases, start=1):
        prediction = model.predict(case["text"], [])
        results.append(build_result_row(idx, case, prediction))

        if idx % 500 == 0:
            logger.info("[%d/%d] completed", idx, len(cases))

    elapsed = time.time() - start
    metrics = compute_metrics(results)
    logger.info("Chinese PKU output model eval finished in %.1fs", elapsed)
    logger.info("Label accuracy: %.2f%%", metrics["label_accuracy"])

    for label, info in metrics["by_label"].items():
        logger.info(
            "  %s: label_acc=%.2f%% correct=%d total=%d",
            label,
            info["label_accuracy"],
            info["correct"],
            info["total"],
        )

    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Chinese PKU output-side ONNX classification model"
    )
    parser.add_argument("--eval_path", default=DEFAULT_EVAL_PATH, help="Eval dataset path")
    parser.add_argument("--model_path", default=None, help="Optional ONNX model path")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of eval samples")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    metrics, results = run_evaluation(
        eval_path=args.eval_path,
        model_path=args.model_path,
        limit=args.limit,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
        logger.info("Saved eval report to %s", args.output)


if __name__ == "__main__":
    main()
