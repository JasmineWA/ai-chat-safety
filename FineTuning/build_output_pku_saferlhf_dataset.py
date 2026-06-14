import argparse
import json
import os
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_JSONL = os.path.join(BASE_DIR, "data", "train.jsonl")
DATAFILES_DIR = os.path.join(SCRIPT_DIR, "datafiles")
DEFAULT_OUTPUT_PREFIX = "output_pku_saferlhf_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PKU-SafeRLHF jsonl data into output-libra-style classification records"
    )
    parser.add_argument(
        "--input_jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Input PKU-SafeRLHF JSONL path",
    )
    parser.add_argument(
        "--output_prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output file prefix under FineTuning/datafiles/",
    )
    return parser.parse_args()


def normalize_text(value):
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in value.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def to_joint_text(question, response):
    return f"人类问题: {question}\n大模型回答: {response}"


def label_to_cn(label):
    return "安全" if label == "safe" else "不安全"


def flatten_harm_categories(harm_categories):
    if not isinstance(harm_categories, dict):
        return []
    return [key for key, value in harm_categories.items() if value]


def convert_row(row, response_index):
    prompt = normalize_text(row.get("prompt", ""))
    response = normalize_text(row.get(f"response_{response_index}", ""))
    safe_flag = row.get(f"is_response_{response_index}_safe")

    if not prompt or not response or not isinstance(safe_flag, bool):
        return None

    label = "safe" if safe_flag else "unsafe"
    harm_categories = row.get(f"response_{response_index}_harm_category", {})

    return {
        "text": to_joint_text(prompt, response),
        "question": prompt,
        "response": response,
        "label": label,
        "label_cn": label_to_cn(label),
        "category": "",
        "subcategory": "",
        "source": normalize_text(row.get(f"response_{response_index}_source", "")),
        "dataset_source": "PKU-SafeRLHF",
        "original_id": row.get("id"),
        "is_refusal_safe": False,
        "prompt_source": normalize_text(row.get("prompt_source", "")),
        "response_index": response_index,
        "better_response_id": row.get("better_response_id"),
        "safer_response_id": row.get("safer_response_id"),
        "severity_level": row.get(f"response_{response_index}_severity_level"),
        "harm_categories": flatten_harm_categories(harm_categories),
        "response_sha256": normalize_text(row.get(f"response_{response_index}_sha256", "")),
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset(source_path):
    samples = []
    bad_lines = []
    seen = set()
    total_lines = 0

    with open(source_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue

            try:
                row = json.loads(raw)
            except Exception as exc:
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "preview": raw[:300],
                    }
                )
                continue

            for response_index in (0, 1):
                sample = convert_row(row, response_index)
                if sample is None:
                    continue

                dedup_key = (
                    sample["question"],
                    sample["response"],
                    sample["label"],
                    sample["response_index"],
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                samples.append(sample)

    label_distribution = Counter(sample["label"] for sample in samples)
    severity_distribution = Counter(str(sample["severity_level"]) for sample in samples)
    source_distribution = Counter(sample["source"] for sample in samples)

    stats = {
        "source_path": source_path,
        "raw_line_count": total_lines,
        "bad_line_count": len(bad_lines),
        "sample_count": len(samples),
        "label_distribution": dict(label_distribution),
        "severity_distribution": dict(severity_distribution),
        "source_distribution": dict(source_distribution),
        "unique_text_count": len({sample["text"] for sample in samples}),
    }

    return samples, bad_lines, stats


def main():
    args = parse_args()
    os.makedirs(DATAFILES_DIR, exist_ok=True)

    output_json = os.path.join(DATAFILES_DIR, f"{args.output_prefix}.json")
    output_jsonl = os.path.join(DATAFILES_DIR, f"{args.output_prefix}.jsonl")
    output_badlines = os.path.join(DATAFILES_DIR, f"{args.output_prefix}_bad_lines.json")
    output_stats = os.path.join(DATAFILES_DIR, f"{args.output_prefix}_stats.json")

    samples, bad_lines, stats = build_dataset(args.input_jsonl)
    save_json(output_json, samples)
    save_jsonl(output_jsonl, samples)
    save_json(output_badlines, bad_lines)
    save_json(output_stats, stats)

    print("=" * 60)
    print("PKU-SafeRLHF output-format dataset built")
    print("=" * 60)
    print(f"input_jsonl:  {args.input_jsonl}")
    print(f"output_json:  {output_json}")
    print(f"output_jsonl: {output_jsonl}")
    print(f"bad_lines:    {output_badlines}")
    print(f"stats_json:   {output_stats}")
    print("")
    print(stats)


if __name__ == "__main__":
    main()
