import json
import os
import random
from collections import Counter, defaultdict


random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAFETY_PROMPTS_DIR = os.path.join(DATA_DIR, "Safety-Prompts")
DATASETS_DIR = os.path.join(BASE_DIR, "FineTuning", "datasets")

TRAIN_JSON = os.path.join(DATASETS_DIR, "safety_prompts_train.json")
TRAIN_JSONL = os.path.join(DATASETS_DIR, "safety_prompts_train.jsonl")
EVAL_JSON = os.path.join(DATASETS_DIR, "safety_prompts_eval.json")
EVAL_JSONL = os.path.join(DATASETS_DIR, "safety_prompts_eval.jsonl")
STATS_JSON = os.path.join(DATASETS_DIR, "safety_prompts_dataset_stats.json")

os.makedirs(DATASETS_DIR, exist_ok=True)


INPUT_LABEL_CONFIG = {
    "Instruction_Attack": {
        "risk_level": "high",
        "risk_subcategory": "instruction_attack",
        "action": "block",
    },
    "Social_Manipulation_And_Harmful_Guidance": {
        "risk_level": "high",
        "risk_subcategory": "social_manipulation",
        "action": "block",
    },
    "Illegal_And_Criminal_Content": {
        "risk_level": "high",
        "risk_subcategory": "illegal_activity",
        "action": "block",
    },
    "Abuse_And_Hate": {
        "risk_level": "medium",
        "risk_subcategory": "abuse_and_hate",
        "action": "replace",
    },
    "Privacy_And_Property_Risk": {
        "risk_level": "medium",
        "risk_subcategory": "privacy_and_property",
        "action": "mask",
    },
    "Mental_Health_Risk": {
        "risk_level": "medium",
        "risk_subcategory": "mental_health",
        "action": "warn",
    },
    "Benign_Safety_Discussion": {
        "risk_level": "safe",
        "risk_subcategory": "benign_safety_discussion",
        "action": "pass",
    },
}


SOURCE_TO_TARGET = {
    "Reverse_Exposure": "Benign_Safety_Discussion",
    "Goal_Hijacking": "Instruction_Attack",
    "Prompt_Leaking": "Instruction_Attack",
    "Unsafe_Instruction_Topic": "Illegal_And_Criminal_Content",
    "Role_Play_Instruction": "Instruction_Attack",
    "Inquiry_With_Unsafe_Opinion": "Social_Manipulation_And_Harmful_Guidance",
    "Unfairness_And_Discrimination": "Abuse_And_Hate",
    "Crimes_And_Illegal_Activities": "Illegal_And_Criminal_Content",
    "Insult": "Abuse_And_Hate",
    "Mental_Health": "Mental_Health_Risk",
    "Physical_Harm": "Social_Manipulation_And_Harmful_Guidance",
    "Privacy_And_Property": "Privacy_And_Property_Risk",
    "Ethics_And_Morality": "Benign_Safety_Discussion",
}


def load_source_files():
    merged = defaultdict(list)
    for filename in [
        "instruction_attack_scenarios.json",
        "typical_safety_scenarios.json",
    ]:
        path = os.path.join(SAFETY_PROMPTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for source_type, items in data.items():
            merged[source_type].extend(items)
    return merged


def map_item(source_type, item):
    target_label = SOURCE_TO_TARGET.get(source_type)
    if not target_label:
        return None

    prompt = (item.get("prompt") or "").strip()
    if not prompt:
        return None

    cfg = INPUT_LABEL_CONFIG[target_label]
    return {
        "text": prompt,
        "label": target_label,
        "risk_level": cfg["risk_level"],
        "risk_subcategory": cfg["risk_subcategory"],
        "action": cfg["action"],
        "source_type": source_type,
        "source_response": (item.get("response") or "").strip(),
    }


def deduplicate(samples):
    seen = set()
    output = []
    for sample in samples:
        text = sample["text"]
        if text in seen:
            continue
        seen.add(text)
        output.append(sample)
    return output


def stratified_split(samples_by_label, eval_ratio=0.15, min_eval_per_class=200):
    train_samples = []
    eval_samples = []

    for label, samples in samples_by_label.items():
        random.shuffle(samples)
        eval_count = max(min_eval_per_class, int(len(samples) * eval_ratio))
        eval_count = min(eval_count, len(samples) - 1) if len(samples) > 1 else len(samples)
        eval_part = samples[:eval_count]
        train_part = samples[eval_count:]

        if not train_part and eval_part:
            train_part = eval_part[:1]
            eval_part = eval_part[1:]

        train_samples.extend(train_part)
        eval_samples.extend(eval_part)

    random.shuffle(train_samples)
    random.shuffle(eval_samples)
    return train_samples, eval_samples


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize(samples):
    counter = Counter(sample["label"] for sample in samples)
    source_counter = Counter(sample["source_type"] for sample in samples)
    return {
        "by_label": dict(counter),
        "by_source_type": dict(source_counter),
        "total": len(samples),
    }


def main():
    merged = load_source_files()

    raw_samples = []
    for source_type, items in merged.items():
        for item in items:
            mapped = map_item(source_type, item)
            if mapped:
                raw_samples.append(mapped)

    raw_samples = deduplicate(raw_samples)

    samples_by_label = defaultdict(list)
    for sample in raw_samples:
        samples_by_label[sample["label"]].append(sample)

    train_samples, eval_samples = stratified_split(samples_by_label)

    save_json(TRAIN_JSON, train_samples)
    save_jsonl(TRAIN_JSONL, train_samples)
    save_json(EVAL_JSON, eval_samples)
    save_jsonl(EVAL_JSONL, eval_samples)

    stats = {
        "labels": list(INPUT_LABEL_CONFIG.keys()),
        "train": summarize(train_samples),
        "eval": summarize(eval_samples),
    }
    save_json(STATS_JSON, stats)

    print("=" * 60)
    print("Safety-Prompts 输入侧数据集已生成")
    print("=" * 60)
    print(f"训练集: {TRAIN_JSON}")
    print(f"评测集: {EVAL_JSON}")
    print(f"统计:   {STATS_JSON}")
    print("")
    print("[Train]")
    for label, count in stats["train"]["by_label"].items():
        print(f"  {label}: {count}")
    print("[Eval]")
    for label, count in stats["eval"]["by_label"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
