"""
LoRA fine-tuning for the local risk classifier.

Defaults:
- Train set: FineTuning/datasets/safety_prompts_train.json
- Eval set:  FineTuning/datasets/safety_prompts_eval.json
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


HF_MIRROR = "https://hf-mirror.com"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_PATH = os.path.join(SCRIPT_DIR, "datasets", "safety_prompts_train.json")
DEFAULT_EVAL_PATH = os.path.join(SCRIPT_DIR, "datasets", "safety_prompts_eval.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "models", "risk_classifier_lora")
BASE_MODEL = os.path.join(SCRIPT_DIR, "models", "chinese-roberta-wwm-ext")


class RiskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_PATH, help="Training JSON path")
    parser.add_argument("--eval_path", type=str, default=DEFAULT_EVAL_PATH, help="Eval JSON path")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=256, help="Max token length")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mirror", action="store_true", help="Use HF mirror")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(train_items, eval_items):
    labels = sorted({item["label"] for item in train_items + eval_items})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def convert_items(items, label2id):
    texts = []
    labels = []
    counts = Counter()

    for item in items:
        label = item["label"]
        if label not in label2id:
            continue
        texts.append(item["text"])
        labels.append(label2id[label])
        counts[label] += 1

    return texts, labels, counts


def print_label_stats(title, label_list, counts, label2id):
    print(title)
    total = sum(counts.values())
    print(f"  total: {total}")
    for label in label_list:
        count = counts.get(label, 0)
        bar = "#" * max(1, count // 200) if count else ""
        print(f"  [{label2id[label]:2d}] {label}: {count} {bar}")


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        print(f"[mirror] HF_ENDPOINT={HF_MIRROR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LoRA fine-tuning")
    print(f"base_model: {BASE_MODEL}")
    print(f"device: {device}")
    print(f"output_dir: {args.output_dir}")
    print("=" * 60)

    print("\n[1/5] loading datasets")
    train_items = load_json(args.train_path)
    eval_items = load_json(args.eval_path) if os.path.exists(args.eval_path) else []

    if not eval_items:
        print("  eval file not found, using a split from training data")
        train_items, eval_items = train_test_split(
            train_items,
            test_size=0.15,
            random_state=args.seed,
        )

    label_list, label2id, id2label = build_label_maps(train_items, eval_items)
    num_labels = len(label_list)

    train_texts, train_labels, train_counts = convert_items(train_items, label2id)
    eval_texts, eval_labels, eval_counts = convert_items(eval_items, label2id)

    print_label_stats("  train stats", label_list, train_counts, label2id)
    print_label_stats("  eval stats", label_list, eval_counts, label2id)

    print("\n[2/5] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=args.max_length)
    eval_enc = tokenizer(eval_texts, truncation=True, padding=False, max_length=args.max_length)
    train_ds = RiskDataset(train_enc, train_labels)
    eval_ds = RiskDataset(eval_enc, eval_labels)

    print("\n[3/5] loading model and applying LoRA")
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value", "output.dense"],
        modules_to_save=["classifier"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    print("\n[4/5] training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    print("\n[5/5] evaluating and saving")
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\nclassification report:")
    print(classification_report(preds.label_ids, y_pred, target_names=label_list, digits=3))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "LABEL_LIST": label_list,
                "LABEL2ID": label2id,
                "ID2LABEL": id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nsaved to: {args.output_dir}")
    print(f"total params: {total/1e6:.1f}M")
    print(f"trainable params: {trainable/1e3:.1f}K ({trainable/total*100:.2f}%)")


if __name__ == "__main__":
    main()
