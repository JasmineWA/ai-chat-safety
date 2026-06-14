import argparse
import concurrent.futures
import json
import os
import time
from typing import Iterable

from openai import OpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAFILES_DIR = os.path.join(SCRIPT_DIR, "datafiles")

DEFAULT_INPUT = os.path.join(DATAFILES_DIR, "output_pku_saferlhf_train.json")
DEFAULT_OUTPUT_PREFIX = "output_pku_saferlhf_train_zh"
DEFAULT_OUTPUT_JSON = os.path.join(DATAFILES_DIR, "output_pku_saferlhf_train_zh.json")
DEFAULT_OUTPUT_JSONL = os.path.join(DATAFILES_DIR, "output_pku_saferlhf_train_zh.jsonl")
DEFAULT_STATS = os.path.join(DATAFILES_DIR, "output_pku_saferlhf_train_zh_stats.json")
DEFAULT_CHECKPOINT = os.path.join(DATAFILES_DIR, "output_pku_saferlhf_train_zh_checkpoint.json")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "60"))

TRANSLATE_SYSTEM_PROMPT = """你是一个安全语料翻译助手。
你的任务是把英文的“用户问题”和“大模型回答”翻译成自然、准确、简洁的中文。

要求：
1. 必须忠实保留原意，不要遗漏、弱化或强化风险、拒绝、规劝、歧视、违法、心理健康、隐私等安全相关语义。
2. 不要解释，不要总结，不要添加原文中没有的信息。
3. 如果原回答本身不安全，必须如实翻译，不要替它改写成安全回答。
4. 如果原回答是拒绝、规劝、求助建议，必须准确保留。
5. 返回 JSON，不要使用 markdown 代码块。
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate output safety dataset question/response fields to Chinese with DeepSeek API"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON dataset path")
    parser.add_argument(
        "--output_prefix",
        default="",
        help="Optional output prefix under FineTuning/datafiles/. "
             "If set, output_json/output_jsonl/checkpoint/stats are auto-derived from it.",
    )
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Translated JSON output path")
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL, help="Translated JSONL output path")
    parser.add_argument("--stats", default=DEFAULT_STATS, help="Stats JSON output path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="DeepSeek base URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to process")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N successful samples")
    parser.add_argument("--sleep_seconds", type=float, default=0.05, help="Sleep between batch submissions")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per batch")
    parser.add_argument("--batch_size", type=int, default=5, help="How many samples to translate in one API call")
    parser.add_argument("--workers", type=int, default=4, help="How many API requests to run concurrently")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path, data: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_joint_text(question, response):
    return f"人类问题: {question}\n大模型回答: {response}"


def build_batch_prompt(batch_items):
    blocks = []
    for item in batch_items:
        blocks.append(
            f"[{item['batch_id']}]\n"
            f"question: {item['question_en']}\n"
            f"response: {item['response_en']}"
        )
    return (
        "请把下面多组英文问答分别翻译成自然中文，并严格保持语义与安全属性。\n"
        "返回 JSON 数组。每个元素必须包含 3 个字段：batch_id, question_zh, response_zh。\n"
        "不要输出任何解释，不要使用 markdown 代码块。\n\n"
        + "\n\n".join(blocks)
    )


def create_client(api_key, base_url, timeout):
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def parse_json_response(text):
    payload = (text or "").strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()
    return json.loads(payload)


def translate_batch(client, model, batch_items):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(batch_items)},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_response(content)
    if not isinstance(parsed, list):
        raise ValueError("Batch translation did not return a JSON list")

    result = {}
    for item in parsed:
        batch_id = int(item["batch_id"])
        result[batch_id] = (
            item["question_zh"].strip(),
            item["response_zh"].strip(),
        )
    return result


def build_output_record(item, question_zh, response_zh, model):
    new_item = dict(item)
    new_item["question_en"] = item.get("question", "")
    new_item["response_en"] = item.get("response", "")
    new_item["text_en"] = item.get("text", "")
    new_item["question"] = question_zh
    new_item["response"] = response_zh
    new_item["text"] = build_joint_text(question_zh, response_zh)
    new_item["translation_model"] = model
    return new_item


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    return load_json(path)


def render_progress(processed, total, translated_count, resumed_count, failed_count, saved_count):
    total = max(total, 1)
    percent = processed / total
    width = 30
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {processed}/{total} {percent * 100:6.2f}%"
        f" | translated={translated_count}"
        f" | resumed={resumed_count}"
        f" | failed={failed_count}"
        f" | saved={saved_count}",
        end="",
        flush=True,
    )


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_batch_with_retry(api_key, base_url, timeout, model, batch_items, max_retries, sleep_seconds):
    client = create_client(api_key, base_url, timeout)
    last_error = ""
    for attempt in range(max_retries):
        try:
            return translate_batch(client, model, batch_items)
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(max(1.0, sleep_seconds))
    raise RuntimeError(last_error)


def persist_outputs(output_json, output_jsonl, checkpoint_path, translated_records_map, checkpoint):
    translated_records = [translated_records_map[i] for i in sorted(translated_records_map.keys())]
    save_json(checkpoint_path, checkpoint)
    save_json(output_json, translated_records)
    save_jsonl(output_jsonl, translated_records)
    return len(translated_records)


def main():
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    if args.output_prefix.strip():
        prefix = args.output_prefix.strip()
        args.output_json = os.path.join(DATAFILES_DIR, f"{prefix}.json")
        args.output_jsonl = os.path.join(DATAFILES_DIR, f"{prefix}.jsonl")
        args.checkpoint = os.path.join(DATAFILES_DIR, f"{prefix}_checkpoint.json")
        args.stats = os.path.join(DATAFILES_DIR, f"{prefix}_stats.json")

    records = load_json(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint) if args.resume else {}

    translated_records_map = {}
    translated_count = 0
    resumed_count = 0
    failed_records = []
    total_records = len(records)
    pending = []
    saved_count = 0
    last_saved_translated_count = 0

    for index, item in enumerate(records):
        cache_key = f"{index}:{item.get('response_sha256','')}:{item.get('original_id','')}"
        if args.resume and cache_key in checkpoint:
            cached = checkpoint[cache_key]
            translated_records_map[index] = build_output_record(
                item,
                cached["question_zh"],
                cached["response_zh"],
                args.model,
            )
            resumed_count += 1
        else:
            pending.append(
                {
                    "index": index,
                    "item": item,
                    "cache_key": cache_key,
                    "question_en": (item.get("question") or "").strip(),
                    "response_en": (item.get("response") or "").strip(),
                }
            )

    processed = resumed_count
    render_progress(
        processed,
        total_records,
        translated_count,
        resumed_count,
        len(failed_records),
        saved_count,
    )

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for batch in batched(pending, max(1, args.batch_size)):
            batch_payload = []
            for batch_id, row in enumerate(batch):
                batch_payload.append(
                    {
                        "batch_id": batch_id,
                        "question_en": row["question_en"],
                        "response_en": row["response_en"],
                    }
                )

            future = executor.submit(
                run_batch_with_retry,
                api_key,
                args.base_url,
                args.timeout,
                args.model,
                batch_payload,
                args.max_retries,
                args.sleep_seconds,
            )
            futures[future] = batch
            time.sleep(args.sleep_seconds)

        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                translated = future.result()
                for batch_id, row in enumerate(batch):
                    if batch_id not in translated:
                        raise ValueError(f"Missing batch_id={batch_id} in translation output")
                    question_zh, response_zh = translated[batch_id]
                    checkpoint[row["cache_key"]] = {
                        "question_zh": question_zh,
                        "response_zh": response_zh,
                    }
                    translated_records_map[row["index"]] = build_output_record(
                        row["item"], question_zh, response_zh, args.model
                    )
                    translated_count += 1
                    processed += 1
            except Exception as exc:
                for row in batch:
                    failed_records.append(
                        {
                            "index": row["index"],
                            "original_id": row["item"].get("original_id"),
                            "question_en": row["question_en"],
                            "response_en": row["response_en"],
                            "error": str(exc),
                        }
                    )
                    processed += 1

            if translated_count > 0 and (translated_count - last_saved_translated_count) >= args.save_every:
                saved_count = persist_outputs(
                    args.output_json,
                    args.output_jsonl,
                    args.checkpoint,
                    translated_records_map,
                    checkpoint,
                )
                last_saved_translated_count = translated_count
                print(f"\ncheckpoint saved: translated={translated_count}, saved={saved_count}", flush=True)

            render_progress(
                processed,
                total_records,
                translated_count,
                resumed_count,
                len(failed_records),
                saved_count,
            )

    saved_count = persist_outputs(
        args.output_json,
        args.output_jsonl,
        args.checkpoint,
        translated_records_map,
        checkpoint,
    )
    translated_records = [translated_records_map[i] for i in sorted(translated_records_map.keys())]
    save_json(
        args.stats,
        {
            "input_path": args.input,
            "output_json": args.output_json,
            "output_jsonl": args.output_jsonl,
            "checkpoint": args.checkpoint,
            "model": args.model,
            "base_url": args.base_url,
            "requested_count": len(records),
            "translated_count": translated_count,
            "resumed_count": resumed_count,
            "failed_count": len(failed_records),
            "failed_records": failed_records[:100],
            "save_every": args.save_every,
            "sleep_seconds": args.sleep_seconds,
            "max_retries": args.max_retries,
            "batch_size": args.batch_size,
            "workers": args.workers,
        },
    )

    print()
    print("=" * 60)
    print("DeepSeek translation finished")
    print("=" * 60)
    print(f"output_json:     {args.output_json}")
    print(f"output_jsonl:    {args.output_jsonl}")
    print(f"checkpoint_json: {args.checkpoint}")
    print(f"stats_json:      {args.stats}")
    print(f"requested:       {len(records)}")
    print(f"translated:      {translated_count}")
    print(f"resumed:         {resumed_count}")
    print(f"failed:          {len(failed_records)}")
    print(f"saved:           {saved_count}")


if __name__ == "__main__":
    main()
