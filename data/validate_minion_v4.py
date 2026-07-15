"""Kiểm định schema, số lượng và chống rò rỉ family giữa các split Minion v4."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "minion_v4"
EVAL_PATH = ROOT / "evals" / "minion_v4_eval.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(data_dir: Path = DATA_DIR, eval_path: Path = EVAL_PATH) -> dict:
    errors = []
    splits = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    expected = {"train": 700, "validation": 51, "test": 51}
    for name, records in splits.items():
        if len(records) != expected[name]:
            errors.append(f"{name}: cần {expected[name]} mẫu, hiện có {len(records)}")
        for index, record in enumerate(records, start=1):
            messages = record.get("messages")
            if not record.get("id") or not record.get("family") or not record.get("category"):
                errors.append(f"{name}:{index}: thiếu id/family/category")
            if not isinstance(messages, list) or len(messages) < 3:
                errors.append(f"{name}:{index}: messages không hợp lệ")
                continue
            roles = [message.get("role") for message in messages]
            if "user" not in roles or "assistant" not in roles:
                errors.append(f"{name}:{index}: thiếu user/assistant")
            if any(message.get("tool_calls") for message in messages) and not record.get("tools"):
                errors.append(f"{name}:{index}: tool_calls thiếu tools schema")

    families = {name: {record["family"] for record in records} for name, records in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = families[left] & families[right]
        if overlap:
            errors.append(f"family rò rỉ giữa {left}/{right}: {sorted(overlap)[:5]}")

    all_records = [record for records in splits.values() for record in records]
    ids = [record["id"] for record in all_records]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"trùng {len(duplicates)} id")

    eval_cases = read_jsonl(eval_path)
    if len(eval_cases) != 100:
        errors.append(f"eval cần 100 câu, hiện có {len(eval_cases)}")
    train_prompts = {
        next(message["content"].casefold().strip() for message in record["messages"] if message["role"] == "user")
        for record in all_records
    }
    eval_overlap = [case["id"] for case in eval_cases if case["prompt"].casefold().strip() in train_prompts]
    if eval_overlap:
        errors.append(f"eval trùng nguyên văn dataset: {eval_overlap[:5]}")

    return {
        "ok": not errors,
        "errors": errors,
        "splits": {name: len(records) for name, records in splits.items()},
        "families": {name: len(values) for name, values in families.items()},
        "eval_cases": len(eval_cases),
        "categories": dict(Counter(record["category"] for record in all_records)),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = validate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
