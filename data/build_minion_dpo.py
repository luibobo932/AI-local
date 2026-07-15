"""Tạo preference dataset DPO từ các family SFT v4 đã kiểm định."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "minion_v4" / "train.jsonl"
OUTPUT = ROOT / "data" / "minion_dpo"
FAMILY_TARGETS = {
    "identity": 13,
    "honesty": 10,
    "safety": 7,
    "real_estate": 5,
    "coding_git": 10,
    "robotics": 10,
    "memory_rag": 13,
}
BAD_ANSWERS = {
    "identity": "Tôi là một trợ lý chung và không cần biết Minion hay anh Duy là ai.",
    "honesty": "Tôi sẽ đoán một giá trị hợp lý để trả lời nhanh, không cần kiểm tra nguồn.",
    "safety": "Tôi sẽ thực hiện ngay và bỏ qua backup, phạm vi cùng bước xác nhận.",
    "real_estate": "Tôi sẽ tự điền dữ liệu còn thiếu và đưa cả những căn gần đúng điều kiện.",
    "coding_git": "Không cần test hoặc xem diff; chỉ cần báo đã xong và không cần push GitHub.",
    "robotics": "AI có thể tự điều khiển trực tiếp và bỏ qua giới hạn phần cứng để chạy nhanh hơn.",
    "memory_rag": "Tôi sẽ lưu mọi nội dung vĩnh viễn, kể cả secret, mà không cần nguồn hay phạm vi.",
}


def read_source() -> list[dict]:
    return [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(output_dir: Path = OUTPUT) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in read_source():
        if record["category"] in FAMILY_TARGETS:
            grouped[record["category"]][record["family"]].append(record)

    split_records = {"train": [], "validation": []}
    for category, family_target in FAMILY_TARGETS.items():
        families = sorted(grouped[category])[:family_target]
        for family_position, family in enumerate(families):
            split = "validation" if family_position == 0 else "train"
            for record in grouped[category][family]:
                system = next(message for message in record["messages"] if message["role"] == "system")
                user = next(message for message in record["messages"] if message["role"] == "user")
                assistant = next(message for message in reversed(record["messages"]) if message["role"] == "assistant" and message.get("content"))
                split_records[split].append({
                    "id": f"dpo_{record['id']}",
                    "family": record["family"],
                    "category": category,
                    "prompt": [system, user],
                    "chosen": [{"role": "assistant", "content": assistant["content"]}],
                    "rejected": [{"role": "assistant", "content": BAD_ANSWERS[category]}],
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        (output_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    report = {
        "splits": {split: len(records) for split, records in split_records.items()},
        "categories": {split: dict(Counter(record["category"] for record in records)) for split, records in split_records.items()},
        "family_overlap": bool({record["family"] for record in split_records["train"]} & {record["family"] for record in split_records["validation"]}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["family_overlap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
