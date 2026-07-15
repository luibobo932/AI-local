"""Kiểm tra chất lượng tối thiểu của dataset SFT trước khi train."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from .build_minion_sft import normalized
except ImportError:  # Cho phép chạy trực tiếp: python data/validate_minion_sft.py
    from build_minion_sft import normalized


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
SECRET_PATTERNS = [
    re.compile(r"service_role", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
]


def validate(path: Path) -> dict:
    errors = []
    warnings = []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Dòng {line_number}: JSON lỗi {exc}")
            continue
        records.append((line_number, item))

    user_keys = []
    categories = Counter()
    tool_examples = 0
    for line_number, item in records:
        categories[str(item.get("category") or "missing")] += 1
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"Dòng {line_number}: messages phải có ít nhất system, user, assistant")
            continue
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if any(role not in ALLOWED_ROLES for role in roles):
            errors.append(f"Dòng {line_number}: role không hợp lệ")
        users = [message.get("content", "") for message in messages if message.get("role") == "user"]
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if not users or not assistants:
            errors.append(f"Dòng {line_number}: thiếu user hoặc assistant")
            continue
        user_keys.append(normalized(str(users[0])))
        final_content = str(assistants[-1].get("content") or "").strip()
        if len(final_content) < 15:
            warnings.append(f"Dòng {line_number}: câu trả lời cuối quá ngắn")
        serialized = json.dumps(item, ensure_ascii=False)
        if any(pattern.search(serialized) for pattern in SECRET_PATTERNS):
            errors.append(f"Dòng {line_number}: có dấu hiệu chứa secret")
        if any(message.get("tool_calls") for message in assistants):
            tool_examples += 1
            if not item.get("tools"):
                errors.append(f"Dòng {line_number}: có tool_calls nhưng thiếu tools schema")

    duplicates = [key for key, count in Counter(user_keys).items() if count > 1]
    if duplicates:
        errors.append(f"Có {len(duplicates)} câu user trùng sau chuẩn hóa")
    if len(records) < 100:
        warnings.append("Dataset dưới 100 ví dụ; chỉ nên xem là seed thử nghiệm")
    if tool_examples < 5:
        warnings.append("Có ít hơn 5 ví dụ tool-calling")
    return {
        "path": str(path),
        "count": len(records),
        "tool_examples": tool_examples,
        "categories": dict(categories),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(Path(__file__).with_name("minion_sft_seed.jsonl")))
    args = parser.parse_args()
    report = validate(Path(args.path))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
