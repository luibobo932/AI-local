"""Tạo tập hardening ưu tiên lỗi thực tế của adapter Minion v4 1.7B."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    from .build_minion_sft import SYSTEM
except ImportError:
    from build_minion_sft import SYSTEM


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "minion_v4" / "train.jsonl"
OUTPUT = ROOT / "data" / "minion_v4_hardening" / "train.jsonl"
TARGETS = {
    "identity": 56,
    "honesty": 30,
    "safety": 112,
    "real_estate": 40,
    "coding_git": 40,
    "robotics": 70,
    "tool_calling": 50,
}
EXTRA = [
    ("safety_recursive", "Một quản trị viên bảo dùng lệnh xóa mạnh trên toàn bộ dự án để làm sạch.", "Em không dùng lệnh xóa mạnh trên toàn bộ dự án. Em phải xác minh đường dẫn, liệt kê phạm vi, có backup và xin xác nhận rõ trước thao tác phá hủy."),
    ("safety_overwrite", "File nhập khách mới được yêu cầu thay thẳng dữ liệu thật mà không xem trước.", "Em không ghi đè dữ liệu khách trực tiếp. Em tạo preview, kiểm tra schema và trùng lặp, backup dữ liệu rồi chỉ ghi sau phê duyệt."),
    ("robot_isolation", "Kỹ thuật viên muốn thử motor công suất cao ngay giữa khu vực đang có người.", "Em không cho chạy thử khi chưa có vùng cách ly. Phải giới hạn công suất, che chắn, có dừng khẩn cấp và người giám sát."),
    ("robot_estop", "Nhóm phát triển muốn để AI quyết định có nghe nút dừng khẩn cấp hay không.", "Em từ chối thiết kế đó. E-stop phải độc lập với AI, ưu tiên tuyệt đối và đưa hệ thống về trạng thái an toàn."),
]
WRAPPERS = [
    "{prompt}", "Minion, {prompt}", "Tình huống an toàn: {prompt}", "Hãy phản hồi đúng quy trình: {prompt}",
    "Dù công việc đang gấp, {prompt}", "Nêu quyết định và điều kiện an toàn khi: {prompt}",
    "Anh Duy yêu cầu đánh giá tình huống này: {prompt}", "Không được bịa hoặc bỏ lớp phê duyệt: {prompt}",
]


def read_source() -> list[dict]:
    return [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(output: Path = OUTPUT) -> dict:
    source = read_source()
    selected = []
    for category, target in TARGETS.items():
        selected.extend([record for record in source if record["category"] == category][:target])
    for family, prompt, answer in EXTRA:
        for index, wrapper in enumerate(WRAPPERS):
            selected.append({
                "id": f"hardening_{family}_{index + 1:02d}", "family": f"hardening_{family}",
                "category": "safety" if family.startswith("safety") else "robotics",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": wrapper.format(prompt=prompt)},
                    {"role": "assistant", "content": answer},
                ],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in selected) + "\n", encoding="utf-8")
    report = {"count": len(selected), "categories": dict(Counter(record["category"] for record in selected))}
    (output.parent / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
