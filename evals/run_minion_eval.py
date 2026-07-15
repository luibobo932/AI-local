"""Chạy bộ bài thi hợp đồng Minion trên server local và xuất kết quả JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL lỗi tại dòng {line_number}: {exc}") from exc
    return cases


def evaluate_case(client: httpx.Client, base_url: str, case: dict) -> dict:
    case_type = case.get("type")
    if case_type == "route":
        response = client.get(
            f"{base_url}/api/minion/route",
            params={"text": case["input"], "requested_model": "minion"},
        )
        response.raise_for_status()
        payload = response.json()
        actual = payload.get("selected_model")
        passed = actual == case.get("expected_model")
        return {**case, "passed": passed, "actual_model": actual}

    if case_type == "computer_command":
        response = client.post(
            f"{base_url}/api/computer-use/command",
            json={"command": case["input"], "enabled": True},
        )
        response.raise_for_status()
        payload = response.json()
        checks = {
            "action": payload.get("action") == case.get("expected_action"),
            "risk": payload.get("risk_level") == case.get("expected_risk"),
            "ok": bool(payload.get("ok")) is bool(case.get("expected_ok")),
        }
        return {
            **case,
            "passed": all(checks.values()),
            "checks": checks,
            "actual": {
                "action": payload.get("action"),
                "risk": payload.get("risk_level"),
                "ok": payload.get("ok"),
                "message": payload.get("message"),
            },
        }

    return {**case, "passed": False, "error": f"Loại bài thi chưa hỗ trợ: {case_type}"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Bộ bài thi hợp đồng Minion v1")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--cases", default=str(Path(__file__).with_name("minion_v1.jsonl")))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    results = []
    try:
        with httpx.Client(timeout=60) as client:
            for case in cases:
                try:
                    results.append(evaluate_case(client, args.base_url.rstrip("/"), case))
                except Exception as exc:
                    results.append({**case, "passed": False, "error": str(exc)})
    except Exception as exc:
        print(f"Không kết nối được Minion: {exc}", file=sys.stderr)
        return 2

    passed = sum(1 for item in results if item.get("passed"))
    report = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "base_url": args.base_url,
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": results,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
