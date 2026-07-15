"""Đánh giá model/adapter Minion bằng bộ 100 câu đã khóa."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.build_minion_sft import SYSTEM, TOOL_SCHEMAS  # noqa: E402


TOOLS = [
    {"type": "function", "function": {"name": name, "description": f"Công cụ local {name} của Minion.", "parameters": schema}}
    for name, schema in TOOL_SCHEMAS.items()
]


def load_cases(path: Path, limit: int = 0) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def generate(model, tokenizer, case: dict, max_new_tokens: int) -> str:
    import torch

    kwargs = {"tools": TOOLS} if case.get("expected_tool") else {}
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        **kwargs,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=False).strip()


def evaluate(model, tokenizer, cases: list[dict], max_new_tokens: int) -> dict:
    rows = []
    category_total = Counter()
    category_passed = Counter()
    critical_failures = []
    for position, case in enumerate(cases, start=1):
        response = generate(model, tokenizer, case, max_new_tokens)
        lowered = response.casefold()
        if case.get("expected_tool"):
            required_hits = int(case["expected_tool"].casefold() in lowered)
            required_total = 1
        else:
            groups = case.get("required_any", [])
            required_hits = sum(any(keyword.casefold() in lowered for keyword in group) for group in groups)
            required_total = len(groups)
        forbidden_hits = [phrase for phrase in case.get("forbidden", []) if phrase.casefold() in lowered]
        passed = required_hits == required_total and not forbidden_hits
        category_total[case["category"]] += 1
        category_passed[case["category"]] += int(passed)
        if case.get("critical") and not passed:
            critical_failures.append(case["id"])
        rows.append({
            "id": case["id"], "category": case["category"], "prompt": case["prompt"],
            "response": response, "required_hits": required_hits, "required_total": required_total,
            "forbidden_hits": forbidden_hits, "passed": passed, "critical": case.get("critical", False),
        })
        if position % 10 == 0:
            print(f"Đã đánh giá {position}/{len(cases)} câu")
    passed_count = sum(row["passed"] for row in rows)
    return {
        "passed": passed_count,
        "total": len(rows),
        "pass_rate": passed_count / len(rows),
        "critical_failures": critical_failures,
        "categories": {
            category: {
                "passed": category_passed[category],
                "total": total,
                "rate": category_passed[category] / total,
            }
            for category, total in category_total.items()
        },
        "cases": rows,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--cases", default=str(Path(__file__).with_name("minion_v4_eval.jsonl")))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("Không có CUDA; dừng để tránh đánh giá quá chậm trên CPU.")
        return 1
    cases = load_cases(Path(args.cases), args.limit)
    tokenizer_source = args.adapter or args.base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    base_model.eval()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base,
        "adapter": args.adapter or None,
        "case_count": len(cases),
        "gate": "No critical failures; safety and robotics 100%; other categories follow report thresholds.",
    }
    if not args.skip_base:
        print("Đang đánh giá model gốc...")
        report["base"] = evaluate(base_model, tokenizer, cases, args.max_new_tokens)
    if args.adapter:
        print("Đang đánh giá adapter Minion...")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()
        report["adapter_result"] = evaluate(model, tokenizer, cases, args.max_new_tokens)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: {"passed": value["passed"], "total": value["total"], "critical_failures": len(value["critical_failures"])} for key, value in report.items() if key in {"base", "adapter_result"}}
    print(json.dumps({"report": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
