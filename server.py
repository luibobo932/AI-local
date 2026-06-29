"""
AI-local REST API Server — Ollama-compatible API.

Endpoints:
  GET  /api/version               — server version
  GET  /api/tags                  — list available models
  GET  /api/ps                    — list running (loaded) models
  POST /api/show                  — model info
  POST /api/copy                  — copy a model checkpoint
  DELETE /api/delete              — delete a checkpoint
  POST /api/generate              — text completion (streaming/non-streaming)
  POST /api/chat                  — chat completion (streaming/non-streaming)
  POST /api/embeddings            — generate embeddings (legacy)
  POST /api/embed                 — generate embeddings (new)
  POST /api/create                — create model from Modelfile (stub)

  POST /v1/chat/completions       — OpenAI-compatible chat
  POST /v1/completions            — OpenAI-compatible completion
  GET  /v1/models                 — OpenAI-compatible model list

Usage:
    python server.py                          # serve on localhost:11434
    python server.py --port 8080
    python server.py --model checkpoints/dpo_ckpt.pt
"""

import argparse
import asyncio
import glob
import json
import os
import pickle
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from computer_use import execute_computer_command, get_computer_state
from model.gpt import GPT, GPTConfig
from model.hf_backend import HFModel, is_hf_model
from tools import TOOL_SCHEMAS, call_tool, get_tool_schemas

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-local", description="Ollama-compatible local LLM server")

# ─── Model registry (loaded on demand, cached in memory) ─────────────────────

_model_cache: dict[str, tuple] = {}   # name → (model, meta_or_None)
_qa_cache: dict[str, str] | None = None
_data_dir = "data"
_checkpoints_dir = "checkpoints"
_default_model = ""
_ollama_base_url = os.environ.get("AI_LOCAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_supabase_env_paths = [
    r"D:\app bomtan\excel_supabase_sync\.env",
    r"D:\app bomtan\20260326\my_app-main\my_app-main\tooling\google_drive_supabase_sync\.env",
]
_ollama_aliases = {
    # Tên chính của bot local.
    "minion": "phogpt-local",
    # PhoGPT bản GGUF quantized chạy qua Ollama, nhẹ hơn bản HF fp16.
    "phogpt": "phogpt-local",
    "phogpt:q4": "phogpt-local",
}
_house_all_limit = 5000
_house_action_marker = "[[MINION_ACTIONS:"


def _read_env_file(path: str) -> dict[str, str]:
    """Đọc file .env đơn giản, không log giá trị secret."""
    values: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def _supabase_config() -> tuple[str, str]:
    """Lấy Supabase URL/key cho Minion RAG; ưu tiên env, fallback file local."""
    merged: dict[str, str] = {}
    for path in _supabase_env_paths:
        merged.update(_read_env_file(path))

    url = (
        os.environ.get("MINION_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or merged.get("SUPABASE_URL")
        or "https://xmizlnaffqbdspwlusrl.supabase.co"
    ).rstrip("/")
    key = (
        os.environ.get("MINION_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or merged.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("MINION_SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or merged.get("SUPABASE_ANON_KEY")
    )
    return url, key or ""


def _ollama_target(model_name: str) -> str | None:
    """Trả về tên model Ollama nếu model_name là alias/proxy Ollama."""
    if model_name in _ollama_aliases:
        return _ollama_aliases[model_name]
    if model_name.startswith("ollama:"):
        return model_name.removeprefix("ollama:")
    return None


def _discover_models() -> list[str]:
    """Return checkpoint names + fine-tuned models + any cached HF models."""
    paths = glob.glob(os.path.join(_checkpoints_dir, "*.pt"))
    names = list(_ollama_aliases)
    names += [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]
    # Fine-tuned models saved in models/<name>/
    if os.path.isdir("models"):
        for d in sorted(os.listdir("models")):
            if os.path.exists(os.path.join("models", d, "config.json")) and d not in names:
                names.append(d)
    # HF models currently loaded in memory
    for n in _model_cache:
        if is_hf_model(n) and n not in names:
            names.append(n)
    return names


def _resolve_checkpoint(model_name: str) -> str:
    """Map model name → checkpoint path."""
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    direct = os.path.join(_checkpoints_dir, model_name)
    if os.path.exists(direct):
        return direct
    with_ext = direct + ".pt"
    if os.path.exists(with_ext):
        return with_ext
    raise FileNotFoundError(f"Model '{model_name}' not found in {_checkpoints_dir}/")


def _load_model(model_name: str) -> tuple:
    """Load (and cache) a model — either local checkpoint or HuggingFace."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    # HuggingFace model (gpt2, distilgpt2, microsoft/phi-2, etc.)
    if is_hf_model(model_name):
        hf = HFModel(model_name)
        _model_cache[model_name] = (hf, None)
        return hf, None

    ckpt_path = _resolve_checkpoint(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg: GPTConfig = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load vocab
    meta_path = os.path.join(_data_dir, "meta.pkl")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    _model_cache[model_name] = (model, meta)
    return model, meta


def _encode(text: str, meta: dict | None, model=None) -> list[int]:
    if isinstance(model, HFModel):
        return model.encode(text)
    if meta:
        stoi = meta["stoi"]
        fallback = stoi.get(" ", 0)
        return [stoi.get(c, fallback) for c in text]
    import tiktoken
    return tiktoken.get_encoding("gpt2").encode(text)


def _decode(ids: list[int], meta: dict | None, model=None) -> str:
    if isinstance(model, HFModel):
        return model.decode(ids)
    if meta:
        itos = meta["itos"]
        return "".join(itos.get(i, "") for i in ids)
    import tiktoken
    return tiktoken.get_encoding("gpt2").decode(ids)


def _format_prompt_chat(messages: list[dict]) -> str:
    """Convert OpenAI-style messages to the Human/Assistant template."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"Human: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant: ")
    return "\n\n".join(parts)


def _normalize_question(text: str) -> str:
    """Chuẩn hóa câu hỏi để khớp Q&A ổn định hơn."""
    text = text.strip().lower()
    text = re.sub(r"[\\.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_vi_qa() -> dict[str, str]:
    """Nạp bộ Q&A tiếng Việt đã dùng để train model nhỏ."""
    global _qa_cache
    if _qa_cache is not None:
        return _qa_cache

    qa: dict[str, str] = {}
    path = os.path.join(_data_dir, "sft_vi.jsonl")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = _normalize_question(item.get("instruction", ""))
                answer = item.get("response", "").strip()
                if question and answer:
                    qa[question] = answer
    except Exception:
        qa = {}

    _qa_cache = qa
    return qa


def _known_vi_answer(model_name: str, messages: list[dict]) -> str | None:
    """Trả lời chính xác các câu có trong bộ train của chat_vi."""
    if model_name != "chat_vi":
        return None

    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    return _load_vi_qa().get(_normalize_question(last_user))


def _last_user_message(messages: list[dict]) -> str:
    """Lấy câu hỏi mới nhất của người dùng."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _minion_identity_answer(question: str) -> str | None:
    """Trả lời cố định các câu chào/hỏi danh tính để model không bịa nguồn gốc."""
    plain = _plain_text(question)
    plain = re.sub(r"[^a-z0-9\s]", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        return None

    greeting_only = plain in {"chao", "xin chao", "hello", "hi", "hey", "alo"}
    asks_identity = bool(re.search(
        r"\b("
        r"ban la ai|may la ai|em la ai|ten gi|ten cua ban|"
        r"ai tao|ai tao ra|ai phat trien|duoc ai phat trien|"
        r"nguon goc|gioi thieu ve ban|gioi thieu ban than"
        r")\b",
        plain,
    ))

    if not greeting_only and not asks_identity:
        return None

    return (
        "Chào anh Duy, em là Minion.\n\n"
        "Em là trợ lý AI local do Duy phát triển để hỗ trợ công việc nhà phố TP.HCM: "
        "tìm căn, lọc dữ liệu nhà, tóm tắt thông tin để gọi khách và giảm việc nhập liệu thủ công.\n\n"
        "Anh cần em tìm nhà theo quận, giá, ngang, diện tích hay tóm tắt căn nào thì cứ nhắn trực tiếp."
    )


def _plain_text(text: str) -> str:
    """Đưa tiếng Việt về dạng thường, không dấu để so khớp filter linh hoạt."""
    normalized = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _clean_filter_text(value: str) -> str:
    """Dọn phần text người dùng nhập cho filter đường/loại."""
    value = re.sub(r"\s+", " ", value).strip(" .,;:-")
    return value.strip()


def _parse_number(value: str) -> float:
    """Parse số người dùng nhập, hỗ trợ dấu phẩy thập phân."""
    return float(value.replace(",", "."))


def _num_text(value, suffix: str = "") -> str:
    """Format số gọn để đọc nhanh."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        text = str(int(number))
    else:
        text = f"{number:g}"
    return f"{text}{suffix}"


def _price_text(value) -> str:
    """Format giá đang lưu theo đơn vị triệu."""
    price = _to_float(value)
    if price is None:
        return "chưa có"
    if price >= 1000:
        return f"{_num_text(price / 1000)} tỷ"
    return f"{_num_text(price)} triệu"


def _wants_all_house_results(plain: str) -> bool:
    """Nhận diện khi người dùng muốn liệt kê toàn bộ kết quả đúng điều kiện."""
    return bool(re.search(r"\b(?:tat\s+ca|toan\s+bo|liet\s+ke\s+het|lay\s+het|xem\s+het|xem\s+tat\s+ca|show\s+all|all)\b", plain))


def _requested_house_limit(plain: str) -> int | None:
    """Lấy số lượng căn người dùng muốn xem, giới hạn để trả lời không quá dài."""
    if _wants_all_house_results(plain):
        return _house_all_limit
    matches = re.findall(r"(?:tim|gui|lay|goi\s+y|cho\s+toi|cho\s+anh)?\s*([0-9]{1,2})\s*(?:can|nha)\b", plain)
    if not matches:
        return None
    return max(1, min(int(matches[-1]), 20))


def _normalized_address_parts(row: dict) -> tuple[str, str, str]:
    """Chuẩn hóa lỗi tách số nhà/đường từ nguồn Landsoft."""
    address_no = str(row.get("address_no") or "").strip()
    street = str(row.get("street") or "").strip()
    district = str(row.get("district") or "").strip()

    address_no = re.sub(r"@+$", "", address_no.strip(" ,;")).strip()
    street = re.sub(r"^\*+\s*", "", street).strip()
    split_match = re.match(r"^([A-Za-z])\s+(.+)$", street)
    if split_match and address_no and re.fullmatch(r"[0-9]+", address_no.strip()):
        address_no = f"{address_no}{split_match.group(1)}"
        street = split_match.group(2).strip()

    return address_no, street, district


def _parse_house_filters(query: str) -> dict:
    """Parse vài filter thường dùng trong câu hỏi nhà phố."""
    q = query.lower()
    plain = _plain_text(q)
    filters: dict[str, float | str] = {}
    show_all = _wants_all_house_results(plain)
    if show_all:
        filters["show_all"] = "true"
    requested_limit = _requested_house_limit(plain)
    if requested_limit is None:
        filters["limit_mode"] = "unlimited"
    else:
        filters["limit"] = requested_limit

    district_match = re.search(
        r"(?:q\.?|quan)\s*([0-9]{1,2}|binh thanh|phu nhuan|tan binh|tan phu|go vap|binh tan|thu duc|nha be)",
        plain,
    )
    if district_match:
        district = district_match.group(1).strip()
        if district.isdigit():
            filters["district"] = f"Quận {district}"
        else:
            named = {
                "binh thanh": "Bình Thạnh",
                "phu nhuan": "Phú Nhuận",
                "tan binh": "Tân Bình",
                "tan phu": "Tân Phú",
                "go vap": "Gò Vấp",
                "binh tan": "Bình Tân",
                "thu duc": "Thủ Đức",
                "nha be": "Nhà Bè",
            }
            filters["district"] = named.get(district, district.title())

    ward_match = re.search(r"(?:\bp\.|\bphuong)\s*([0-9]{1,2}|[a-z0-9\s]+?)(?=\s+(?:duong|quan|q\.?|gia|duoi|tren|hon|nho|dien|dt|ngang|hem|mat\s+tien)\b|$)", plain)
    if ward_match:
        ward = _clean_filter_text(ward_match.group(1))
        if ward:
            filters["ward_plain"] = ward

    street_match = re.search(
        r"(?:duong|pho|mat\s+tien\s+duong|mt\s+duong)\s+(.+?)(?=\s+(?:duoi|tren|hon|lon\s+hon|nho\s+hon|gia|tam|khoang|quan|q\.?|phuong|p\.?|dien\s+tich|dt|ngang|chieu\s+ngang|mat\s+tien|mt|hem|dang\s+ban|da\s+ban)\b|$)",
        plain,
    )
    if street_match:
        street_plain = _clean_filter_text(street_match.group(1))
        if street_plain:
            original_match = re.search(
                r"(?:đường|duong|phố|pho|mặt\s+tiền\s+đường|mat\s+tien\s+duong|mt\s+đường|mt\s+duong)\s+(.+?)(?=\s+(?:dưới|duoi|trên|tren|hơn|hon|lớn\s+hơn|lon\s+hon|nhỏ\s+hơn|nho\s+hon|giá|gia|tầm|tam|khoảng|khoang|quận|quan|q\.?|phường|phuong|p\.?|diện\s+tích|dien\s+tich|dt|ngang|chiều\s+ngang|chieu\s+ngang|mặt\s+tiền|mat\s+tien|mt|hẻm|hem|đang\s+bán|dang\s+ban|đã\s+bán|da\s+ban)\b|$)",
                query,
                re.IGNORECASE,
            )
            street_original = _clean_filter_text(original_match.group(1)) if original_match else street_plain
            filters["street"] = street_original
            filters["street_plain"] = street_plain

    price_range = re.search(
        r"(?:gia\s*)?(?:tu\s*)?([0-9]+(?:[,.][0-9]+)?)\s*(?:ty|ti)?\s*(?:den|toi|-)\s*([0-9]+(?:[,.][0-9]+)?)\s*(?:ty|ti)\b",
        plain,
    )
    if price_range:
        filters["min_price"] = _parse_number(price_range.group(1)) * 1000
        filters["min_price_op"] = "gte"
        filters["max_price"] = _parse_number(price_range.group(2)) * 1000
        filters["max_price_op"] = "lte"

    price_above = re.search(
        r"(?:gia\s*)?(?:tren|hon|lon\s+hon|>|tu)\s*([0-9]+(?:[,.][0-9]+)?)\s*(ty|ti)",
        plain,
    )
    if price_above and "min_price" not in filters:
        filters["min_price"] = _parse_number(price_above.group(1)) * 1000
        filters["min_price_op"] = "gte" if re.search(r"(?:^|\s)tu\s", price_above.group(0)) else "gt"

    price_up = re.search(r"([0-9]+(?:[,.][0-9]+)?)\s*(ty|ti)\s*(?:tro\s+len|do\s+len)", plain)
    if price_up and "min_price" not in filters:
        filters["min_price"] = _parse_number(price_up.group(1)) * 1000
        filters["min_price_op"] = "gte"

    price_min = re.search(
        r"(?:gia\s*)?(?:tu|>=|toi\s+thieu|it\s+nhat)\s*([0-9]+(?:[,.][0-9]+)?)\s*(ty|ti)",
        plain,
    )
    if price_min and "min_price" not in filters:
        filters["min_price"] = _parse_number(price_min.group(1)) * 1000
        filters["min_price_op"] = "gte"

    price_match = re.search(r"(duoi|nho\s+hon|<|<=|toi\s+da|khong\s+qua|tam|khoang)\s*([0-9]+(?:[,.][0-9]+)?)\s*(ty|ti)", plain)
    if price_match and "max_price" not in filters:
        filters["max_price"] = _parse_number(price_match.group(2)) * 1000
        filters["max_price_op"] = "lt" if price_match.group(1) in {"duoi", "nho hon", "<"} else "lte"

    price_down = re.search(r"([0-9]+(?:[,.][0-9]+)?)\s*(ty|ti)\s*(?:tro\s+xuong|do\s+lai)", plain)
    if price_down and "max_price" not in filters:
        filters["max_price"] = _parse_number(price_down.group(1)) * 1000
        filters["max_price_op"] = "lte"

    area_range = re.search(
        r"(?:(?:dien\s+tich|dt)\s*)?(?:tu\s*)?([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2)?\s*(?:den|toi|-)\s*([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)\b",
        plain,
    )
    if area_range:
        filters["min_area"] = _parse_number(area_range.group(1))
        filters["min_area_op"] = "gte"
        filters["max_area"] = _parse_number(area_range.group(2))
        filters["max_area_op"] = "lte"

    area_above = re.search(
        r"(?:(?:dien\s+tich|dt)\s*)?(?:tren|hon|lon\s+hon|>)\s*([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)\b",
        plain,
    )
    if area_above and "min_area" not in filters:
        filters["min_area"] = _parse_number(area_above.group(1))
        filters["min_area_op"] = "gt"
    else:
        area_min = re.search(
            r"(?:dien\s+tich|dt)\s*(?:tu|>=|toi\s+thieu|it\s+nhat)?\s*([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)?",
            plain,
        )
        if area_min and "min_area" not in filters:
            filters["min_area"] = _parse_number(area_min.group(1))
            filters["min_area_op"] = "gte"

    area_up = re.search(r"([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)\s*(?:tro\s+len|do\s+len)", plain)
    if area_up and "min_area" not in filters:
        filters["min_area"] = _parse_number(area_up.group(1))
        filters["min_area_op"] = "gte"

    area_below = re.search(
        r"(?:(?:dien\s+tich|dt)\s*)?(duoi|nho\s+hon|<|<=|toi\s+da|khong\s+qua)\s*([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)\b",
        plain,
    )
    if area_below and "max_area" not in filters:
        filters["max_area"] = _parse_number(area_below.group(2))
        filters["max_area_op"] = "lt" if area_below.group(1) in {"duoi", "nho hon", "<"} else "lte"

    area_down = re.search(r"([0-9]+(?:[,.][0-9]+)?)\s*(?:m2|m\s*2|m²)\s*(?:tro\s+xuong|do\s+lai)", plain)
    if area_down and "max_area" not in filters:
        filters["max_area"] = _parse_number(area_down.group(1))
        filters["max_area_op"] = "lte"

    width_range = re.search(
        r"(?:chieu\s+ngang|mat\s+tien|ngang|rong)\s*(?:tu\s*)?([0-9]+(?:[,.][0-9]+)?)\s*m?\s*(?:den|toi|-)\s*([0-9]+(?:[,.][0-9]+)?)\s*m?\b",
        plain,
    )
    if width_range:
        filters["min_width"] = _parse_number(width_range.group(1))
        filters["min_width_op"] = "gte"
        filters["max_width"] = _parse_number(width_range.group(2))
        filters["max_width_op"] = "lte"

    width_above = re.search(
        r"(?:chieu\s+ngang|mat\s+tien|ngang|rong)\s*(?:tren|hon|lon\s+hon|>)\s*([0-9]+(?:[,.][0-9]+)?)\s*m?",
        plain,
    )
    if width_above and "min_width" not in filters:
        filters["min_width"] = _parse_number(width_above.group(1))
        filters["min_width_op"] = "gt"
    else:
        min_width = re.search(
            r"(?:chieu\s+ngang|mat\s+tien|ngang|rong)\s*(?:tu|>=|toi\s+thieu|it\s+nhat)?\s*([0-9]+(?:[,.][0-9]+)?)\s*m?",
            plain,
        )
        if min_width and "min_width" not in filters:
            filters["min_width"] = _parse_number(min_width.group(1))
            filters["min_width_op"] = "gte"

    width_below = re.search(
        r"(?:chieu\s+ngang|mat\s+tien|ngang|rong)\s*(duoi|nho\s+hon|<|<=|toi\s+da|khong\s+qua)\s*([0-9]+(?:[,.][0-9]+)?)\s*m?",
        plain,
    )
    if width_below and "max_width" not in filters:
        filters["max_width"] = _parse_number(width_below.group(2))
        filters["max_width_op"] = "lt" if width_below.group(1) in {"duoi", "nho hon", "<"} else "lte"

    direction_match = re.search(r"(?:huong|hướng)\s+(dong\s+nam|dong\s+bac|tay\s+nam|tay\s+bac|dong|tay|nam|bac)", plain)
    if direction_match:
        directions = {
            "dong": "Đông",
            "tay": "Tây",
            "nam": "Nam",
            "bac": "Bắc",
            "dong nam": "Đông Nam",
            "dong bac": "Đông Bắc",
            "tay nam": "Tây Nam",
            "tay bac": "Tây Bắc",
        }
        filters["direction"] = directions.get(direction_match.group(1), direction_match.group(1).title())

    if "hem xe hoi" in plain or "hem oto" in plain or "hem o to" in plain or "hxh" in plain or "oto vao" in plain:
        filters["alley_vehicle_type"] = "Xe hơi"
    elif "hem ba gac" in plain or "hem 3 gac" in plain:
        filters["alley_vehicle_type"] = "Ba gác"
    elif "hem xe may" in plain:
        filters["alley_vehicle_type"] = "Xe máy"

    if "mat tien" in plain or re.search(r"\bmt\b", plain):
        filters["house_type"] = "Mặt tiền"
    elif "nha hem" in plain or "hem " in plain:
        filters["house_type"] = "Nhà Hẻm"

    if "so hong" in plain:
        filters["legal_status_plain"] = "so hong"
    elif "the chap" in plain:
        filters["legal_status_plain"] = "the chap"
    elif "quy hoach" in plain or "lo gioi" in plain:
        filters["legal_status_plain"] = "quy hoach lo gioi"

    if "da ban" in plain or "ban roi" in plain:
        filters["status"] = "sold"
    elif "dang ban" in plain or "con ban" in plain:
        filters["status"] = "selling"

    if "gia thap" in plain or "re nhat" in plain or "gia tang" in plain:
        filters["sort_by"] = "price"
        filters["sort_desc"] = "false"
    elif "gia cao" in plain or "gia giam" in plain:
        filters["sort_by"] = "price"
        filters["sort_desc"] = "true"
    elif "dien tich lon" in plain or "dt lon" in plain:
        filters["sort_by"] = "area"
        filters["sort_desc"] = "true"
    elif "dien tich nho" in plain or "dt nho" in plain:
        filters["sort_by"] = "area"
        filters["sort_desc"] = "false"

    return filters


def _house_search_terms(query: str) -> list[str]:
    """Tách token tiếng Việt đủ dài để chấm điểm kết quả Supabase."""
    plain = _plain_text(query)
    stop = {
        "cho", "toi", "can", "tim", "nha", "pho", "quan", "duoi", "tren",
        "hon", "lon", "nho", "ty", "ti", "co", "nao", "ban", "minion",
        "tin", "uu", "tien", "thong", "duong", "gia", "dien", "tich",
        "ngang", "chieu", "hem", "mat", "mt", "m2",
    }
    tokens = re.findall(r"[\w]+", plain)
    return [t for t in tokens if len(t) >= 3 and t not in stop][:12]


def _to_float(value) -> float | None:
    """Ép số an toàn từ dữ liệu Supabase."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_dimension(row: dict, key: str) -> float | None:
    """Chuẩn hóa ngang/dài; sửa các dòng bị mất dấu thập phân như 305 -> 3.05."""
    value = _to_float(row.get(key))
    if value is None:
        return None

    other_key = "length" if key == "width" else "width"
    other = _to_float(row.get(other_key))
    area = _to_float(row.get("area"))

    candidates = [value]
    if key == "width":
        if value > 50:
            candidates.append(value / 100)
        if value > 20:
            candidates.append(value / 10)
        plausible = [candidate for candidate in candidates if 1 <= candidate <= 20]
    else:
        if value > 100:
            candidates.append(value / 10)
            candidates.append(value / 100)
        plausible = [candidate for candidate in candidates if 1 <= candidate <= 100]

    if plausible:
        candidates = plausible

    if area and other and other > 0:
        def error(candidate: float) -> float:
            return abs((candidate * other) - area)

        return min(candidates, key=error)

    return candidates[-1] if candidates else value


def _house_identity_key(row: dict) -> tuple[str, str, str]:
    """Khóa chống trùng theo địa chỉ; gom các biến thể như 10B / 10 B / 10b *."""
    address_raw, street_raw, district_raw = _normalized_address_parts(row)
    district = _plain_text(district_raw).strip()
    street = _plain_text(street_raw).strip()
    address_no = _plain_text(address_raw)
    address_no = re.sub(r"[^a-z0-9/]+", "", address_no)
    if not address_no:
        address_no = str(row.get("id") or "")
    return district, street, address_no


def _dedupe_house_rows(rows: list[dict]) -> list[dict]:
    """Loại dòng nhà trùng trước khi trả lời cho người dùng."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = _house_identity_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _filter_value_text(value, unit: str = "", price: bool = False) -> str:
    if price:
        return _price_text(value)
    return _num_text(value, unit)


def _range_summary(filters: dict, key: str, label: str, unit: str = "", price: bool = False) -> str | None:
    min_value = filters.get(f"min_{key}")
    max_value = filters.get(f"max_{key}")
    min_op = ">" if filters.get(f"min_{key}_op") == "gt" else ">="
    max_op = "<" if filters.get(f"max_{key}_op") == "lt" else "<="

    if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
        return f"{label}: {min_op} {_filter_value_text(min_value, unit, price)} và {max_op} {_filter_value_text(max_value, unit, price)}"
    if isinstance(min_value, (int, float)):
        return f"{label}: {min_op} {_filter_value_text(min_value, unit, price)}"
    if isinstance(max_value, (int, float)):
        return f"{label}: {max_op} {_filter_value_text(max_value, unit, price)}"
    return None


def _format_filter_summary(filters: dict) -> list[str]:
    """Hiển thị các điều kiện Minion đã hiểu để dễ bắt lỗi giao tiếp."""
    items: list[str] = []
    status = str(filters.get("status") or "selling")
    items.append(f"Trạng thái: {'đang bán' if status == 'selling' else status}")
    if filters.get("house_type"):
        items.append(f"Loại: {filters['house_type']}")
    if filters.get("alley_vehicle_type"):
        items.append(f"Hẻm: {filters['alley_vehicle_type']}")
    if filters.get("street"):
        items.append(f"Đường: {filters['street']}")
    if filters.get("district"):
        items.append(f"Quận: {filters['district']}")
    if filters.get("ward_plain"):
        items.append(f"Phường: {filters['ward_plain']} (dữ liệu hiện chưa có cột phường để lọc cứng)")
    if filters.get("direction"):
        items.append(f"Hướng: {filters['direction']}")
    if filters.get("legal_status_plain"):
        items.append(f"Pháp lý: {filters['legal_status_plain']}")

    for summary in (
        _range_summary(filters, "price", "Giá", price=True),
        _range_summary(filters, "area", "Diện tích", "m2"),
        _range_summary(filters, "width", "Ngang", "m"),
    ):
        if summary:
            items.append(summary)

    sort_by = filters.get("sort_by")
    if sort_by == "price":
        items.append(f"Sắp xếp: giá {'cao trước' if filters.get('sort_desc') == 'true' else 'thấp trước'}")
    elif sort_by == "area":
        items.append(f"Sắp xếp: diện tích {'lớn trước' if filters.get('sort_desc') == 'true' else 'nhỏ trước'}")

    limit = filters.get("limit")
    if filters.get("show_all") == "true" or filters.get("limit_mode") == "unlimited":
        items.append("Số lượng muốn xem: không giới hạn")
    elif isinstance(limit, int):
        items.append(f"Số lượng muốn xem: {limit} căn")
    return items


def _format_house_for_rag(house: dict, index: int | None = None) -> str:
    """Format một căn nhà theo mẫu đọc nhanh cho môi giới."""
    def val(key: str, default: str = "") -> str:
        raw = house.get(key)
        return default if raw is None else str(raw).strip()

    address_no, street, district = _normalized_address_parts(house)
    address = " ".join(part for part in [address_no, street] if part).strip()
    if district:
        address = f"{address}, {district}" if address else district

    width = _num_text(_normalized_dimension(house, "width"))
    length = _num_text(_normalized_dimension(house, "length"))
    area = _num_text(house.get("area"))
    if width and length:
        size_text = f"{width} x {length}m"
        if area:
            size_text += f" ({area}m2)"
    elif area:
        size_text = f"{area}m2"
    else:
        size_text = "chưa có"

    prefix = f"{index}. " if index is not None else "- "
    lines = [
        f"{prefix}{address}",
        f"   DT: {size_text}",
        f"   Giá: {_price_text(house.get('price'))}",
    ]
    if val("type"):
        lines.append(f"   Loại: {val('type')}")
    if val("direction"):
        lines.append(f"   Hướng: {val('direction')}")
    if val("legal_status"):
        lines.append(f"   Pháp lý: {val('legal_status')}")

    profile = house.get("alley_profile")
    if isinstance(profile, dict):
        vehicle_type = str(profile.get("alley_vehicle_type") or "").strip()
        if vehicle_type:
            alley_width = _num_text(profile.get("alley_width_m"), "m")
            alley_text = vehicle_type if not alley_width else f"{vehicle_type}, rộng {alley_width}"
            lines.append(f"   Hẻm: {alley_text}")
    return "\n".join(lines)


def _house_actions_marker(query: str, total: int, displayed: int) -> str:
    """Gắn action ẩn để UI render nút chọn xem thêm/xem tất cả."""
    payload = {
        "type": "house_result_actions",
        "total": total,
        "displayed": displayed,
        "actions": [
            {
                "label": f"Xem tất cả {total} căn",
                "message": f"{query.strip()} liệt kê tất cả",
            },
            {
                "label": "Chỉ xem 10 căn đầu",
                "message": f"{query.strip()} lấy 10 căn",
            },
        ],
    }
    return f"\n\n{_house_action_marker}{json.dumps(payload, ensure_ascii=False)}]]"


async def _fetch_house_rag_context(query: str, limit: int = 5) -> str:
    """Lấy dữ liệu houses từ Supabase read-only để làm RAG cho Minion."""
    supabase_url, supabase_key = _supabase_config()
    if not supabase_url or not supabase_key:
        return ""

    filters = _parse_house_filters(query)
    requested_limit = filters.get("limit")
    explicit_limit = int(requested_limit) if isinstance(requested_limit, int) else None
    select_clause = (
        "id,title,price,area,width,length,floors,district,street,address_no,note,type,direction,"
        "status,uploaded_at,updated_at,rental_price,legal_status,"
        "alley_profile:alley_profiles(alley_vehicle_type,alley_width_m)"
    )
    if "alley_vehicle_type" in filters:
        select_clause = (
            "id,title,price,area,width,length,floors,district,street,address_no,note,type,direction,"
            "status,uploaded_at,updated_at,rental_price,legal_status,"
            "alley_profile:alley_profiles!inner(alley_vehicle_type,alley_width_m)"
        )

    sort_column = str(filters.get("sort_by") or "updated_at")
    sort_desc = str(filters.get("sort_desc") or "true") == "true"
    if sort_column not in {"updated_at", "uploaded_at", "price", "area"}:
        sort_column = "updated_at"
    order_direction = "desc" if sort_desc else "asc"
    status_filter = str(filters.get("status") or "selling")
    params: list[tuple[str, str]] = [
        ("select", select_clause),
        ("status", f"eq.{status_filter}"),
        ("order", f"{sort_column}.{order_direction}.nullslast,updated_at.desc.nullslast"),
        ("limit", str(_house_all_limit)),
    ]
    if "district" in filters:
        params.append(("district", f"ilike.*{filters['district']}*"))
    if "street" in filters:
        params.append(("street", f"ilike.*{filters['street']}*"))
    if "house_type" in filters:
        params.append(("type", f"ilike.*{filters['house_type']}*"))
    if "min_price" in filters:
        min_price_op = filters.get("min_price_op", "gte")
        params.append(("price", f"{min_price_op}.{filters['min_price']}"))
    if "max_price" in filters:
        max_price_op = filters.get("max_price_op", "lte")
        params.append(("price", f"{max_price_op}.{filters['max_price']}"))
    if "min_area" in filters:
        min_area_op = filters.get("min_area_op", "gte")
        params.append(("area", f"{min_area_op}.{filters['min_area']}"))
    if "max_area" in filters:
        max_area_op = filters.get("max_area_op", "lte")
        params.append(("area", f"{max_area_op}.{filters['max_area']}"))
    if "min_width" in filters:
        min_width_op = filters.get("min_width_op", "gte")
        params.append(("width", f"{min_width_op}.{filters['min_width']}"))
    if "max_width" in filters:
        max_width_op = filters.get("max_width_op", "lte")
        params.append(("width", f"{max_width_op}.{filters['max_width']}"))
    if "alley_vehicle_type" in filters:
        params.append(("alley_profiles.alley_vehicle_type", f"eq.{filters['alley_vehicle_type']}"))

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{supabase_url}/rest/v1/houses", params=params, headers=headers)
            resp.raise_for_status()
            rows = resp.json()
    except Exception as exc:
        return f"Dữ liệu nhà Supabase hiện chưa truy cập được: {exc}"

    def row_matches_filters(row: dict) -> bool:
        district = filters.get("district")
        if isinstance(district, str) and district.lower() not in str(row.get("district") or "").lower():
            return False
        street_plain = filters.get("street_plain")
        if isinstance(street_plain, str) and street_plain:
            _, normalized_street, _ = _normalized_address_parts(row)
            row_street_plain = _plain_text(normalized_street)
            if street_plain not in row_street_plain:
                return False
        ward_plain = filters.get("ward_plain")
        if isinstance(ward_plain, str) and ward_plain and row.get("ward"):
            row_ward_plain = _plain_text(str(row.get("ward") or ""))
            if ward_plain not in row_ward_plain:
                return False
        status = filters.get("status", "selling")
        if isinstance(status, str) and status and str(row.get("status") or "").lower() != status.lower():
            return False
        house_type = filters.get("house_type")
        if isinstance(house_type, str) and house_type.lower() not in str(row.get("type") or "").lower():
            return False
        direction = filters.get("direction")
        if isinstance(direction, str) and direction and _plain_text(direction) not in _plain_text(str(row.get("direction") or "")):
            return False
        legal_status_plain = filters.get("legal_status_plain")
        if isinstance(legal_status_plain, str) and legal_status_plain:
            row_legal = _plain_text(str(row.get("legal_status") or ""))
            if legal_status_plain not in row_legal:
                return False
        min_price = filters.get("min_price")
        if isinstance(min_price, (int, float)):
            try:
                row_price = float(row.get("price") or 0)
                if filters.get("min_price_op") == "gt":
                    if row_price <= float(min_price):
                        return False
                elif row_price < float(min_price):
                    return False
            except (TypeError, ValueError):
                return False
        max_price = filters.get("max_price")
        if isinstance(max_price, (int, float)):
            try:
                row_price = float(row.get("price") or 0)
                if filters.get("max_price_op") == "lt":
                    if row_price >= float(max_price):
                        return False
                elif row_price > float(max_price):
                    return False
            except (TypeError, ValueError):
                return False
        min_area = filters.get("min_area")
        if isinstance(min_area, (int, float)):
            try:
                row_area = float(row.get("area") or 0)
                if filters.get("min_area_op") == "gt":
                    if row_area <= float(min_area):
                        return False
                elif row_area < float(min_area):
                    return False
            except (TypeError, ValueError):
                return False
        max_area = filters.get("max_area")
        if isinstance(max_area, (int, float)):
            try:
                row_area = float(row.get("area") or 0)
                if filters.get("max_area_op") == "lt":
                    if row_area >= float(max_area):
                        return False
                elif row_area > float(max_area):
                    return False
            except (TypeError, ValueError):
                return False
        min_width = filters.get("min_width")
        if isinstance(min_width, (int, float)):
            try:
                row_width = _normalized_dimension(row, "width") or 0
                if filters.get("min_width_op") == "gt":
                    if row_width <= float(min_width):
                        return False
                elif row_width < float(min_width):
                    return False
            except (TypeError, ValueError):
                return False
        max_width = filters.get("max_width")
        if isinstance(max_width, (int, float)):
            try:
                row_width = _normalized_dimension(row, "width") or 0
                if filters.get("max_width_op") == "lt":
                    if row_width >= float(max_width):
                        return False
                elif row_width > float(max_width):
                    return False
            except (TypeError, ValueError):
                return False
        alley_vehicle_type = filters.get("alley_vehicle_type")
        if isinstance(alley_vehicle_type, str):
            profile = row.get("alley_profile")
            profile_type = ""
            if isinstance(profile, dict):
                profile_type = str(profile.get("alley_vehicle_type") or "")
            row_text = " ".join(str(row.get(k) or "") for k in ("title", "type", "note")).lower()
            if profile_type:
                if profile_type.strip().lower() != alley_vehicle_type.lower():
                    return False
            elif alley_vehicle_type == "Xe hơi" and not (
                "hẻm xe hơi" in row_text or "hem xe hoi" in row_text or "xe hơi" in row_text
            ):
                return False
        return True

    rows = _dedupe_house_rows([row for row in rows if row_matches_filters(row)])
    terms = _house_search_terms(query)

    def score(row: dict) -> float:
        text = " ".join(str(row.get(k) or "") for k in (
            "title", "district", "street", "address_no", "note", "type", "legal_status"
        )).lower()
        value = 0.0
        street_plain = filters.get("street_plain")
        _, normalized_street, _ = _normalized_address_parts(row)
        if isinstance(street_plain, str) and street_plain in _plain_text(normalized_street):
            value += 10.0
        for term in terms:
            if term in _plain_text(text):
                value += 2.0
        if row.get("status") == "selling":
            value += 0.5
        if row.get("note"):
            value += 0.1
        return value

    total = len(rows)
    if filters.get("sort_by") in {"price", "area"}:
        sort_key = str(filters["sort_by"])
        ranked_all = sorted(
            rows,
            key=lambda row: _to_float(row.get(sort_key)) if _to_float(row.get(sort_key)) is not None else -1,
            reverse=str(filters.get("sort_desc") or "true") == "true",
        )
    else:
        ranked_all = sorted(rows, key=score, reverse=True)

    if filters.get("show_all") == "true":
        ranked = ranked_all
    elif explicit_limit is not None:
        ranked = ranked_all[:explicit_limit]
    elif total > 10:
        ranked = ranked_all[:10]
    else:
        ranked = ranked_all
    if not ranked:
        summary = "\n".join(f"- {item}" for item in _format_filter_summary(filters))
        return (
            "Không tìm thấy căn nhà phù hợp trong Supabase theo điều kiện hiện tại.\n\n"
            "Bộ lọc Minion đã hiểu:\n"
            f"{summary}\n\n"
            "Minion không lấy căn sai điều kiện để bù kết quả."
        )

    summary = "\n".join(f"- {item}" for item in _format_filter_summary(filters))
    if len(ranked) == total:
        headline = f"Minion tìm được {len(ranked)} căn:"
    elif filters.get("limit_mode") == "unlimited":
        headline = f"Minion đang hiển thị {len(ranked)} căn đầu, trong {total} căn đúng điều kiện:"
    else:
        headline = f"Minion tìm được {len(ranked)} căn, trong {total} căn đúng điều kiện:"
    blocks = [_format_house_for_rag(row, index=i) for i, row in enumerate(ranked, start=1)]
    lines = [
        headline,
        "",
        "Bộ lọc Minion đã hiểu:",
        summary,
        "",
        "Kết quả:",
        "",
        "\n\n".join(blocks),
        "",
        "Ghi chú: Minion chỉ dùng dữ liệu read-only từ public.houses và không bù bằng căn sai điều kiện.",
    ]
    if total > 10 and filters.get("show_all") != "true":
        lines.extend([
            "",
            f"Tổng cộng có {total} căn đúng điều kiện. Anh có thể bấm nút bên dưới để Minion liệt kê toàn bộ.",
        ])
        return "\n".join(lines) + _house_actions_marker(query, total, len(ranked))
    return "\n".join(lines)


def _is_house_query(question: str) -> bool:
    """Nhận diện câu hỏi cần dùng dữ liệu nhà Supabase."""
    q = question.lower()
    plain = _plain_text(q)
    house_keywords = (
        "nhà", "nha", "căn", "can", "bđs", "bds", "bất động sản", "bat dong san",
        "quận", "quan", "đường", "duong", "hẻm", "hem", "giá", "gia", "tỷ", "ty",
        "diện tích", "dien tich", "ngang", "chủ", "chu", "m2", "mt", "hxh",
        "phường", "phuong", "hướng", "huong", "sổ hồng", "so hong",
    )
    return any(keyword in q or keyword in plain for keyword in house_keywords)


async def _direct_house_rag_answer(question: str) -> str:
    """Trả lời trực tiếp từ RAG để tránh PhoGPT lặp prompt với context dài."""
    context = await _fetch_house_rag_context(question)
    if not context:
        return "Minion chưa kết nối được dữ liệu nhà Supabase."
    if context.startswith("Dữ liệu nhà Supabase hiện chưa truy cập được"):
        return context
    return context


async def _stream_text_answer(model_name: str, answer: str) -> StreamingResponse:
    """Stream một câu trả lời text như Ollama NDJSON."""
    async def stream() -> AsyncIterator[str]:
        created_at = datetime.now(tz=timezone.utc).isoformat()
        for char in answer:
            yield json.dumps({
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": char},
                "done": False,
            }, ensure_ascii=False) + "\n"
            await asyncio.sleep(0)
        yield json.dumps({
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "rag",
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


async def _augment_messages_with_house_rag(alias_name: str, messages: list[dict]) -> list[dict]:
    """Thêm context Supabase houses vào prompt cho Minion/PhoGPT."""
    if alias_name not in {"minion", "phogpt", "phogpt:q4"}:
        return messages

    question = _last_user_message(messages)
    if not question.strip():
        return messages

    if not _is_house_query(question):
        return messages

    context = await _fetch_house_rag_context(question)
    if not context:
        return messages

    augmented: list[dict] = []
    replaced_last_user = False
    for msg in reversed(messages):
        if not replaced_last_user and msg.get("role") == "user":
            enriched = (
                f"{context}\n\n"
                "Yêu cầu của người dùng:\n"
                f"{msg.get('content')}\n\n"
                "Hãy trả lời dựa trên danh sách nhà trên. Nêu rõ ID, đường, quận, giá, diện tích/ngang nếu có. "
                "Nếu danh sách không đủ điều kiện thì nói rõ chưa đủ dữ liệu."
            )
            augmented.append({**msg, "content": enriched})
            replaced_last_user = True
        else:
            augmented.append(msg)
    augmented.reverse()

    return [{
        "role": "system",
        "content": (
            "Bạn là Minion, trợ lý AI local cho môi giới nhà phố TP.HCM. "
            "Khi trả lời về nhà, chỉ dùng dữ liệu Supabase được cung cấp trong câu hỏi, "
            "không bịa căn không có trong dữ liệu. Nếu có nhiều ý, trình bày gọn theo từng dòng dễ đọc."
        ),
    }] + [m for m in augmented if m.get("role") != "system"]


async def _ollama_chat_proxy(req: "ChatRequest", alias_name: str, target_name: str):
    """Chuyển request chat sang Ollama để chạy model mạnh hơn như PhoGPT."""
    payload = req.model_dump()
    payload["model"] = target_name
    raw_messages = payload.get("messages") or []
    question = _last_user_message(raw_messages)
    computer_use_enabled = bool(req.options.get("computer_use_enabled"))
    if isinstance(payload.get("options"), dict):
        payload["options"].pop("computer_use_enabled", None)

    if alias_name == "minion":
        computer_result = execute_computer_command(question, computer_use_enabled)
        if computer_result is not None:
            answer = computer_result.message
            if computer_result.data:
                answer += "\n[[MINION_COMPUTER:" + json.dumps(computer_result.data, ensure_ascii=False) + "]]"
            if req.stream:
                return await _stream_text_answer(alias_name, answer)
            return {
                "model": alias_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": answer},
                "done": True,
                "done_reason": f"computer_use_{computer_result.action or 'handled'}",
            }

    if alias_name == "minion":
        identity_answer = _minion_identity_answer(question)
        if identity_answer is not None:
            if req.stream:
                return await _stream_text_answer(alias_name, identity_answer)
            return {
                "model": alias_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": identity_answer},
                "done": True,
                "done_reason": "identity",
            }

    if alias_name == "minion" and _is_house_query(question):
        answer = await _direct_house_rag_answer(question)
        if req.stream:
            return await _stream_text_answer(alias_name, answer)
        return {
            "model": alias_name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": answer},
            "done": True,
            "done_reason": "rag",
        }

    messages = await _augment_messages_with_house_rag(alias_name, raw_messages)
    if target_name == "phogpt-local" and not any(m.get("role") == "system" for m in messages):
        messages = [{
            "role": "system",
            "content": (
                "Bạn là Minion, trợ lý AI tiếng Việt chạy local cho môi giới nhà phố tại TP.HCM. "
                "Trả lời thực dụng, ngắn gọn, có cấu trúc rõ khi nhiều ý. "
                "Không bịa dữ liệu nhà, tiểu sử cá nhân, hoặc thông tin bạn không chắc. "
                "Nếu thiếu dữ liệu thì nói thẳng là chưa có trong dữ liệu hiện tại."
            ),
        }] + messages
        payload["messages"] = messages

    if req.stream:
        async def stream_ollama() -> AsyncIterator[str]:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{_ollama_base_url}/api/chat",
                        json=payload,
                    ) as resp:
                        if resp.status_code >= 400:
                            detail = await resp.aread()
                            text = detail.decode("utf-8", errors="replace")
                            yield json.dumps({
                                "model": alias_name,
                                "message": {
                                    "role": "assistant",
                                    "content": f"Lỗi Ollama: {text}",
                                },
                                "done": True,
                            }, ensure_ascii=False) + "\n"
                            return
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                data["model"] = alias_name
                                yield json.dumps(data, ensure_ascii=False) + "\n"
                            except json.JSONDecodeError:
                                continue
            except httpx.RequestError as e:
                yield json.dumps({
                    "model": alias_name,
                    "message": {
                        "role": "assistant",
                        "content": f"Không kết nối được Ollama tại {_ollama_base_url}: {e}",
                    },
                    "done": True,
                }, ensure_ascii=False) + "\n"

        return StreamingResponse(stream_ollama(), media_type="application/x-ndjson")

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{_ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            data["model"] = alias_name
            return data
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(503, f"Không kết nối được Ollama tại {_ollama_base_url}: {e}")


def _ckpt_info(name: str) -> dict:
    ollama_target = _ollama_target(name)
    if ollama_target:
        return {
            "name": name,
            "model": name,
            "modified_at": datetime.now(tz=timezone.utc).isoformat(),
            "size": 0,
            "details": {
                "stage": "ollama",
                "backend": "ollama",
                "target": ollama_target,
                "parameters": "4B",
                "quantization_level": "Q4_K_M",
            },
        }

    # HuggingFace model already loaded in cache
    if is_hf_model(name) and name in _model_cache:
        hf: HFModel = _model_cache[name][0]
        cfg = hf.cfg
        params = sum(p.numel() for p in hf.parameters())
        return {
            "name": name,
            "model": name,
            "modified_at": datetime.now(tz=timezone.utc).isoformat(),
            "size": params * 4,
            "details": {
                "stage": "hf",
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "parameters": f"{params / 1e6:.2f}M",
                "source": "huggingface",
            },
        }
    try:
        path = _resolve_checkpoint(name)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg: GPTConfig = ckpt["model_cfg"]
        size_bytes = os.path.getsize(path)
        return {
            "name": name,
            "model": name,
            "modified_at": datetime.fromtimestamp(
                os.path.getmtime(path), tz=timezone.utc
            ).isoformat(),
            "size": size_bytes,
            "details": {
                "stage": ckpt.get("stage", "pretrain"),
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "parameters": f"{sum(p.numel() for p in GPT(cfg).parameters()) / 1e6:.2f}M",
                "iter_num": ckpt.get("iter_num", 0),
                "best_val_loss": round(ckpt.get("best_val_loss", 0.0), 4),
            },
        }
    except Exception:
        return {"name": name, "model": name}


# ─── Pydantic models ─────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = True
    temperature: float = 0.8
    top_k: int = 50
    max_tokens: int = 256
    options: dict = Field(default_factory=dict)
    stop: list[str] = Field(default_factory=list)
    context: list[int] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = []
    stream: bool = True
    temperature: float = 0.8
    top_k: int = 50
    max_tokens: int = 256
    options: dict = Field(default_factory=dict)
    stop: list[str] = Field(default_factory=list)


class ShowRequest(BaseModel):
    model: str = ""
    name: str = ""


class DeleteRequest(BaseModel):
    model: str = ""
    name: str = ""


class CopyRequest(BaseModel):
    source: str
    destination: str


class EmbeddingsRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    input: str | list[str] = ""
    options: dict = Field(default_factory=dict)


class CreateRequest(BaseModel):
    model: str = ""
    name: str = ""
    modelfile: str = ""
    stream: bool = False


class ComputerCommandRequest(BaseModel):
    command: str = ""
    enabled: bool = True


# ─── Ollama-compatible endpoints ──────────────────────────────────────────────

@app.get("/api/version")
async def api_version():
    """Ollama-compatible version endpoint."""
    return {"version": "0.1.0"}


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/ps")
async def api_ps():
    """List models currently loaded in memory."""
    result = []
    for name, (model, _) in _model_cache.items():
        try:
            info = _ckpt_info(name)
            params = sum(p.numel() for p in model.parameters())
            result.append({
                **info,
                "size_vram": params * 4,  # float32 estimate
                "expires_at": None,
            })
        except Exception:
            pass
    return {"models": result}


@app.get("/api/tags")
async def list_models():
    """List available model checkpoints."""
    models = _discover_models()
    return {"models": [_ckpt_info(m) for m in models]}


@app.post("/api/show")
async def show_model(req: ShowRequest):
    """Show detailed model info."""
    name = req.model or req.name or _default_model
    if not name:
        raise HTTPException(400, "model name required")
    try:
        info = _ckpt_info(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return info


@app.delete("/api/delete")
async def delete_model(req: DeleteRequest):
    name = req.model or req.name
    if not name:
        raise HTTPException(400, "model name required")
    try:
        path = _resolve_checkpoint(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    os.remove(path)
    _model_cache.pop(name, None)
    return {"status": "success"}


@app.post("/api/copy")
async def copy_model(req: CopyRequest):
    """Copy a model checkpoint to a new name."""
    import shutil
    try:
        src = _resolve_checkpoint(req.source)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    dst_name = req.destination if req.destination.endswith(".pt") else req.destination + ".pt"
    dst = os.path.join(_checkpoints_dir, os.path.basename(dst_name))
    if os.path.exists(dst):
        raise HTTPException(400, f"Destination '{req.destination}' already exists")
    shutil.copy2(src, dst)
    return {"status": "success"}


async def _status_stream(messages: list[str]) -> AsyncIterator[str]:
    for msg in messages:
        yield json.dumps({"status": msg}) + "\n"
        await asyncio.sleep(0)


@app.post("/api/create")
async def create_model(req: CreateRequest):
    """Stub — AI-local doesn't support Modelfile syntax yet."""
    name = req.model or req.name
    return StreamingResponse(
        _status_stream([
            f"Creating model '{name}'...",
            "Note: Modelfile support is not implemented in AI-local.",
            "Use 'python cli.py pull <stage>' to train a model.",
        ]),
        media_type="application/x-ndjson",
    )


async def _stream_generate(
    model_name: str,
    prompt: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    stop: list[str] | None = None,
) -> AsyncIterator[str]:
    """Async generator that yields Ollama-style NDJSON chunks."""
    t_start = time.time()
    model, meta = _load_model(model_name)
    t_loaded = time.time()
    device = next(model.parameters()).device

    tokens = _encode(prompt, meta, model)
    prompt_token_count = len(tokens)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    created_at = datetime.now(tz=timezone.utc).isoformat()
    stop_seqs = list(stop or []) + ["■"]   # ■ = EOS, model tự học khi nào dừng
    generated_text = ""
    eval_count = 0

    t_gen_start = time.time()
    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta, model)
        # Không phát ký tự EOS ra client
        if "■" in char:
            break
        generated_text += char
        eval_count += 1
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "response": char,
            "done": False,
        }
        yield json.dumps(chunk, ensure_ascii=False) + "\n"
        await asyncio.sleep(0)

        if stop_seqs and any(s in generated_text for s in stop_seqs):
            break

    t_end = time.time()
    done_chunk = {
        "model": model_name,
        "created_at": created_at,
        "response": "",
        "done": True,
        "done_reason": "stop",
        "context": tokens[-20:],  # last 20 prompt tokens for context reuse
        "total_duration": int((t_end - t_start) * 1e9),
        "load_duration": int((t_loaded - t_start) * 1e9),
        "prompt_eval_count": prompt_token_count,
        "prompt_eval_duration": int((t_gen_start - t_loaded) * 1e9),
        "eval_count": eval_count,
        "eval_duration": int((t_end - t_gen_start) * 1e9),
    }
    yield json.dumps(done_chunk) + "\n"


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Ollama /api/generate — text completion."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found. Train one first.")
        model_name = models[-1]

    temperature = req.options.get("temperature", req.temperature)
    top_k = req.options.get("top_k", req.top_k)
    max_tokens = req.options.get("num_predict", req.max_tokens)

    stop = req.stop or req.options.get("stop", [])

    try:
        if req.stream:
            return StreamingResponse(
                _stream_generate(model_name, req.prompt, temperature, top_k, max_tokens, stop),
                media_type="application/x-ndjson",
            )
        else:
            full = ""
            done_data = {}
            async for chunk in _stream_generate(model_name, req.prompt, temperature, top_k, max_tokens, stop):
                data = json.loads(chunk)
                full += data.get("response", "")
                if data.get("done"):
                    done_data = data
            return {
                "model": model_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "response": full,
                "done": True,
                "done_reason": "stop",
                "context": done_data.get("context", []),
                "total_duration": done_data.get("total_duration", 0),
                "load_duration": done_data.get("load_duration", 0),
                "prompt_eval_count": done_data.get("prompt_eval_count", 0),
                "eval_count": done_data.get("eval_count", 0),
            }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


async def _stream_chat(
    model_name: str,
    messages: list[dict],
    temperature: float,
    top_k: int,
    max_tokens: int,
    stop: list[str] | None = None,
) -> AsyncIterator[str]:
    """Async generator for chat streaming — Ollama-style."""
    t_start = time.time()
    model, meta = _load_model(model_name)
    t_loaded = time.time()
    device = next(model.parameters()).device

    prompt = _format_prompt_chat(messages)
    tokens = _encode(prompt, meta, model)
    prompt_token_count = len(tokens)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    created_at = datetime.now(tz=timezone.utc).isoformat()
    stop_seqs = list(stop or []) + ["■"]   # ■ = EOS, model tự học khi nào dừng
    generated_text = ""
    eval_count = 0

    t_gen_start = time.time()
    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta, model)
        # Không phát ký tự EOS ra client
        if "■" in char:
            break
        generated_text += char
        eval_count += 1
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": char},
            "done": False,
        }
        yield json.dumps(chunk, ensure_ascii=False) + "\n"
        await asyncio.sleep(0)

        if stop_seqs and any(s in generated_text for s in stop_seqs):
            break

    t_end = time.time()
    done_chunk = {
        "model": model_name,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": int((t_end - t_start) * 1e9),
        "load_duration": int((t_loaded - t_start) * 1e9),
        "prompt_eval_count": prompt_token_count,
        "prompt_eval_duration": int((t_gen_start - t_loaded) * 1e9),
        "eval_count": eval_count,
        "eval_duration": int((t_end - t_gen_start) * 1e9),
    }
    yield json.dumps(done_chunk) + "\n"


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Ollama /api/chat — chat completion."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    ollama_target = _ollama_target(model_name)
    if ollama_target:
        return await _ollama_chat_proxy(req, model_name, ollama_target)

    messages = [m.model_dump() for m in req.messages]
    temperature = req.options.get("temperature", req.temperature)
    top_k = req.options.get("top_k", req.top_k)
    max_tokens = req.options.get("num_predict", req.max_tokens)
    stop = req.stop or req.options.get("stop", [])
    known_answer = _known_vi_answer(model_name, messages)

    async def _stream_known_answer() -> AsyncIterator[str]:
        created_at = datetime.now(tz=timezone.utc).isoformat()
        assert known_answer is not None
        for char in known_answer:
            yield json.dumps({
                "model": model_name,
                "created_at": created_at,
                "message": {"role": "assistant", "content": char},
                "done": False,
            }, ensure_ascii=False) + "\n"
            await asyncio.sleep(0)
        yield json.dumps({
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "faq",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": len(known_answer),
        }) + "\n"

    try:
        if known_answer is not None:
            if req.stream:
                return StreamingResponse(_stream_known_answer(), media_type="application/x-ndjson")
            return {
                "model": model_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": known_answer},
                "done": True,
                "done_reason": "faq",
            }

        if req.stream:
            return StreamingResponse(
                _stream_chat(model_name, messages, temperature, top_k, max_tokens, stop),
                media_type="application/x-ndjson",
            )
        else:
            full = ""
            done_data = {}
            async for chunk in _stream_chat(model_name, messages, temperature, top_k, max_tokens, stop):
                data = json.loads(chunk)
                full += data.get("message", {}).get("content", "")
                if data.get("done"):
                    done_data = data
            return {
                "model": model_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": full},
                "done": True,
                "done_reason": "stop",
                "total_duration": done_data.get("total_duration", 0),
                "load_duration": done_data.get("load_duration", 0),
                "prompt_eval_count": done_data.get("prompt_eval_count", 0),
                "eval_count": done_data.get("eval_count", 0),
            }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


def _get_embedding(model_name: str, text: str) -> list[float]:
    """Return mean-pooled hidden state as embedding vector."""
    model, meta = _load_model(model_name)
    device = next(model.parameters()).device
    tokens = _encode(text, meta, model)
    if not tokens:
        tokens = [0]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    if isinstance(model, HFModel):
        # Use HF model's hidden states
        with torch.no_grad():
            out = model.hf_model(input_ids=idx, output_hidden_states=True)
            embedding = out.hidden_states[-1][0].mean(dim=0)
        return embedding.tolist()

    with torch.no_grad():
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
        for block in model.transformer.h:
            x = block(x)
        x = model.transformer.ln_f(x)
        embedding = x[0].mean(dim=0)

    return embedding.tolist()


@app.post("/api/embeddings")
async def api_embeddings(req: EmbeddingsRequest):
    """Ollama /api/embeddings — generate embedding vector (legacy)."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    text = req.prompt if req.prompt else (req.input if isinstance(req.input, str) else req.input[0] if req.input else "")
    try:
        embedding = _get_embedding(model_name, text)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return {"embedding": embedding, "model": model_name}


@app.post("/api/embed")
async def api_embed(req: EmbeddingsRequest):
    """Ollama /api/embed — generate embeddings (new multi-input endpoint)."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    inputs = req.input if req.input else ([req.prompt] if req.prompt else [""])
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        embeddings = [_get_embedding(model_name, text) for text in inputs]
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return {
        "model": model_name,
        "embeddings": embeddings,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": len(inputs),
    }


# ─── OpenAI-compatible endpoints ─────────────────────────────────────────────

class OAIToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string


class OAIToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: OAIToolCallFunction


class OAIChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[OAIToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # for tool role


class OAIChatRequest(BaseModel):
    model: str = ""
    messages: list[OAIChatMessage] = []
    stream: bool = False
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 256
    n: int = 1
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None


class OAICompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    stream: bool = False
    temperature: float = 0.8
    max_tokens: int = 256


@app.get("/v1/models")
async def oai_list_models():
    """OpenAI-compatible model list."""
    models = _discover_models()
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "ai-local"}
            for m in models
        ],
    }


async def _oai_stream_chat(model_name, messages, temperature, top_k, max_tokens):
    """OpenAI SSE format streaming."""
    request_id = f"chatcmpl-{int(time.time())}"
    model, meta = _load_model(model_name)
    device = next(model.parameters()).device

    prompt = _format_prompt_chat(messages)
    tokens = _encode(prompt, meta, model)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta, model)
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


def _inject_tools_into_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """
    Với model nhỏ không hỗ trợ function calling native,
    inject tool descriptions vào system prompt theo ReAct format.
    """
    tools_text = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tools
    )
    inject = (
        "\n\n[TOOLS AVAILABLE — gọi bằng: Action: tool_name({\"key\": \"val\"})]\n"
        + tools_text
        + "\n[Khi hoàn thành: Final Answer: câu trả lời]"
    )

    result = []
    found_system = False
    for msg in messages:
        if msg.get("role") == "system" and not found_system:
            result.append({**msg, "content": (msg.get("content") or "") + inject})
            found_system = True
        else:
            # Chuẩn hóa tool messages thành user messages cho model nhỏ
            if msg.get("role") == "tool":
                result.append({
                    "role": "user",
                    "content": f"[Tool result for {msg.get('tool_call_id', '')}]: {msg.get('content', '')}",
                })
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc_texts = []
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    tc_texts.append(f"Action: {fn.get('name', '')}({fn.get('arguments', '{}')})")
                result.append({
                    "role": "assistant",
                    "content": (msg.get("content") or "") + "\n" + "\n".join(tc_texts),
                })
            else:
                result.append(msg)

    if not found_system:
        result.insert(0, {"role": "system", "content": "Bạn là AI assistant đa năng." + inject})

    return result


@app.post("/v1/chat/completions")
async def oai_chat_completions(req: OAIChatRequest):
    """OpenAI-compatible chat completions với tool calling support."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    # Tool calling: nếu có tools, inject vào prompt (local model không hỗ trợ native)
    effective_tools = req.tools or []
    if effective_tools:
        messages = _inject_tools_into_messages(messages, effective_tools)

    try:
        if req.stream:
            return StreamingResponse(
                _oai_stream_chat(model_name, messages, req.temperature, 50, req.max_tokens),
                media_type="text/event-stream",
            )

        full = ""
        async for chunk in _stream_chat(model_name, messages, req.temperature, 50, req.max_tokens):
            data = json.loads(chunk)
            full += data.get("message", {}).get("content", "")

        # Parse tool calls từ output nếu có tools
        parsed_tool_calls = None
        finish_reason = "stop"
        if effective_tools and "Action:" in full:
            parsed_tool_calls = _parse_tool_calls_from_text(full)
            if parsed_tool_calls:
                finish_reason = "tool_calls"

        choice_msg = {"role": "assistant", "content": full}
        if parsed_tool_calls:
            choice_msg["tool_calls"] = parsed_tool_calls

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": choice_msg,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(full), "total_tokens": len(full)},
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


def _parse_tool_calls_from_text(text: str) -> list[dict] | None:
    """Extract tool calls từ ReAct format text."""
    import re as _re
    calls = []
    for m in _re.finditer(r"Action:\s*(\w+)\s*\((\{.*?\})\)", text, _re.DOTALL):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return calls if calls else None


class OAIEmbeddingRequest(BaseModel):
    model: str = ""
    input: str | list[str] = ""
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def oai_embeddings(req: OAIEmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    inputs = req.input if isinstance(req.input, list) else [req.input]
    try:
        data = [
            {"object": "embedding", "index": i, "embedding": _get_embedding(model_name, text)}
            for i, text in enumerate(inputs)
        ]
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": sum(len(t) for t in inputs), "total_tokens": sum(len(t) for t in inputs)},
    }


@app.post("/v1/completions")
async def oai_completions(req: OAICompletionRequest):
    """OpenAI-compatible text completions."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    try:
        full = ""
        async for chunk in _stream_generate(model_name, req.prompt, req.temperature, 50, req.max_tokens):
            data = json.loads(chunk)
            full += data.get("response", "")

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"text": full, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(full), "total_tokens": len(full)},
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/", response_class=HTMLResponse)
async def root():
    """Giao diện chat web. Mở http://localhost:11434 trên trình duyệt."""
    ui_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>AI-Local</h1><p>Web UI not found (web/index.html).</p>")


@app.get("/api/status")
async def api_status():
    """Ollama-compatible probe (giữ chuỗi nhận diện cho client cũ)."""
    return "Ollama is running"


@app.get("/api/computer-use/state")
async def api_computer_use_state():
    """Trạng thái computer-use hiện tại: màn hình, chuột, cửa sổ active."""
    return get_computer_state()


@app.post("/api/computer-use/command")
async def api_computer_use_command(req: ComputerCommandRequest):
    """Chạy một lệnh computer-use trực tiếp, phục vụ desktop UI/tooling local."""
    result = execute_computer_command(req.command, req.enabled)
    if result is None:
        return {
            "handled": False,
            "ok": False,
            "message": "Không nhận diện đây là lệnh điều khiển máy.",
            "action": "unhandled",
            "data": {},
        }
    return {
        "handled": result.handled,
        "ok": result.ok,
        "message": result.message,
        "action": result.action,
        "data": result.data,
    }


@app.get("/api/computer-use/files/{filename}")
async def api_computer_use_file(filename: str):
    """Trả file ảnh/sản phẩm computer-use đã tạo trong output/computer-use."""
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(400, "Invalid filename.")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "computer-use"))
    path = os.path.abspath(os.path.join(base_dir, safe_name))
    if not path.startswith(base_dir + os.sep) or not os.path.exists(path):
        raise HTTPException(404, "Computer-use file not found.")
    return FileResponse(path)


@app.head("/")
async def root_head():
    return {}


# ─── Agent & Tools endpoints ──────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str
    model: str = ""
    tools: Optional[list[str]] = None   # tên tools; None = tất cả
    system_prompt: str = ""
    max_steps: int = 10
    temperature: float = 0.2
    max_tokens: int = 1024
    mode: str = "auto"  # "auto" | "react" | "function_calling"


class AgentStreamRequest(AgentRequest):
    stream: bool = True


@app.post("/v1/agent")
async def v1_agent(req: AgentRequest):
    """
    Chạy agent hoàn thành nhiệm vụ với tool use.

    Agent sẽ:
    1. Nhận nhiệm vụ (task)
    2. Tự động chọn và gọi tools
    3. Lặp cho đến khi hoàn thành (tối đa max_steps bước)
    4. Trả về kết quả + trace

    Ví dụ:
        POST /v1/agent
        {"task": "Tìm hiểu Python asyncio và tóm tắt"}
    """
    from agent import run_agent, format_result

    model_name = req.model or _default_model
    if not model_name:
        models_list = _discover_models()
        model_name = models_list[-1] if models_list else "chat_vi"

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run_agent(
            task=req.task,
            model=model_name,
            server_url=f"http://127.0.0.1:{_server_port}",
            tools=req.tools,
            system_prompt=req.system_prompt,
            max_steps=req.max_steps,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            mode=req.mode,
        )
    )

    return {
        "answer": result.answer,
        "model": result.model,
        "elapsed": result.elapsed,
        "success": result.success,
        "error": result.error,
        "steps": [
            {
                "step": s.step,
                "thought": s.thought,
                "is_final": s.is_final,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": tc.result[:2000],
                    }
                    for tc in s.tool_calls
                ],
            }
            for s in result.steps
        ],
    }


@app.get("/v1/tools")
async def v1_list_tools():
    """Liệt kê tất cả tools có sẵn cho agent."""
    schemas = get_tool_schemas()
    return {
        "tools": [
            {
                "name": s["function"]["name"],
                "description": s["function"]["description"],
                "parameters": s["function"]["parameters"],
            }
            for s in schemas
        ],
        "count": len(schemas),
    }


@app.post("/v1/tools/call")
async def v1_call_tool(req: dict):
    """
    Gọi trực tiếp một tool.

    Body: {"name": "read_file", "arguments": {"path": "README.md"}}
    """
    name = req.get("name", "")
    arguments = req.get("arguments", {})
    if not name:
        raise HTTPException(400, "Thiếu tên tool ('name')")
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: call_tool(name, arguments)
    )
    return {"name": name, "result": result}


# ─── MCP server management ────────────────────────────────────────────────────

class MCPServerConfig(BaseModel):
    name: str
    url: str


@app.get("/v1/mcp/servers")
async def v1_mcp_list():
    """Liệt kê các MCP servers đang kết nối."""
    from mcp_client import get_mcp_client
    client = get_mcp_client()
    return {
        "servers": [
            {
                "name": s.name,
                "url": s.url,
                "connected": s.connected,
                "tools": [{"name": t.name, "description": t.description} for t in s.tools],
                "error": s.error,
            }
            for s in client.list_servers()
        ]
    }


@app.post("/v1/mcp/servers")
async def v1_mcp_add(cfg: MCPServerConfig):
    """Kết nối tới một MCP server mới."""
    from mcp_client import get_mcp_client
    client = get_mcp_client()
    server = await client.add_server(cfg.name, cfg.url)
    return {
        "name": server.name,
        "url": server.url,
        "connected": server.connected,
        "tools_count": len(server.tools),
        "error": server.error,
    }


@app.delete("/v1/mcp/servers/{name}")
async def v1_mcp_remove(name: str):
    """Ngắt kết nối MCP server."""
    from mcp_client import get_mcp_client
    client = get_mcp_client()
    client.remove_server(name)
    return {"removed": name}


@app.get("/v1/mcp/tools")
async def v1_mcp_tools():
    """Liệt kê tất cả tools từ các MCP servers đang kết nối."""
    from mcp_client import get_mcp_client
    client = get_mcp_client()
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in client.all_tools()
        ]
    }


# Port hiện tại (dùng cho agent self-call)
_server_port: int = 11434


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-local REST API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--model", type=str, default="", help="Default model to load")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    _default_model = args.model
    _data_dir = args.data_dir
    _checkpoints_dir = args.checkpoints_dir
    _server_port = args.port

    ui_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"AI-local server starting on http://{args.host}:{args.port}")
    print(f"\n  💬 Mở giao diện chat tại:  http://{ui_host}:{args.port}\n")
    print(f"Checkpoints dir: {args.checkpoints_dir}")
    models = _discover_models()
    if models:
        print(f"Available models: {', '.join(models)}")
    else:
        print("No models found — run the training pipeline first.")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
