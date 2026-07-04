"""
Trợ lý BĐS chuyên sâu cho Minion — quản lý khách hàng + soạn tin gọi khách.

Lệnh tự nhiên hỗ trợ (gõ thẳng trong chat Minion):
    khách anh Tuấn cần nhà Quận 10 dưới 8 tỷ ngang 4m   -> lưu/cập nhật nhu cầu
    gợi ý cho khách anh Tuấn                             -> lọc căn theo nhu cầu đã lưu
    danh sách khách                                      -> xem toàn bộ khách + nhu cầu
    xóa khách anh Tuấn                                   -> xóa khách
    soạn tin gửi khách căn 87 Nguyễn Văn Cừ              -> tin nhắn sẵn để copy gửi Zalo/SMS

Dữ liệu lưu tại .ai-local/customers.json (per-machine, không commit).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

_STORE_DIR = Path(__file__).resolve().parent / ".ai-local"
CUSTOMERS_PATH = _STORE_DIR / "customers.json"


def _plain(text: str) -> str:
    text = text.lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


# ─── Kho khách hàng ───────────────────────────────────────────────────────────

def load_customers(path: Path = CUSTOMERS_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_customers(data: dict, path: Path = CUSTOMERS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def upsert_customer(name: str, need: str, path: Path = CUSTOMERS_PATH) -> dict:
    data = load_customers(path)
    key = _plain(name)
    entry = data.get(key) or {}
    entry.update({
        "name": name.strip(),
        "need": need.strip(),
        "updated": datetime.now(tz=timezone.utc).isoformat(),
    })
    data[key] = entry
    save_customers(data, path)
    return entry


def find_customer(name: str, path: Path = CUSTOMERS_PATH) -> dict | None:
    data = load_customers(path)
    key = _plain(name)
    if key in data:
        return data[key]
    # khớp mềm: "anh tuan" ~ "tuan"
    for k, v in data.items():
        if key in k or k in key:
            return v
    return None


def delete_customer(name: str, path: Path = CUSTOMERS_PATH) -> bool:
    data = load_customers(path)
    key = _plain(name)
    if key in data:
        del data[key]
        save_customers(data, path)
        return True
    for k in list(data):
        if key in k or k in key:
            del data[k]
            save_customers(data, path)
            return True
    return False


# ─── Nhận diện ý định ─────────────────────────────────────────────────────────

def detect_intent(question: str) -> tuple[str, dict] | None:
    """Trả về (intent, params) hoặc None nếu không phải lệnh BĐS chuyên sâu."""
    plain = _plain(question)

    # danh sách khách
    if re.match(r"^(?:danh sach|xem|liet ke)\s+khach(?:\s+hang)?\s*$", plain):
        return "list_customers", {}

    # xóa khách <tên>
    m = re.match(r"^xoa\s+khach(?:\s+hang)?\s+(.+)$", plain)
    if m:
        return "delete_customer", {"name": _original_tail(question, m.group(1))}

    # gợi ý / tìm nhà cho khách <tên>
    m = re.match(
        r"^(?:goi y|tim nha|tim can|de xuat|loc can)(?:\s+\S+)*?\s+cho\s+khach(?:\s+hang)?\s+(.+)$",
        plain,
    )
    if m:
        return "suggest_for_customer", {"name": _original_tail(question, m.group(1))}

    # soạn tin gửi khách [cho] căn <mô tả căn>
    m = re.match(
        r"^(?:soan|viet)\s+tin(?:\s+nhan)?(?:\s+(?:goi|gui))?(?:\s+khach(?:\s+hang)?)?"
        r"(?:\s+cho)?(?:\s+can)?\s+(.+)$",
        plain,
    )
    if m:
        return "call_script", {"house_query": _original_tail(question, m.group(1))}

    # khách <tên> cần/muốn/tìm <nhu cầu>   (hoặc "lưu khách ...")
    m = re.match(
        r"^(?:luu\s+)?khach(?:\s+hang)?\s+(.+?)\s+(?:can|muon|tim|thich|dang tim|:)\s+(.+)$",
        plain,
    )
    if m:
        name = _original_tail(question, m.group(1))
        need = _original_tail(question, m.group(2))
        if name and need:
            return "save_customer", {"name": name, "need": need}

    return None


def _original_tail(original: str, plain_fragment: str) -> str:
    """Khôi phục đoạn text CÓ DẤU tương ứng với fragment đã bỏ dấu.

    So khớp theo số từ: fragment n từ cuối/giữa câu — tìm chuỗi từ trong câu gốc
    có bản plain trùng fragment.
    """
    orig_words = original.split()
    frag_words = plain_fragment.split()
    n = len(frag_words)
    for start in range(len(orig_words) - n + 1):
        window = orig_words[start:start + n]
        if _plain(" ".join(window)) == plain_fragment:
            return " ".join(window).strip(" .,!?:")
    return plain_fragment  # fallback: dùng bản không dấu


# ─── Soạn tin nhắn gọi khách ──────────────────────────────────────────────────

def render_call_script(house_block: str, note: str = "") -> str:
    """Tạo tin nhắn sẵn để copy gửi Zalo/SMS từ block thông tin căn nhà."""
    # bỏ số thứ tự "1. " ở đầu block nếu có
    body = re.sub(r"^\s*(?:\d+\.|-)\s*", "", house_block.strip())
    lines = [
        "📨 Tin nhắn gửi khách (copy gửi Zalo/SMS):",
        "─" * 30,
        "Chào anh/chị 😊",
        "Em gửi anh/chị thông tin căn nhà đang bán:",
        "",
        body,
        "",
        "Vị trí đẹp, giá tốt so với khu này. Anh/chị xem qua giúp em,",
        "nếu thấy hợp em sắp lịch dẫn anh/chị đi xem nhà luôn ạ!",
        "─" * 30,
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)


# ─── Handler chính (server gọi) ───────────────────────────────────────────────

async def handle(question: str, fetch_context, path: Path = CUSTOMERS_PATH) -> str | None:
    """Xử lý lệnh BĐS chuyên sâu. fetch_context = async fn(query) -> str.

    Trả None nếu câu hỏi không phải lệnh chuyên sâu (để rơi xuống RAG thường).
    """
    intent = detect_intent(question)
    if intent is None:
        return None
    kind, params = intent

    if kind == "list_customers":
        data = load_customers(path)
        if not data:
            return "Chưa có khách nào. Lưu bằng cách gõ: `khách anh Tuấn cần nhà Quận 10 dưới 8 tỷ`."
        lines = [f"📇 Danh sách khách ({len(data)}):"]
        for i, entry in enumerate(sorted(data.values(), key=lambda e: e.get("updated", ""), reverse=True), 1):
            lines.append(f"{i}. {entry.get('name')} — cần: {entry.get('need')}")
        lines.append("\nGõ `gợi ý cho khách <tên>` để Minion lọc căn phù hợp.")
        return "\n".join(lines)

    if kind == "delete_customer":
        ok = delete_customer(params["name"], path)
        return (
            f"Đã xóa khách {params['name']}." if ok
            else f"Không tìm thấy khách {params['name']} trong danh sách."
        )

    if kind == "save_customer":
        entry = upsert_customer(params["name"], params["need"], path)
        return (
            f"✅ Đã lưu khách **{entry['name']}**\n"
            f"Nhu cầu: {entry['need']}\n\n"
            f"Gõ `gợi ý cho khách {entry['name']}` để lọc căn phù hợp ngay."
        )

    if kind == "suggest_for_customer":
        entry = find_customer(params["name"], path)
        if entry is None:
            return (
                f"Chưa có khách tên {params['name']}. "
                f"Lưu trước bằng: `khách {params['name']} cần <nhu cầu>`."
            )
        context = await fetch_context(entry.get("need", ""))
        header = f"🎯 Gợi ý cho khách **{entry.get('name')}** (nhu cầu: {entry.get('need')}):\n\n"
        if not context or context.startswith("Dữ liệu nhà Supabase"):
            return header + (context or "Chưa kết nối được dữ liệu nhà.")
        return header + context

    if kind == "call_script":
        query = params["house_query"]
        context = await fetch_context(f"{query} lấy 1 căn")
        if not context or context.startswith("Dữ liệu nhà Supabase") or "chưa" in context[:60].lower():
            return f"Không tìm thấy căn khớp \"{query}\" để soạn tin. Thử ghi rõ số nhà/đường/quận hơn."
        # tìm căn ĐẦU TIÊN trong toàn bộ kết quả: dòng bắt đầu bằng "1." / "2." ...
        lines = context.strip().splitlines()
        start = next(
            (i for i, ln in enumerate(lines) if re.match(r"^\s*\d+\.\s", ln)),
            None,
        )
        if start is None:
            return f"Không tách được thông tin căn từ kết quả cho \"{query}\"."
        block = [lines[start]]
        for ln in lines[start + 1:]:
            if not ln.strip() or re.match(r"^\s*\d+\.\s", ln):
                break
            block.append(ln)
        return render_call_script("\n".join(block))

    return None
