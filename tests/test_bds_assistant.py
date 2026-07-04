"""Test trợ lý BĐS chuyên sâu: khách hàng + soạn tin gọi khách."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bds_assistant as bds


# ─── detect_intent ────────────────────────────────────────────────────────────

def test_intent_save_customer():
    kind, p = bds.detect_intent("khách anh Tuấn cần nhà Quận 10 dưới 8 tỷ ngang 4m")
    assert kind == "save_customer"
    assert p["name"] == "anh Tuấn"
    assert "Quận 10" in p["need"]


def test_intent_save_customer_with_luu():
    kind, p = bds.detect_intent("lưu khách chị Hoa muốn nhà Gò Vấp tầm 5 tỷ")
    assert kind == "save_customer"
    assert p["name"] == "chị Hoa"


def test_intent_suggest():
    kind, p = bds.detect_intent("gợi ý cho khách anh Tuấn")
    assert kind == "suggest_for_customer"
    assert p["name"] == "anh Tuấn"


def test_intent_list_and_delete():
    assert bds.detect_intent("danh sách khách")[0] == "list_customers"
    kind, p = bds.detect_intent("xóa khách anh Tuấn")
    assert kind == "delete_customer" and p["name"] == "anh Tuấn"


def test_intent_call_script():
    kind, p = bds.detect_intent("soạn tin gửi khách căn 87 Nguyễn Văn Cừ")
    assert kind == "call_script"
    assert "Nguyễn Văn Cừ" in p["house_query"]


def test_normal_house_query_not_hijacked():
    assert bds.detect_intent("Tìm nhà Quận 10 dưới 10 tỷ") is None
    assert bds.detect_intent("Có căn nào ngang từ 4m không?") is None
    assert bds.detect_intent("Tóm tắt căn phù hợp để gọi khách") is None
    assert bds.detect_intent("Minion, bạn là ai?") is None


# ─── store CRUD ───────────────────────────────────────────────────────────────

def test_customer_crud(tmp_path):
    store = tmp_path / "customers.json"
    bds.upsert_customer("anh Tuấn", "nhà Quận 10 dưới 8 tỷ", store)
    bds.upsert_customer("chị Hoa", "Gò Vấp 5 tỷ", store)

    found = bds.find_customer("anh tuan", store)  # không dấu vẫn tìm ra
    assert found and found["need"] == "nhà Quận 10 dưới 8 tỷ"

    # khớp mềm theo tên ngắn
    assert bds.find_customer("Tuấn", store) is not None

    assert bds.delete_customer("chị Hoa", store) is True
    assert bds.find_customer("chị Hoa", store) is None
    assert len(bds.load_customers(store)) == 1


# ─── handler end-to-end (fetch giả) ──────────────────────────────────────────

FAKE_HOUSE_BLOCK = (
    "Tìm thấy 1 căn đúng điều kiện:\n"
    "1. 87 Nguyễn Văn Cừ, Quận 5\n"
    "   DT: 4.5 x 45m (202.5m2)\n"
    "   Giá: 45 tỷ\n"
    "   Loại: Nhà Mặt tiền"
)


async def fake_fetch(query: str) -> str:
    return FAKE_HOUSE_BLOCK


def _run(coro):
    return asyncio.run(coro)


def test_handle_save_then_suggest(tmp_path):
    store = tmp_path / "c.json"
    out1 = _run(bds.handle("khách anh Tuấn cần nhà Quận 5 mặt tiền", fake_fetch, store))
    assert "Đã lưu khách" in out1

    out2 = _run(bds.handle("gợi ý cho khách anh Tuấn", fake_fetch, store))
    assert "Gợi ý cho khách" in out2
    assert "Nguyễn Văn Cừ" in out2


def test_handle_call_script(tmp_path):
    out = _run(bds.handle("soạn tin gửi khách căn 87 Nguyễn Văn Cừ", fake_fetch, tmp_path / "c.json"))
    assert "Tin nhắn gửi khách" in out
    assert "87 Nguyễn Văn Cừ, Quận 5" in out
    assert "Giá: 45 tỷ" in out
    assert "sắp lịch" in out


def test_handle_returns_none_for_normal_chat(tmp_path):
    assert _run(bds.handle("thời tiết hôm nay thế nào", fake_fetch, tmp_path / "c.json")) is None
