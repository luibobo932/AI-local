"""
rag.py — Bộ nhớ ngoài (RAG) ngữ nghĩa cho minion.

Tự chia nhỏ mã nguồn trong dự án -> tạo embedding (qua Ollama/LM Studio/OpenAI)
-> lưu vào vector store local (.ai-local/rag_index.json) -> tra cứu đoạn code
liên quan để nạp vào context trước khi trả lời.

Nhẹ, chỉ dùng stdlib (+ numpy nếu có để tính nhanh hơn). Không cần langchain.

CLI:
    python rag.py index [--root .]          # lập chỉ mục dự án
    python rag.py search "câu truy vấn" [-k 5]
    python rag.py status

Cấu hình (env / .env):
    LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY  — như llm.py
    LLM_EMBED_MODEL                          — model embedding, vd: nomic-embed-text
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import os
from typing import Iterable

from llm import LLMClient, LLMError, load_env

logger = logging.getLogger("minion.rag")

INDEX_DIR = ".ai-local"
INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.json")

CODE_EXT = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c",
            ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
            ".html", ".css", ".scss", ".vue", ".svelte", ".sql",
            ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".txt"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".ai-local", "checkpoints",
             "models", "dist", "build", ".venv", "venv", ".gradle", "android",
             "hf_space", ".idea", ".vscode"}
MAX_FILE_BYTES = 400_000
CHUNK_LINES = 60
CHUNK_OVERLAP = 12

try:
    import numpy as _np
except Exception:
    _np = None


# ── Thu thập & chia nhỏ ───────────────────────────────────────────────────────

def iter_code_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in CODE_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def chunk_file(path: str, root: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    rel = os.path.relpath(path, root)
    chunks = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, max(1, len(lines)), step):
        block = lines[start:start + CHUNK_LINES]
        text = "".join(block).strip()
        if not text:
            continue
        chunks.append({
            "file": rel,
            "start": start + 1,
            "end": min(start + CHUNK_LINES, len(lines)),
            "text": text,
        })
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ── Vector store (JSON local) ─────────────────────────────────────────────────

def load_index() -> dict:
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"model": None, "dim": None, "items": []}


def save_index(index: dict) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


# ── Lập chỉ mục ───────────────────────────────────────────────────────────────

def index_project(root: str = ".", client: LLMClient | None = None,
                  batch: int = 32, progress=None) -> dict:
    """Quét dự án, embed các chunk mới/đổi, lưu index. Tái dùng vector cũ theo hash."""
    client = client or LLMClient.from_env()
    old = load_index()
    cache = {it["id"]: it for it in old.get("items", [])}

    wanted = []
    for path in iter_code_files(root):
        for ch in chunk_file(path, root):
            ch["id"] = _hash(f"{ch['file']}:{ch['start']}:{ch['text'][:200]}")
            wanted.append(ch)

    items, to_embed = [], []
    for ch in wanted:
        hit = cache.get(ch["id"])
        if hit and hit.get("vector") and hit.get("text") == ch["text"]:
            items.append(hit)
        else:
            to_embed.append(ch)

    if progress:
        progress(f"{len(wanted)} chunk, cần embed {len(to_embed)} (tái dùng {len(items)})")

    dim = old.get("dim")
    for i in range(0, len(to_embed), batch):
        group = to_embed[i:i + batch]
        vecs = client.embed([c["text"] for c in group])
        if isinstance(vecs[0], (int, float)):  # phòng khi API trả 1 vector
            vecs = [vecs]
        for c, v in zip(group, vecs):
            c["vector"] = v
            dim = dim or len(v)
            items.append({k: c[k] for k in ("id", "file", "start", "end", "text", "vector")})
        if progress:
            progress(f"  đã embed {min(i + batch, len(to_embed))}/{len(to_embed)}")

    index = {"model": os.environ.get("LLM_EMBED_MODEL", "") or client.model,
             "dim": dim, "items": items}
    save_index(index)
    return index


# ── Truy vấn ──────────────────────────────────────────────────────────────────

def _cosine(a, b) -> float:
    if _np is not None:
        a = _np.asarray(a, dtype=_np.float32); b = _np.asarray(b, dtype=_np.float32)
        na = float(_np.linalg.norm(a)); nb = float(_np.linalg.norm(b))
        return float(a.dot(b) / (na * nb)) if na and nb else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query: str, client: LLMClient | None = None, k: int = 5,
             index: dict | None = None) -> list[dict]:
    """Trả về k chunk liên quan nhất: [{file,start,end,text,score}]."""
    index = index or load_index()
    items = index.get("items", [])
    if not items:
        return []
    client = client or LLMClient.from_env()
    qv = client.embed(query)
    scored = []
    for it in items:
        scored.append((_cosine(qv, it["vector"]), it))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, it in scored[:k]:
        out.append({"file": it["file"], "start": it["start"], "end": it["end"],
                    "text": it["text"], "score": round(score, 4)})
    return out


def format_context(chunks: list[dict], max_chars: int = 6000) -> str:
    """Ghép các chunk thành block context để nạp vào prompt."""
    parts, total = [], 0
    for c in chunks:
        header = f"# {c['file']} (dòng {c['start']}-{c['end']}, score {c.get('score','')})"
        block = header + "\n" + c["text"]
        if total + len(block) > max_chars:
            break
        parts.append(block); total += len(block)
    return "\n\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="RAG ngữ nghĩa cho minion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="Lập chỉ mục dự án")
    pi.add_argument("--root", default=".")
    ps = sub.add_parser("search", help="Tìm đoạn code liên quan")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=5)
    sub.add_parser("status", help="Xem trạng thái index")
    args = ap.parse_args()

    try:
        if args.cmd == "index":
            idx = index_project(args.root, progress=lambda m: print(m))
            print(f"✓ Đã lập chỉ mục {len(idx['items'])} chunk (model={idx['model']}, dim={idx['dim']}).")
        elif args.cmd == "search":
            hits = retrieve(args.query, k=args.k)
            if not hits:
                print("Chưa có index. Chạy: python rag.py index"); return
            for h in hits:
                print(f"\n[{h['score']}] {h['file']} (dòng {h['start']}-{h['end']})")
                print("  " + h["text"][:200].replace("\n", "\n  "))
        elif args.cmd == "status":
            idx = load_index()
            print(f"Index: {len(idx.get('items', []))} chunk | model={idx.get('model')} | dim={idx.get('dim')}")
            print(f"File: {INDEX_PATH}")
    except LLMError as e:
        print(f"Lỗi LLM: {e}")


if __name__ == "__main__":
    main()
