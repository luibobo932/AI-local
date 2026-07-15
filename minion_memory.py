"""Kho trí nhớ SQLite local cho Minion, không phụ thuộc dịch vụ cloud."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from minion_core import plain_text


@dataclass(slots=True)
class MemoryItem:
    id: str
    content: str
    category: str
    source: str
    importance: float
    created_at: str
    updated_at: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", plain_text(text)) if len(token) > 1}


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


class MemoryStore:
    """Lưu trí nhớ, tìm theo từ khóa và có thể cộng điểm semantic embedding."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    source TEXT NOT NULL DEFAULT 'user',
                    importance REAL NOT NULL DEFAULT 0.5,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    embedding_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC)")

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM memories").fetchone()
        return int(row["total"] if row else 0)

    def add(
        self,
        content: str,
        category: str = "general",
        source: str = "user",
        importance: float = 0.5,
        embedding: list[float] | None = None,
        embedding_model: str = "",
    ) -> MemoryItem:
        content = content.strip()
        if not content:
            raise ValueError("Nội dung trí nhớ không được để trống.")
        importance = max(0.0, min(float(importance), 1.0))
        item_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, content, normalized_content, category, source, importance,
                    embedding_model, embedding_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    content,
                    plain_text(content),
                    (category or "general").strip(),
                    (source or "user").strip(),
                    importance,
                    embedding_model,
                    json.dumps(embedding or []),
                    now,
                    now,
                ),
            )
        return MemoryItem(item_id, content, category or "general", source or "user", importance, now, now)

    def list(self, limit: int = 50, category: str = "") -> list[MemoryItem]:
        limit = max(1, min(int(limit), 500))
        sql = "SELECT * FROM memories"
        params: list[object] = []
        if category.strip():
            sql += " WHERE category = ?"
            params.append(category.strip())
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search(
        self,
        query: str,
        limit: int = 5,
        query_embedding: list[float] | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryItem]:
        query = query.strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 50))
        query_tokens = _tokens(query)
        query_plain = plain_text(query)

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC LIMIT 2000").fetchall()

        ranked: list[MemoryItem] = []
        for row in rows:
            content_tokens = _tokens(row["content"])
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            phrase = 1.0 if query_plain and query_plain in row["normalized_content"] else 0.0
            embedding = self._decode_embedding(row["embedding_json"])
            semantic = max(0.0, _cosine(query_embedding, embedding))
            importance = float(row["importance"])
            score = (0.50 * semantic) + (0.30 * overlap) + (0.15 * phrase) + (0.05 * importance)
            if score >= min_score:
                item = self._row_to_item(row)
                item.score = round(score, 6)
                ranked.append(item)

        ranked.sort(key=lambda item: (item.score, item.updated_at), reverse=True)
        return ranked[:limit]

    def delete(self, item_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
        return cursor.rowcount > 0

    def delete_by_source(self, source: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE source = ?", (source,))
        return int(cursor.rowcount)

    @staticmethod
    def format_context(items: Iterable[MemoryItem]) -> str:
        lines = []
        for item in items:
            lines.append(f"- [{item.category} | nguồn: {item.source}] {item.content}")
        return "\n".join(lines)

    @staticmethod
    def _decode_embedding(raw: str) -> list[float] | None:
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return [float(item) for item in value] if isinstance(value, list) else None
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            source=row["source"],
            importance=float(row["importance"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
