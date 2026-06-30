"""
LLM client provider-agnostic cho Minion — CHỈ DÙNG STDLIB (urllib).

Kết nối tới bất kỳ endpoint OpenAI-compatible nào:
- Ollama       (http://localhost:11434/v1)   — Qwen Coder, DeepSeek Coder...
- LM Studio    (http://localhost:1234/v1)
- OpenAI       (https://api.openai.com/v1)    — hoặc API tương thích
- ai-local     (server built-in của dự án)

Cấu hình qua biến môi trường (hoặc file .env):
    LLM_PROVIDER   ollama | lmstudio | openai | ai-local   (mặc định: ollama)
    LLM_BASE_URL   ghi đè base url
    LLM_MODEL      tên model (vd: qwen2.5-coder)
    LLM_API_KEY    API key (chỉ cần cho openai); KHÔNG hard-code

Sử dụng:
    from llm import LLMClient, load_env
    load_env()
    client = LLMClient.from_env()
    resp = client.chat([{"role": "user", "content": "Xin chào"}])
    print(resp["choices"][0]["message"]["content"])
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger("minion.llm")

# ─── Provider mặc định ────────────────────────────────────────────────────────

PROVIDERS = {
    "ollama":   {"base_url": "http://localhost:11434/v1", "api_key_env": None,
                 "default_model": "qwen2.5-coder", "native": "http://localhost:11434"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "api_key_env": None,
                 "default_model": "local-model", "native": None},
    "openai":   {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
                 "default_model": "gpt-4o-mini", "native": None},
    "ai-local": {"base_url": "http://localhost:11434/v1", "api_key_env": None,
                 "default_model": "chat_vi", "native": "http://localhost:11434"},
}


# ─── Đọc file .env (không cần thư viện ngoài) ─────────────────────────────────

def load_env(path: str = ".env") -> dict:
    """Đọc .env vào os.environ (không ghi đè biến đã có). Trả về dict đã đọc."""
    values = {}
    if not os.path.isfile(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                values[key] = val
                os.environ.setdefault(key, val)
    except OSError as e:
        logger.warning("Không đọc được %s: %s", path, e)
    return values


# ─── LLM Client ───────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Lỗi khi gọi LLM."""


class LLMClient:
    """Client gọi LLM qua OpenAI-compatible API, dùng stdlib urllib."""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 provider: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.provider = provider
        self.timeout = timeout

    @classmethod
    def from_env(cls, provider: str = "", model: str = "", base_url: str = "",
                 api_key: str = "") -> "LLMClient":
        """Tạo client từ biến môi trường (đã load_env trước đó nếu cần)."""
        provider = provider or os.environ.get("LLM_PROVIDER", "ollama")
        meta = PROVIDERS.get(provider, PROVIDERS["ollama"])

        base_url = base_url or os.environ.get("LLM_BASE_URL") or meta["base_url"]
        model = model or os.environ.get("LLM_MODEL") or meta["default_model"]

        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")
            if not api_key and meta.get("api_key_env"):
                api_key = os.environ.get(meta["api_key_env"], "")

        return cls(base_url=base_url, model=model, api_key=api_key, provider=provider)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: list, tools: list = None, temperature: float = 0.2,
             max_tokens: int = 1024, stream: bool = False, **kwargs) -> dict:
        """
        Gọi chat completion (non-streaming). Trả về response dict OpenAI-style.
        """
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {e.code} từ {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"Không kết nối được {url}: {e.reason}. "
                f"Kiểm tra Ollama/LM Studio đã chạy chưa (provider={self.provider})."
            ) from e
        except json.JSONDecodeError as e:
            raise LLMError(f"Phản hồi không phải JSON từ {url}") from e

    def chat_text(self, messages: list, **kwargs) -> str:
        """Tiện ích: gọi chat và trả về text câu trả lời."""
        resp = self.chat(messages, **kwargs)
        try:
            return resp["choices"][0]["message"].get("content", "") or ""
        except (KeyError, IndexError):
            return ""

    def chat_stream(self, messages: list, temperature: float = 0.2,
                    max_tokens: int = 1024, **kwargs):
        """
        Generator stream từng mảnh text (SSE OpenAI-style).
        """
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0].get("delta", {})
                        piece = delta.get("content", "")
                        if piece:
                            yield piece
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except urllib.error.URLError as e:
            raise LLMError(f"Không kết nối được {url}: {getattr(e, 'reason', e)}") from e

    def embed(self, inputs, model: str = "") -> list:
        """
        Tạo embedding cho text qua endpoint OpenAI-compatible (POST /embeddings).
        - inputs: str hoặc list[str]
        - model: tên model embedding (vd 'nomic-embed-text'); mặc định lấy
          LLM_EMBED_MODEL từ env, nếu không có thì dùng self.model.
        Trả về list[list[float]] theo thứ tự input. Raise LLMError nếu lỗi.
        """
        single = isinstance(inputs, str)
        items = [inputs] if single else list(inputs)
        emb_model = model or os.environ.get("LLM_EMBED_MODEL", "") or self.model
        payload = {"model": emb_model, "input": items}
        url = f"{self.base_url}/embeddings"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            vecs = [d["embedding"] for d in body.get("data", [])]
            if not vecs:
                raise LLMError(f"Không nhận được embedding từ {url}")
            return vecs[0] if single else vecs
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(
                f"HTTP {e.code} từ {url}: {detail}. "
                f"Cần model embedding (vd: ollama pull nomic-embed-text) và đặt LLM_EMBED_MODEL."
            ) from e
        except urllib.error.URLError as e:
            raise LLMError(f"Không kết nối được {url}: {getattr(e, 'reason', e)}") from e
        except (KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"Phản hồi embedding không hợp lệ từ {url}: {e}") from e

    def list_models(self) -> list:
        """Liệt kê model có sẵn (GET /models). Trả về list tên."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return [m.get("id", "") for m in data.get("data", [])]
        except Exception as e:
            logger.warning("Không liệt kê được model: %s", e)
            return []


def describe_config() -> str:
    """Mô tả cấu hình LLM hiện tại (che key)."""
    client = LLMClient.from_env()
    key_status = "có" if client.api_key else "không"
    return (
        f"Provider: {client.provider}\n"
        f"Base URL: {client.base_url}\n"
        f"Model:    {client.model}\n"
        f"API key:  {key_status}"
    )


if __name__ == "__main__":
    load_env()
    print(describe_config())
