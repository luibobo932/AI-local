"""
AI-local REST API Server — Ollama-compatible API.

Endpoints:
  POST /api/generate              — text completion (streaming/non-streaming)
  POST /api/chat                  — chat completion (streaming/non-streaming)
  GET  /api/tags                  — list available models
  POST /api/show                  — model info
  DELETE /api/delete              — delete a checkpoint

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
import time
from datetime import datetime, timezone
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model.gpt import GPT, GPTConfig

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-local", description="Ollama-compatible local LLM server")

# ─── Model registry (loaded on demand, cached in memory) ─────────────────────

_model_cache: dict[str, tuple[GPT, dict | None]] = {}
_data_dir = "data"
_checkpoints_dir = "checkpoints"
_default_model = ""


def _discover_models() -> list[str]:
    """Return list of checkpoint names (without .pt extension)."""
    paths = glob.glob(os.path.join(_checkpoints_dir, "*.pt"))
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]


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


def _load_model(model_name: str) -> tuple[GPT, dict | None]:
    """Load (and cache) a model checkpoint."""
    if model_name in _model_cache:
        return _model_cache[model_name]

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


def _encode(text: str, meta: dict | None) -> list[int]:
    if meta:
        stoi = meta["stoi"]
        fallback = stoi.get(" ", 0)
        return [stoi.get(c, fallback) for c in text]
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text)


def _decode(ids: list[int], meta: dict | None) -> str:
    if meta:
        itos = meta["itos"]
        return "".join(itos.get(i, "") for i in ids)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(ids)


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


def _ckpt_info(name: str) -> dict:
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


class ShowRequest(BaseModel):
    model: str = ""
    name: str = ""


class DeleteRequest(BaseModel):
    model: str = ""
    name: str = ""


# ─── Ollama-compatible endpoints ──────────────────────────────────────────────

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


async def _stream_generate(
    model_name: str, prompt: str, temperature: float, top_k: int, max_tokens: int
) -> AsyncIterator[str]:
    """Async generator that yields Ollama-style NDJSON chunks."""
    model, meta = _load_model(model_name)
    device = next(model.parameters()).device

    tokens = _encode(prompt, meta)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    created_at = datetime.now(tz=timezone.utc).isoformat()

    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta)
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "response": char,
            "done": False,
        }
        yield json.dumps(chunk, ensure_ascii=False) + "\n"
        await asyncio.sleep(0)  # yield control to event loop

    done_chunk = {
        "model": model_name,
        "created_at": created_at,
        "response": "",
        "done": True,
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

    try:
        if req.stream:
            return StreamingResponse(
                _stream_generate(model_name, req.prompt, temperature, top_k, max_tokens),
                media_type="application/x-ndjson",
            )
        else:
            # Collect full response
            full = ""
            async for chunk in _stream_generate(model_name, req.prompt, temperature, top_k, max_tokens):
                data = json.loads(chunk)
                full += data.get("response", "")
            return {
                "model": model_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "response": full,
                "done": True,
            }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


async def _stream_chat(
    model_name: str, messages: list[dict], temperature: float, top_k: int, max_tokens: int
) -> AsyncIterator[str]:
    """Async generator for chat streaming — Ollama-style."""
    model, meta = _load_model(model_name)
    device = next(model.parameters()).device

    prompt = _format_prompt_chat(messages)
    tokens = _encode(prompt, meta)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    created_at = datetime.now(tz=timezone.utc).isoformat()
    full_response = ""

    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta)
        full_response += char
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": char},
            "done": False,
        }
        yield json.dumps(chunk, ensure_ascii=False) + "\n"
        await asyncio.sleep(0)

    done_chunk = {
        "model": model_name,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
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

    messages = [m.model_dump() for m in req.messages]
    temperature = req.options.get("temperature", req.temperature)
    top_k = req.options.get("top_k", req.top_k)
    max_tokens = req.options.get("num_predict", req.max_tokens)

    try:
        if req.stream:
            return StreamingResponse(
                _stream_chat(model_name, messages, temperature, top_k, max_tokens),
                media_type="application/x-ndjson",
            )
        else:
            full = ""
            async for chunk in _stream_chat(model_name, messages, temperature, top_k, max_tokens):
                data = json.loads(chunk)
                full += data.get("message", {}).get("content", "")
            return {
                "model": model_name,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": full},
                "done": True,
            }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# ─── OpenAI-compatible endpoints ─────────────────────────────────────────────

class OAIChatMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = ""
    messages: list[OAIChatMessage] = []
    stream: bool = False
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 256
    n: int = 1


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
    tokens = _encode(prompt, meta)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta)
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


@app.post("/v1/chat/completions")
async def oai_chat_completions(req: OAIChatRequest):
    """OpenAI-compatible chat completions."""
    model_name = req.model or _default_model
    if not model_name:
        models = _discover_models()
        if not models:
            raise HTTPException(404, "No models found.")
        model_name = models[-1]

    messages = [m.model_dump() for m in req.messages]

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

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(full), "total_tokens": len(full)},
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


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


@app.get("/")
async def root():
    return {"status": "ok", "name": "ai-local", "version": "0.1.0"}


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

    print(f"AI-local server starting on http://{args.host}:{args.port}")
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
