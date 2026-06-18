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
import time
from datetime import datetime, timezone
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model.gpt import GPT, GPTConfig
from model.hf_backend import HFModel, is_hf_model

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-local", description="Ollama-compatible local LLM server")

# ─── Model registry (loaded on demand, cached in memory) ─────────────────────

_model_cache: dict[str, tuple] = {}   # name → (model, meta_or_None)
_data_dir = "data"
_checkpoints_dir = "checkpoints"
_default_model = ""


def _discover_models() -> list[str]:
    """Return checkpoint names + any cached HF models."""
    paths = glob.glob(os.path.join(_checkpoints_dir, "*.pt"))
    local = [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]
    hf_cached = [n for n in _model_cache if is_hf_model(n)]
    return local + [n for n in hf_cached if n not in local]


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


def _ckpt_info(name: str) -> dict:
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
    stop_seqs = stop or []
    generated_text = ""
    eval_count = 0

    t_gen_start = time.time()
    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta, model)
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
    stop_seqs = stop or []
    generated_text = ""
    eval_count = 0

    t_gen_start = time.time()
    for token_id in model.generate_iter(idx, max_tokens, temperature=temperature, top_k=top_k):
        char = _decode([token_id], meta, model)
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

    messages = [m.model_dump() for m in req.messages]
    temperature = req.options.get("temperature", req.temperature)
    top_k = req.options.get("top_k", req.top_k)
    max_tokens = req.options.get("num_predict", req.max_tokens)
    stop = req.stop or req.options.get("stop", [])

    try:
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


@app.get("/")
async def root():
    return "Ollama is running"


@app.head("/")
async def root_head():
    return {}


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
