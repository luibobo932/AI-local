"""
AI-local CLI — Ollama-compatible command line interface.

Commands:
  serve               Start the REST API server
  run   <model>       Interactive chat with a model
  list                List available models
  show  <model>       Show model details
  ps                  Show currently loaded models (from running server)
  pull  <stage>       Run training pipeline for a stage
  rm    <model>       Delete a model checkpoint

Usage:
    python cli.py serve
    python cli.py run dpo_ckpt
    python cli.py list
    python cli.py show ckpt
    python cli.py pull sft
    python cli.py rm dpo_ckpt
"""

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time


CHECKPOINTS_DIR = "checkpoints"
MODELS_DIR = "models"
DATA_DIR = "data"
SERVER_URL = "http://127.0.0.1:11434"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _discover_models() -> list[str]:
    paths = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pt"))
    names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(paths)]
    # Model fine-tune lưu trong models/<name>/ (thư mục có config.json)
    if os.path.isdir(MODELS_DIR):
        for d in sorted(os.listdir(MODELS_DIR)):
            if os.path.exists(os.path.join(MODELS_DIR, d, "config.json")):
                names.append(d)
    return names


def _sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _load_ckpt_meta(name: str) -> dict:
    import torch
    path = os.path.join(CHECKPOINTS_DIR, name + ".pt")
    if not os.path.exists(path):
        path = os.path.join(CHECKPOINTS_DIR, name)
    if not os.path.exists(path):
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_cfg")
        size = os.path.getsize(path)
        params = sum(p.numel() for p in __import__("model.gpt", fromlist=["GPT"]).GPT(cfg).parameters()) if cfg else 0
        return {
            "stage": ckpt.get("stage", "pretrain"),
            "iter_num": ckpt.get("iter_num", 0),
            "best_val_loss": ckpt.get("best_val_loss", 0.0),
            "size": size,
            "params": params,
            "cfg": cfg,
        }
    except Exception:
        return {}


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_serve(args):
    """Start the REST API server."""
    cmd = [
        sys.executable, "server.py",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.model:
        cmd += ["--model", args.model]
    print(f"{GREEN}Starting AI-local server on http://{args.host}:{args.port}{RESET}")
    os.execv(sys.executable, cmd)


def cmd_list(args):
    """List available model checkpoints."""
    models = _discover_models()
    if not models:
        print("No models found.")
        print("Run training pipeline:")
        print("  python train.py && python finetune_sft.py && python align_dpo.py")
        return

    print(f"\n{'NAME':<25} {'STAGE':<12} {'PARAMS':<12} {'SIZE':<10} {'VAL LOSS'}")
    print("─" * 72)
    for name in models:
        meta = _load_ckpt_meta(name)
        stage = meta.get("stage", "?")
        params = f"{meta.get('params', 0)/1e6:.1f}M" if meta.get("params") else "?"
        size = _sizeof_fmt(meta.get("size", 0)) if meta.get("size") else "?"
        val_loss = f"{meta.get('best_val_loss', 0):.4f}" if meta.get("best_val_loss") else "?"
        print(f"{name:<25} {stage:<12} {params:<12} {size:<10} {val_loss}")
    print()


def cmd_show(args):
    """Show detailed model info."""
    import torch
    name = args.model
    path = os.path.join(CHECKPOINTS_DIR, name + ".pt")
    if not os.path.exists(path):
        path = os.path.join(CHECKPOINTS_DIR, name)
    if not os.path.exists(path):
        print(f"Model '{name}' not found.")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg")
    size = os.path.getsize(path)

    print(f"\n{BOLD}Model: {name}{RESET}")
    print(f"  Path:       {path}")
    print(f"  Size:       {_sizeof_fmt(size)}")
    print(f"  Stage:      {ckpt.get('stage', 'pretrain')}")
    print(f"  Iterations: {ckpt.get('iter_num', 0)}")
    print(f"  Val loss:   {ckpt.get('best_val_loss', 0):.4f}")

    if cfg:
        print(f"\n{BOLD}Architecture:{RESET}")
        for k, v in vars(cfg).items():
            print(f"  {k:<20} {v}")

        from model.gpt import GPT
        params = sum(p.numel() for p in GPT(cfg).parameters())
        print(f"\n  {'parameters':<20} {params/1e6:.2f}M  ({params:,})")

    if cfg:
        from model.gpt import GPT
        dpo_beta = ckpt.get("dpo_beta")
        if dpo_beta:
            print(f"\n{BOLD}DPO:{RESET}")
            print(f"  beta: {dpo_beta}")
    print()


def cmd_run(args):
    """Interactive chat with a model — like `ollama run`."""
    import torch
    from model.hf_backend import is_hf_model, HFModel

    name = args.model

    # HuggingFace model
    if name and is_hf_model(name):
        model = HFModel(name)
        stage = "hf"
        params = sum(p.numel() for p in model.parameters())

        def encode(text): return model.encode(text)
        def decode(ids): return model.decode(ids)

        print(f"\n{GREEN}AI-local{RESET} — {name}  (HuggingFace, {params/1e6:.0f}M params)")
        print(f"Type your message, /bye to exit, /clear to reset, /set temp <value>\n")
        _run_chat_loop(model, encode, decode, args)
        return

    # Local checkpoint
    path = os.path.join(CHECKPOINTS_DIR, name + ".pt")
    if not os.path.exists(path):
        path = os.path.join(CHECKPOINTS_DIR, name)
    if not os.path.exists(path):
        models = _discover_models()
        if not models:
            print("No models found. Run: python cli.py pull pretrain")
            return
        name = models[-1]
        path = os.path.join(CHECKPOINTS_DIR, name + ".pt")
        print(f"Using latest model: {name}")

    from model.gpt import GPT, GPTConfig

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading {name} ...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg: GPTConfig = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    def encode(text):
        if meta:
            stoi = meta["stoi"]
            fallback = stoi.get(" ", 0)
            return [stoi.get(c, fallback) for c in text]
        import tiktoken
        return tiktoken.get_encoding("gpt2").encode(text)

    def decode(ids):
        if meta:
            itos = meta["itos"]
            return "".join(itos.get(i, "") for i in ids)
        import tiktoken
        return tiktoken.get_encoding("gpt2").decode(ids)

    stage = ckpt.get("stage", "pretrain")
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{GREEN}AI-local{RESET} — {name}  ({stage}, {params/1e6:.1f}M params)")
    print(f"Type your message, /bye to exit, /clear to reset, /set temp <value>\n")
    _run_chat_loop(model, encode, decode, args)


def _run_chat_loop(model, encode, decode, args):
    """Shared interactive chat loop for both local and HuggingFace models."""
    import torch

    device = next(model.parameters()).device
    temperature = args.temperature
    top_k = args.top_k
    history = []

    while True:
        try:
            user_input = input(f"{CYAN}>>> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/bye":
            print("Bye!")
            break
        if user_input == "/clear":
            history.clear()
            print("History cleared.")
            continue
        if user_input.startswith("/set temp "):
            try:
                temperature = float(user_input.split()[-1])
                print(f"Temperature set to {temperature}")
            except ValueError:
                print("Usage: /set temp <float>")
            continue
        if user_input.startswith("/set top_k "):
            try:
                top_k = int(user_input.split()[-1])
                print(f"top_k set to {top_k}")
            except ValueError:
                print("Usage: /set top_k <int>")
            continue
        if user_input == "/help":
            print("/bye    exit  |  /clear  clear history  |  /set temp <f>  |  /set top_k <n>")
            continue

        history.append({"role": "user", "content": user_input})
        prompt_parts = []
        for msg in history:
            if msg["role"] == "user":
                prompt_parts.append(f"Human: {msg['content']}")
            else:
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append("Assistant: ")
        prompt_text = "\n\n".join(prompt_parts)

        tokens = encode(prompt_text)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        print(f"{YELLOW}", end="", flush=True)
        response_chars = []
        with torch.no_grad():
            for token_id in model.generate_iter(
                idx, args.max_tokens, temperature=temperature, top_k=top_k
            ):
                char = decode([token_id])
                # EOS — model báo hiệu kết thúc câu trả lời
                if "■" in char:
                    break
                response_chars.append(char)
                print(char, end="", flush=True)

                if "".join(response_chars[-4:]) == "\n\n\n\n":
                    break

        print(f"{RESET}")
        response_text = "".join(response_chars).strip()
        history.append({"role": "assistant", "content": response_text})


def cmd_pull(args):
    """Download a HuggingFace model OR run training pipeline for a stage."""
    stage = args.stage.lower()

    # HuggingFace model download
    from model.hf_backend import is_hf_model
    if is_hf_model(stage) or stage.startswith("hf:"):
        model_id = stage.removeprefix("hf:")
        _pull_hf_model(model_id)
        return

    stages = {
        "pretrain": [
            ("Preparing data", [sys.executable, "data/prepare.py"]),
            ("Training (Stage 1: Pre-training)", [sys.executable, "train.py"]),
        ],
        "sft": [
            ("Preparing SFT data", [sys.executable, "data/prepare_sft.py"]),
            ("Fine-tuning (Stage 2: SFT)", [sys.executable, "finetune_sft.py"]),
        ],
        "dpo": [
            ("Preparing DPO data", [sys.executable, "data/prepare_dpo.py"]),
            ("Aligning (Stage 3: DPO)", [sys.executable, "align_dpo.py"]),
        ],
        "all": [
            ("Preparing data", [sys.executable, "data/prepare.py"]),
            ("Training (Stage 1: Pre-training)", [sys.executable, "train.py"]),
            ("Preparing SFT data", [sys.executable, "data/prepare_sft.py"]),
            ("Fine-tuning (Stage 2: SFT)", [sys.executable, "finetune_sft.py"]),
            ("Preparing DPO data", [sys.executable, "data/prepare_dpo.py"]),
            ("Aligning (Stage 3: DPO)", [sys.executable, "align_dpo.py"]),
        ],
    }

    if stage not in stages:
        print(f"Unknown stage '{stage}'.")
        print("Training stages: pretrain, sft, dpo, all")
        print("HuggingFace models: gpt2, gpt2-medium, distilgpt2, or any repo/model-id")
        return

    for label, cmd in stages[stage]:
        print(f"\n{GREEN}▶ {label}{RESET}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"{YELLOW}Stage failed: {label}{RESET}")
            sys.exit(1)
    print(f"\n{GREEN}✓ '{stage}' pipeline complete.{RESET}")


def _pull_hf_model(model_id: str):
    """Download and cache a HuggingFace model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print(f"{GREEN}Pulling {model_id} from HuggingFace...{RESET}")
    print("(model will be cached in ~/.cache/huggingface/)\n")
    try:
        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(model_id)
        print("Downloading model weights...")
        AutoModelForCausalLM.from_pretrained(model_id)
        print(f"\n{GREEN}✓ {model_id} downloaded successfully.{RESET}")
        print(f"\nRun it:")
        print(f"  python cli.py run {model_id}")
        print(f"  python cli.py serve  →  use model '{model_id}' in API requests")
    except Exception as e:
        print(f"{YELLOW}Download failed: {e}{RESET}")
        sys.exit(1)


def cmd_finetune(args):
    """Fine-tune a pre-trained model on your own data — 'make it your own'."""
    cmd = [
        sys.executable, "finetune_hf.py",
        "--base", args.base,
        "--out", args.out,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.data:
        cmd += ["--data", args.data]
    if args.lora:
        cmd.append("--lora")
    if args.merge:
        cmd.append("--merge")
    print(f"{GREEN}▶ Fine-tuning '{args.base}' → '{args.out}'{RESET}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_rm(args):
    """Delete a model checkpoint or fine-tuned model directory."""
    import shutil
    name = args.model

    # Fine-tuned model directory?
    mdir = os.path.join(MODELS_DIR, name)
    if os.path.isdir(mdir):
        confirm = input(f"Delete fine-tuned model '{name}' (dir {mdir})? [y/N] ").strip().lower()
        if confirm == "y":
            shutil.rmtree(mdir)
            print(f"Deleted {mdir}")
        else:
            print("Cancelled.")
        return

    path = os.path.join(CHECKPOINTS_DIR, name + ".pt")
    if not os.path.exists(path):
        path = os.path.join(CHECKPOINTS_DIR, name)
    if not os.path.exists(path):
        print(f"Model '{name}' not found.")
        return

    confirm = input(f"Delete '{name}'? [y/N] ").strip().lower()
    if confirm == "y":
        os.remove(path)
        print(f"Deleted {path}")
    else:
        print("Cancelled.")


def cmd_ps(args):
    """Show server status (if running)."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{SERVER_URL}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        models = data.get("models", [])
        if models:
            print(f"\n{'NAME':<25} {'STAGE':<12} {'PARAMS'}")
            print("─" * 50)
            for m in models:
                d = m.get("details", {})
                print(f"{m['name']:<25} {d.get('stage','?'):<12} {d.get('parameters','?')}")
        else:
            print("Server running — no models loaded yet.")
    except Exception:
        print(f"Server not running at {SERVER_URL}")
        print("Start with: python cli.py serve")


def cmd_tools(args):
    """Liệt kê các tools có sẵn cho agent."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{SERVER_URL}/v1/tools", timeout=3) as r:
            data = json.loads(r.read())
        tools = data.get("tools", [])
        print(f"\n{BOLD}Tools có sẵn ({len(tools)}){RESET}")
        print("─" * 60)
        for t in tools:
            print(f"  {CYAN}{t['name']:<20}{RESET}  {t['description']}")
        print()
    except Exception:
        # Fallback: đọc trực tiếp từ module
        try:
            from tools import TOOL_SCHEMAS
            print(f"\n{BOLD}Tools có sẵn ({len(TOOL_SCHEMAS)}){RESET}")
            print("─" * 60)
            for s in TOOL_SCHEMAS:
                fn = s["function"]
                print(f"  {CYAN}{fn['name']:<20}{RESET}  {fn['description']}")
            print()
        except ImportError as e:
            print(f"{YELLOW}Không load được tools: {e}{RESET}")


def cmd_agent(args):
    """Chạy agent hoàn thành nhiệm vụ với tool use."""
    import urllib.request

    # Kiểm tra server
    try:
        urllib.request.urlopen(f"{SERVER_URL}/api/version", timeout=2)
    except Exception:
        print(f"{YELLOW}Server chưa chạy. Khởi động bằng: python cli.py serve{RESET}")
        sys.exit(1)

    task = " ".join(args.task) if isinstance(args.task, list) else args.task
    model = args.model or ""

    tools = [t.strip() for t in args.tools.split(",")] if args.tools else None

    print(f"\n{BOLD}🤖 Agent đang chạy...{RESET}")
    print(f"Task: {task}")
    if tools:
        print(f"Tools: {', '.join(tools)}")
    if model:
        print(f"Model: {model}")
    print(f"Mode: {args.mode}")
    print("─" * 60)

    payload = json.dumps({
        "task": task,
        "model": model,
        "tools": tools,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/v1/agent",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
    except Exception as e:
        print(f"{YELLOW}Lỗi: {e}{RESET}")
        sys.exit(1)

    if not result.get("success"):
        print(f"{YELLOW}Lỗi agent: {result.get('error')}{RESET}")
        sys.exit(1)

    # In trace nếu verbose
    if args.verbose:
        steps = result.get("steps", [])
        for s in steps:
            print(f"\n{BOLD}── Bước {s['step']} ──{RESET}")
            if s.get("thought"):
                print(f"💭 {s['thought']}")
            for tc in s.get("tool_calls", []):
                args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                print(f"🔧 {CYAN}{tc['name']}{RESET}({args_str})")
                res_preview = tc.get("result", "")[:300]
                if res_preview:
                    print(f"   → {res_preview}")

    # In câu trả lời
    answer = result.get("answer", "")
    elapsed = result.get("elapsed", 0)
    n_steps = len(result.get("steps", []))

    print(f"\n{GREEN}{'─'*60}{RESET}")
    print(f"{BOLD}✅ Hoàn thành{RESET} ({result.get('model')}, {n_steps} bước, {elapsed:.1f}s)")
    print()
    print(answer)
    print()


def cmd_mcp(args):
    """Quản lý MCP servers."""
    import urllib.request

    try:
        urllib.request.urlopen(f"{SERVER_URL}/api/version", timeout=2)
    except Exception:
        print(f"{YELLOW}Server chưa chạy.{RESET}")
        sys.exit(1)

    if args.mcp_cmd == "list":
        with urllib.request.urlopen(f"{SERVER_URL}/v1/mcp/servers", timeout=5) as r:
            data = json.loads(r.read())
        servers = data.get("servers", [])
        if not servers:
            print("Chưa kết nối MCP server nào.")
            print("Thêm: python cli.py mcp add <name> <url>")
            return
        print(f"\n{BOLD}MCP Servers ({len(servers)}){RESET}")
        print("─" * 60)
        for s in servers:
            status = f"{GREEN}●{RESET}" if s["connected"] else f"{YELLOW}✗{RESET}"
            print(f"  {status} {s['name']:<20} {s['url']}")
            if s["connected"]:
                print(f"     Tools: {', '.join(t['name'] for t in s['tools'][:5])}")
            elif s.get("error"):
                print(f"     {YELLOW}Lỗi: {s['error']}{RESET}")

    elif args.mcp_cmd == "add":
        payload = json.dumps({"name": args.name, "url": args.url}).encode()
        req = urllib.request.Request(
            f"{SERVER_URL}/v1/mcp/servers",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("connected"):
            print(f"{GREEN}✅ Kết nối thành công: {args.name} ({data['tools_count']} tools){RESET}")
        else:
            print(f"{YELLOW}Kết nối thất bại: {data.get('error')}{RESET}")

    elif args.mcp_cmd == "remove":
        req = urllib.request.Request(
            f"{SERVER_URL}/v1/mcp/servers/{args.name}",
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        print(f"Đã xóa: {data.get('removed')}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="AI-local — Ollama-compatible local LLM CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=11434)
    p_serve.add_argument("--model", default="", help="Pre-load this model")
    p_serve.set_defaults(func=cmd_serve)

    # run
    p_run = sub.add_parser("run", help="Interactive chat with a model")
    p_run.add_argument("model", nargs="?", default="", help="Model name (checkpoint without .pt)")
    p_run.add_argument("--temperature", type=float, default=0.8)
    p_run.add_argument("--top_k", type=int, default=50)
    p_run.add_argument("--max_tokens", type=int, default=200)
    p_run.set_defaults(func=cmd_run)

    # list
    p_list = sub.add_parser("list", help="List available models")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="Show model details")
    p_show.add_argument("model", help="Model name")
    p_show.set_defaults(func=cmd_show)

    # ps
    p_ps = sub.add_parser("ps", help="Show server status")
    p_ps.set_defaults(func=cmd_ps)

    # pull
    p_pull = sub.add_parser("pull", help="Download HuggingFace model or run training pipeline")
    p_pull.add_argument("stage",
                        help="Training stage (pretrain/sft/dpo/all) OR HuggingFace model id (gpt2, distilgpt2, ...)")
    p_pull.set_defaults(func=cmd_pull)

    # finetune
    p_ft = sub.add_parser("finetune", help="Fine-tune a pre-trained model on your own data")
    p_ft.add_argument("--base", default="gpt2", help="Base model (gpt2, distilgpt2, repo/id, models/<dir>)")
    p_ft.add_argument("--data", default="", help="Your JSONL data ({instruction, response}). Empty = built-in")
    p_ft.add_argument("--out", default="my-model", help="Output model name (saved to models/<out>/)")
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--batch_size", type=int, default=4)
    p_ft.add_argument("--learning_rate", type=float, default=2e-5)
    p_ft.add_argument("--lora", action="store_true", help="Use LoRA (lighter, for big models)")
    p_ft.add_argument("--merge", action="store_true", help="Merge LoRA into base when saving")
    p_ft.set_defaults(func=cmd_finetune)

    # rm
    p_rm = sub.add_parser("rm", help="Delete a model checkpoint")
    p_rm.add_argument("model", help="Model name")
    p_rm.set_defaults(func=cmd_rm)

    # tools
    p_tools = sub.add_parser("tools", help="Liệt kê các tools có sẵn cho agent")
    p_tools.set_defaults(func=cmd_tools)

    # agent
    p_agent = sub.add_parser("agent", help="Chạy agent để hoàn thành nhiệm vụ với tool use")
    p_agent.add_argument("task", nargs="+", help="Nhiệm vụ cần thực hiện")
    p_agent.add_argument("--model", default="", help="Model (mặc định: model đang chạy)")
    p_agent.add_argument("--tools", default="", help="Danh sách tools cách nhau dấu phẩy (mặc định: tất cả)")
    p_agent.add_argument("--max-steps", type=int, default=10, dest="max_steps")
    p_agent.add_argument("--temperature", type=float, default=0.2)
    p_agent.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p_agent.add_argument("--mode", choices=["auto", "react", "function_calling"], default="auto")
    p_agent.add_argument("--verbose", "-v", action="store_true", help="Hiển thị từng bước chi tiết")
    p_agent.set_defaults(func=cmd_agent)

    # mcp
    p_mcp = sub.add_parser("mcp", help="Quản lý MCP servers")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_cmd", required=True)

    p_mcp_list = mcp_sub.add_parser("list", help="Liệt kê MCP servers")
    p_mcp_add = mcp_sub.add_parser("add", help="Thêm MCP server")
    p_mcp_add.add_argument("name", help="Tên server")
    p_mcp_add.add_argument("url", help="URL server")
    p_mcp_rm = mcp_sub.add_parser("remove", help="Xóa MCP server")
    p_mcp_rm.add_argument("name", help="Tên server")
    p_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
