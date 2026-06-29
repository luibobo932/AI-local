#!/usr/bin/env python3
"""
Minion — AI coding agent local, chạy được trên máy cá nhân.

KHÔNG cần torch/fastapi. Chỉ dùng stdlib + một LLM endpoint
(Ollama / LM Studio / OpenAI-compatible).

Lệnh:
    python minion.py providers              # xem cấu hình LLM hiện tại
    python minion.py models                 # liệt kê model của provider
    python minion.py chat                    # chat tương tác (streaming)
    python minion.py agent "nhiệm vụ"        # chạy agent với tool use
    python minion.py agent "..." --plan -v   # chế độ chỉ-đọc, hiện từng bước
    python minion.py serve --port 8000       # HTTP API: POST /chat, /agent/run

Cấu hình qua .env (xem .env.example) hoặc biến môi trường:
    LLM_PROVIDER=ollama
    LLM_MODEL=qwen2.5-coder
"""

import argparse
import json
import logging
import os
import sys

from llm import LLMClient, LLMError, describe_config, load_env

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    os.makedirs(".ai-local", exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]
    try:
        handlers.append(logging.FileHandler(os.path.join(".ai-local", "minion.log"), encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger("minion")

GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; RESET = "\033[0m"; BOLD = "\033[1m"


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_providers(args):
    print(describe_config())


def cmd_models(args):
    client = LLMClient.from_env(provider=args.provider, model=args.model)
    print(f"Provider: {client.provider}  ({client.base_url})")
    models = client.list_models()
    if not models:
        print(f"{YELLOW}Không lấy được danh sách model. "
              f"Kiểm tra {client.provider} đã chạy chưa.{RESET}")
        return
    print(f"\n{BOLD}Model có sẵn ({len(models)}):{RESET}")
    for m in models:
        print(f"  - {m}")


def cmd_chat(args):
    client = LLMClient.from_env(provider=args.provider, model=args.model)
    print(f"{BOLD}💬 Chat với {client.model}{RESET} (provider: {client.provider})")
    print(f"{CYAN}Gõ 'exit' hoặc Ctrl+C để thoát.{RESET}\n")
    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    try:
        while True:
            try:
                user = input(f"{GREEN}Bạn> {RESET}").strip()
            except EOFError:
                break
            if not user or user.lower() in ("exit", "quit"):
                break
            history.append({"role": "user", "content": user})
            print(f"{CYAN}AI> {RESET}", end="", flush=True)
            answer = ""
            try:
                for piece in client.chat_stream(history, temperature=args.temperature,
                                                 max_tokens=args.max_tokens):
                    print(piece, end="", flush=True)
                    answer += piece
            except LLMError as e:
                print(f"\n{YELLOW}Lỗi: {e}{RESET}")
                history.pop()
                continue
            print("\n")
            history.append({"role": "assistant", "content": answer})
    except KeyboardInterrupt:
        print("\nTạm biệt!")


def cmd_agent(args):
    from agent import run_agent

    task = " ".join(args.task)
    permission_mode = "plan" if args.plan else args.permission_mode
    tools = [t.strip() for t in args.tools.split(",")] if args.tools else None

    print(f"{BOLD}🤖 Minion agent{RESET}")
    print(f"Task: {task}")
    print(f"Permission: {permission_mode}  ·  Skill: {args.skill or '(none)'}")
    print("─" * 60)

    def _on_step(step):
        if not args.verbose:
            return
        icon = "✅" if step.is_final else ("🔧" if step.tool_calls else "💭")
        print(f"\n{icon} Bước {step.step}")
        if step.thought:
            print(f"   💭 {step.thought[:200]}")
        for tc in step.tool_calls:
            print(f"   🔧 {CYAN}{tc.name}{RESET}({json.dumps(tc.arguments, ensure_ascii=False)[:120]})")
            print(f"      → {tc.result[:200]}")

    try:
        result = run_agent(
            task=task,
            provider=args.provider,
            model=args.model,
            tools=tools,
            skill=args.skill,
            max_steps=args.max_steps,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            permission_mode=permission_mode,
            on_step=_on_step,
        )
    except Exception as e:
        print(f"{YELLOW}Lỗi agent: {e}{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}{'─'*60}{RESET}")
    if not result.success:
        print(f"{YELLOW}❌ {result.error}{RESET}")
        sys.exit(1)
    print(f"{BOLD}✅ Hoàn thành{RESET} ({result.model}, {len(result.steps)} bước, {result.elapsed:.1f}s)\n")
    print(result.answer)


def cmd_serve(args):
    """HTTP API stdlib: POST /chat, POST /agent/run, GET /health."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    client_cfg = {"provider": args.provider, "model": args.model}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            logger.info("%s - %s", self.address_string(), fmt % a)

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "service": "minion"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/chat":
                    self._handle_chat()
                elif self.path == "/agent/run":
                    self._handle_agent()
                else:
                    self._send(404, {"error": "not found", "paths": ["/chat", "/agent/run", "/health"]})
            except Exception as e:
                logger.exception("Lỗi xử lý request")
                self._send(500, {"error": str(e)})

        def _handle_chat(self):
            data = self._body()
            messages = data.get("messages")
            if not messages and data.get("prompt"):
                messages = [{"role": "user", "content": data["prompt"]}]
            if not messages:
                return self._send(400, {"error": "thiếu 'messages' hoặc 'prompt'"})
            client = LLMClient.from_env(
                provider=data.get("provider") or client_cfg["provider"],
                model=data.get("model") or client_cfg["model"],
            )
            try:
                text = client.chat_text(
                    messages,
                    temperature=data.get("temperature", 0.2),
                    max_tokens=data.get("max_tokens", 1024),
                )
            except LLMError as e:
                return self._send(502, {"error": str(e)})
            self._send(200, {"model": client.model, "message": {"role": "assistant", "content": text}})

        def _handle_agent(self):
            from agent import run_agent, step_to_dict
            data = self._body()
            task = data.get("task", "")
            if not task:
                return self._send(400, {"error": "thiếu 'task'"})
            try:
                result = run_agent(
                    task=task,
                    provider=data.get("provider") or client_cfg["provider"],
                    model=data.get("model") or client_cfg["model"],
                    tools=data.get("tools"),
                    skill=data.get("skill", ""),
                    max_steps=data.get("max_steps", 10),
                    temperature=data.get("temperature", 0.2),
                    max_tokens=data.get("max_tokens", 1024),
                    permission_mode=data.get("permission_mode", "auto"),
                )
            except Exception as e:
                logger.exception("agent lỗi")
                return self._send(500, {"error": str(e)})
            self._send(200, {
                "answer": result.answer,
                "model": result.model,
                "success": result.success,
                "error": result.error,
                "elapsed": result.elapsed,
                "steps": [step_to_dict(s) for s in result.steps],
            })

    addr = (args.host, args.port)
    httpd = ThreadingHTTPServer(addr, Handler)
    cfg = LLMClient.from_env(provider=args.provider, model=args.model)
    print(f"{BOLD}🚀 Minion API chạy tại http://{args.host}:{args.port}{RESET}")
    print(f"   Provider: {cfg.provider} · Model: {cfg.model} · Base: {cfg.base_url}")
    print(f"   Endpoints: POST /chat · POST /agent/run · GET /health")
    print(f"{CYAN}   Ctrl+C để dừng.{RESET}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")
        httpd.shutdown()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    load_env()
    parser = argparse.ArgumentParser(
        prog="python minion.py",
        description="Minion — AI coding agent local (Ollama / LM Studio / OpenAI)",
    )
    parser.add_argument("--provider", default="", help="ollama | lmstudio | openai | ai-local")
    parser.add_argument("--model", default="", help="Tên model (vd qwen2.5-coder)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="Xem cấu hình LLM").set_defaults(func=cmd_providers)
    sub.add_parser("models", help="Liệt kê model của provider").set_defaults(func=cmd_models)

    p_chat = sub.add_parser("chat", help="Chat tương tác")
    p_chat.add_argument("--system", default="", help="System prompt")
    p_chat.add_argument("--temperature", type=float, default=0.3)
    p_chat.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p_chat.set_defaults(func=cmd_chat)

    p_agent = sub.add_parser("agent", help="Chạy agent với tool use")
    p_agent.add_argument("task", nargs="+")
    p_agent.add_argument("--tools", default="")
    p_agent.add_argument("--skill", default="")
    p_agent.add_argument("--plan", action="store_true", help="Chế độ chỉ-đọc (lập kế hoạch)")
    p_agent.add_argument("--permission-mode", dest="permission_mode",
                         choices=["auto", "plan", "approve", "readonly"], default="auto")
    p_agent.add_argument("--max-steps", type=int, default=10, dest="max_steps")
    p_agent.add_argument("--temperature", type=float, default=0.2)
    p_agent.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p_agent.add_argument("-v", "--verbose", action="store_true")
    p_agent.set_defaults(func=cmd_agent)

    p_serve = sub.add_parser("serve", help="Chạy HTTP API (stdlib)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
