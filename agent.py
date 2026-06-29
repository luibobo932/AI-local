"""
AI-local Agent Executor.

Vòng lặp agent dùng OpenAI function calling (cho model mạnh)
hoặc ReAct text template (cho model nhỏ).

Sử dụng:
    from agent import run_agent, AgentConfig

    result = run_agent(
        task="Đọc file README.md và tóm tắt nội dung",
        model="gpt2",               # hoặc bất kỳ model nào trong server
        tools=["read_file", "web_search"],
        server_url="http://localhost:11434",
    )
    print(result.answer)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llm import LLMClient, LLMError
from tools import TOOL_SCHEMAS, call_tool

logger = logging.getLogger("minion.agent")

# ─── System prompt coding agent (kiểu Claude Code) ───────────────────────────

CODING_AGENT_SYSTEM = """Bạn là Minion Code — một coding agent chạy local, mạnh mẽ như Claude Code.

Bạn giúp người dùng làm việc với codebase: đọc/sửa file, chạy lệnh, debug, viết code.

## Nguyên tắc làm việc
1. **Hiểu trước, làm sau**: dùng `repo_map`, `glob`, `search_files`, `read_file` để nắm ngữ cảnh trước khi sửa.
2. **Lập kế hoạch**: với nhiệm vụ nhiều bước, dùng `todo_write` để chia nhỏ và theo dõi tiến độ; cập nhật status khi làm.
3. **Sửa chính xác**: dùng `edit_file`/`multi_edit`/`apply_patch` với thay đổi tối thiểu, đúng trọng tâm. Khớp code hiện có (style, naming).
4. **Kiểm chứng**: sau khi sửa, chạy test/lint bằng `run_command` nếu có. Đừng báo "xong" khi chưa kiểm tra.
5. **An toàn**: không xóa/ghi đè thứ bạn không tạo ra mà chưa xem; giải thích trước hành động khó đảo ngược.

## Phong cách
- Trả lời tiếng Việt, ngắn gọn, thực dụng.
- Dẫn chứng file:dòng cụ thể.
- Không bịa API/đường dẫn — kiểm tra bằng tool.

Khi hoàn thành, tóm tắt ngắn gọn những gì đã làm và kết quả kiểm chứng."""


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    model: str = "chat_vi"
    server_url: str = "http://localhost:11434"
    base_url: str = "http://localhost:11434/v1"  # endpoint OpenAI-compatible
    api_key: str = ""
    provider: str = ""
    max_steps: int = 10
    temperature: float = 0.2
    max_tokens: int = 1024
    tools: list[str] = field(default_factory=list)   # tên tools, rỗng = tất cả
    system_prompt: str = ""
    mode: str = "auto"  # "auto" | "react" | "function_calling"
    timeout: float = 60.0
    on_step: Optional[object] = None  # callback(AgentStep) gọi sau mỗi bước (cho streaming)
    policy: Optional[object] = None   # PermissionPolicy kiểm soát quyền tool


# ─── Kết quả ──────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    result: str = ""


@dataclass
class AgentStep:
    step: int
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observation: str = ""
    is_final: bool = False


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    model: str
    elapsed: float
    success: bool = True
    error: str = ""


# ─── ReAct template ───────────────────────────────────────────────────────────

_REACT_SYSTEM = """Bạn là AI assistant đa năng. Bạn có thể sử dụng các công cụ để hoàn thành nhiệm vụ.

Quy trình làm việc:
1. Suy nghĩ về nhiệm vụ (Thought)
2. Gọi công cụ nếu cần (Action)
3. Nhận kết quả (Observation)
4. Lặp lại cho đến khi hoàn thành
5. Trả lời cuối cùng (Final Answer)

Định dạng phản hồi:
Thought: [suy nghĩ của bạn]
Action: tool_name({"param": "value"})
Observation: [kết quả tool - hệ thống sẽ điền vào đây]
...
Final Answer: [câu trả lời cuối cùng]

Công cụ có sẵn:
{tools_desc}

Lưu ý:
- Luôn bắt đầu bằng "Thought:"
- Action phải đúng cú pháp JSON
- Kết thúc bằng "Final Answer:" khi đã có đủ thông tin
"""

_REACT_SYSTEM_EN = """You are a versatile AI assistant with access to tools.

Format:
Thought: [your reasoning]
Action: tool_name({"param": "value"})
Observation: [tool result - system fills this in]
...
Final Answer: [your final answer]

Available tools:
{tools_desc}

Rules:
- Always start with "Thought:"
- Action JSON must be valid
- End with "Final Answer:" when done
"""


def _build_tools_desc(tool_names: list[str]) -> str:
    """Mô tả tools dạng text cho ReAct template."""
    from tools import TOOL_REGISTRY, get_tool_schemas
    schemas = get_tool_schemas(tool_names if tool_names else None)
    lines = []
    for s in schemas:
        fn = s["function"]
        params = fn["parameters"].get("properties", {})
        required = fn["parameters"].get("required", [])
        param_desc = ", ".join(
            f"{k}{'*' if k in required else ''}: {v.get('description', v.get('type', ''))}"
            for k, v in params.items()
        ) if params else "không có tham số"
        lines.append(f"- {fn['name']}({param_desc}): {fn['description']}")
    return "\n".join(lines)


# ─── Helper: gọi LLM (Ollama / LM Studio / OpenAI / ai-local) ─────────────────

def _chat_request(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    api_key: str = "",
) -> dict:
    """Gọi {base_url}/chat/completions qua llm.LLMClient (stdlib)."""
    client = LLMClient(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
    return client.chat(
        messages, tools=tools, temperature=temperature, max_tokens=max_tokens
    )


# ─── Streaming callback helper ────────────────────────────────────────────────

def _emit(cfg: AgentConfig, step: "AgentStep"):
    """Gọi callback on_step (nếu có) sau mỗi bước — phục vụ streaming."""
    if cfg.on_step:
        try:
            cfg.on_step(step)
        except Exception:
            pass


# ─── Function Calling mode ────────────────────────────────────────────────────

def _run_function_calling(task: str, cfg: AgentConfig) -> AgentResult:
    """Agent dùng OpenAI function calling (cho model hỗ trợ tool_calls)."""
    from tools import get_tool_schemas

    tool_schemas = get_tool_schemas(cfg.tools if cfg.tools else None)
    system = cfg.system_prompt or "Bạn là AI assistant đa năng. Hãy hoàn thành nhiệm vụ của người dùng."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    steps = []
    t0 = time.time()

    for step_num in range(1, cfg.max_steps + 1):
        step = AgentStep(step=step_num)

        try:
            data = _chat_request(
                cfg.base_url, cfg.model, messages,
                tool_schemas, cfg.temperature, cfg.max_tokens, cfg.timeout, cfg.api_key
            )
        except Exception as e:
            return AgentResult(
                answer="", steps=steps, model=cfg.model,
                elapsed=time.time() - t0, success=False, error=str(e)
            )

        choice = data["choices"][0]
        msg = choice["message"]

        # Model đã có câu trả lời cuối
        if choice.get("finish_reason") == "stop" or not msg.get("tool_calls"):
            step.thought = msg.get("content", "")
            step.is_final = True
            steps.append(step)
            _emit(cfg, step)
            messages.append({"role": "assistant", "content": msg.get("content", "")})
            return AgentResult(
                answer=msg.get("content", ""),
                steps=steps, model=cfg.model,
                elapsed=time.time() - t0
            )

        # Model muốn dùng tools
        tool_calls_raw = msg.get("tool_calls", [])
        step.thought = msg.get("content", "")
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls_raw})

        for tc in tool_calls_raw:
            tc_id = tc.get("id", f"call_{step_num}")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            result = call_tool(name, args, policy=cfg.policy)
            tool_call = ToolCall(id=tc_id, name=name, arguments=args, result=result)
            step.tool_calls.append(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

        steps.append(step)
        _emit(cfg, step)

    # Hết max_steps — lấy câu trả lời cuối cùng
    try:
        data = _chat_request(
            cfg.base_url, cfg.model, messages + [
                {"role": "user", "content": "Tổng kết kết quả và đưa ra câu trả lời cuối cùng."}
            ],
            None, cfg.temperature, cfg.max_tokens, cfg.timeout, cfg.api_key
        )
        answer = data["choices"][0]["message"].get("content", "")
    except Exception:
        answer = "[Đã đạt giới hạn bước, không có câu trả lời cuối]"

    return AgentResult(answer=answer, steps=steps, model=cfg.model, elapsed=time.time() - t0)


# ─── ReAct mode ───────────────────────────────────────────────────────────────

def _run_react(task: str, cfg: AgentConfig) -> AgentResult:
    """Agent dùng ReAct text template (cho model nhỏ không hỗ trợ function calling)."""
    tools_desc = _build_tools_desc(cfg.tools)
    system = (
        cfg.system_prompt or
        _REACT_SYSTEM.format(tools_desc=tools_desc)
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Nhiệm vụ: {task}"},
    ]

    steps = []
    t0 = time.time()
    full_trace = ""

    for step_num in range(1, cfg.max_steps + 1):
        step = AgentStep(step=step_num)

        try:
            data = _chat_request(
                cfg.base_url, cfg.model, messages,
                None, cfg.temperature, cfg.max_tokens, cfg.timeout, cfg.api_key
            )
        except Exception as e:
            return AgentResult(
                answer="", steps=steps, model=cfg.model,
                elapsed=time.time() - t0, success=False, error=str(e)
            )

        text = data["choices"][0]["message"].get("content", "")
        full_trace += text + "\n"

        # Parse Thought
        thought_match = _re_between(text, "Thought:", "Action:")
        step.thought = thought_match.strip() if thought_match else text.split("\n")[0]

        # Check Final Answer
        if "Final Answer:" in text:
            answer = text.split("Final Answer:", 1)[1].strip()
            step.is_final = True
            steps.append(step)
            _emit(cfg, step)
            return AgentResult(
                answer=answer, steps=steps, model=cfg.model,
                elapsed=time.time() - t0
            )

        # Parse Action
        action_text = _re_between(text, "Action:", "Observation:")
        if not action_text:
            action_text = _re_after(text, "Action:")

        if action_text:
            action_text = action_text.strip()
            # Parse: tool_name({"key": "val"}) hoặc tool_name(key=val)
            tool_name, tool_args = _parse_action(action_text)
            if tool_name:
                result = call_tool(tool_name, tool_args, policy=cfg.policy)
                tc = ToolCall(id=f"react_{step_num}", name=tool_name, arguments=tool_args, result=result)
                step.tool_calls.append(tc)
                step.observation = result

                # Append observation vào messages
                obs = f"\nObservation: {result}\n"
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": obs + "\nTiếp tục với Thought:"})
            else:
                # Không parse được action, yêu cầu thử lại
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Không thể parse Action. Hãy dùng đúng định dạng: Action: tool_name({\"key\": \"value\"})"
                })
        else:
            # Không có action, có thể model đã trả lời trực tiếp
            messages.append({"role": "assistant", "content": text})
            if step_num == 1:
                # Lần đầu không có action = trả lời thẳng
                step.is_final = True
                steps.append(step)
                _emit(cfg, step)
                return AgentResult(
                    answer=text, steps=steps, model=cfg.model,
                    elapsed=time.time() - t0
                )
            messages.append({"role": "user", "content": "Đưa ra Final Answer nếu đã hoàn thành."})

        steps.append(step)
        _emit(cfg, step)

    return AgentResult(
        answer=full_trace, steps=steps, model=cfg.model,
        elapsed=time.time() - t0
    )


# ─── Helpers parse ReAct ──────────────────────────────────────────────────────

def _re_between(text: str, start: str, end: str) -> str | None:
    import re
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else None


def _re_after(text: str, marker: str) -> str | None:
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker):]


def _parse_action(action_text: str) -> tuple[str, dict]:
    """Parse 'tool_name({"key": "value"})' hoặc 'tool_name(key="value")'."""
    import re
    action_text = action_text.strip()

    # Dạng JSON: tool_name({"key": "val"})
    m = re.match(r"(\w+)\s*\((\{.*\})\)\s*$", action_text, re.DOTALL)
    if m:
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
            return name, args
        except json.JSONDecodeError:
            pass

    # Dạng không có tham số: tool_name()
    m = re.match(r"(\w+)\s*\(\s*\)\s*$", action_text)
    if m:
        return m.group(1), {}

    # Dạng key=value: tool_name(key="value", key2=123)
    m = re.match(r"(\w+)\s*\((.*)\)\s*$", action_text, re.DOTALL)
    if m:
        name = m.group(1)
        args_str = m.group(2)
        try:
            # Thử eval an toàn với ast.literal_eval
            import ast
            # Chuyển thành dict expression
            args = {}
            for part in re.split(r",\s*(?=\w+=)", args_str):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        args[k.strip()] = ast.literal_eval(v.strip())
                    except Exception:
                        args[k.strip()] = v.strip().strip('"\'')
            return name, args
        except Exception:
            pass

    # Fallback: chỉ lấy tên tool
    words = action_text.split()
    if words:
        return words[0].rstrip("("), {}
    return "", {}


# ─── Main entry point ─────────────────────────────────────────────────────────

def _build_context_prefix(project_root: str, skill: str = "") -> str:
    """Gom project memory + skill prompt để inject vào system prompt."""
    parts = []
    try:
        from memory import load_project_context
        ctx = load_project_context(project_root)
        if ctx.strip():
            parts.append(ctx)
    except Exception:
        pass

    if skill:
        try:
            from skills import get_skill_prompt
            sp = get_skill_prompt(skill, project_root)
            if sp.strip():
                parts.append(f"# Skill: {skill}\n{sp}")
        except Exception:
            pass

    return "\n\n".join(parts)


def run_agent(
    task: str,
    model: str = "",
    server_url: str = "",
    tools: list[str] = None,
    system_prompt: str = "",
    max_steps: int = 10,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    mode: str = "auto",
    timeout: float = 120.0,
    project_root: str = ".",
    skill: str = "",
    use_memory: bool = True,
    on_step=None,
    permission_mode: str = "auto",
    allow_tools=None,
    deny_tools=None,
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
) -> AgentResult:
    """
    Chạy agent để hoàn thành nhiệm vụ.

    Args:
        task: Nhiệm vụ cần hoàn thành
        model: Tên model (phải đang chạy trên server)
        server_url: URL của AI-local server
        tools: Danh sách tên tools (None = tất cả)
        system_prompt: System prompt tùy chỉnh
        max_steps: Số bước tối đa
        temperature: Nhiệt độ sinh text
        max_tokens: Số token tối đa mỗi bước
        mode: "auto" | "react" | "function_calling"
        timeout: Timeout HTTP mỗi request
        project_root: Thư mục dự án (để load memory + skills)
        skill: Tên skill áp dụng (rỗng = không dùng)
        use_memory: Tự động load project memory vào system prompt

    Returns:
        AgentResult với answer, steps, và metadata
    """
    # Resolve provider/base_url/model/api_key (Ollama / LM Studio / OpenAI / ai-local)
    from llm import LLMClient
    # server_url cũ (ai-local) → base_url = server_url + /v1
    if server_url and not base_url:
        base_url = server_url.rstrip("/") + "/v1"
    client = LLMClient.from_env(
        provider=provider, model=model, base_url=base_url, api_key=api_key
    )
    base_url = client.base_url
    api_key = client.api_key
    if not model:
        model = client.model

    # Nếu skill chỉ định tools, dùng tools đó
    if skill and not tools:
        try:
            from skills import load_skill
            sk = load_skill(skill, project_root)
            if sk and sk.get("tools"):
                tools = sk["tools"]
        except Exception:
            pass

    # System prompt mặc định = coding agent (kiểu Claude Code)
    if not system_prompt:
        system_prompt = CODING_AGENT_SYSTEM

    # Inject project memory + skill vào system prompt
    if use_memory or skill:
        prefix = _build_context_prefix(project_root if use_memory else "/nonexistent", skill)
        if prefix:
            system_prompt = f"{system_prompt}\n\n{prefix}"

    # Tạo permission policy
    from tools.permissions import make_policy
    policy = make_policy(permission_mode, allow_tools, deny_tools)

    # Ở chế độ plan, nhắc model rằng nó chỉ được lập kế hoạch
    if permission_mode in ("plan", "readonly"):
        system_prompt += (
            "\n\n## CHẾ ĐỘ PLAN\n"
            "Bạn ĐANG ở chế độ chỉ-đọc. KHÔNG được ghi/sửa file hay chạy lệnh. "
            "Hãy khám phá, phân tích, rồi trình bày KẾ HOẠCH thay đổi chi tiết "
            "(file nào, sửa gì) để người dùng duyệt — đừng cố thực thi."
        )

    cfg = AgentConfig(
        model=model,
        server_url=server_url,
        base_url=base_url,
        api_key=api_key,
        provider=client.provider,
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools or [],
        system_prompt=system_prompt,
        mode=mode,
        timeout=timeout,
        on_step=on_step,
        policy=policy,
    )

    # Auto-detect mode: model mạnh (Ollama/LM Studio/OpenAI) → function_calling
    if mode == "auto":
        # ai-local provider (model nhỏ built-in) → react; còn lại → function_calling
        is_tiny_local = cfg.provider == "ai-local" or model in ("chat_vi", "minion", "phogpt")
        capable = any(
            m in model.lower() for m in [
                "llama", "mistral", "gpt-4", "gpt-3.5", "gpt2-xl", "o1", "o3",
                "phi", "gemma", "qwen", "yi", "deepseek", "coder", "codestral",
                "mixtral", "command", "nemo", "granite",
            ]
        )
        effective_mode = "react" if is_tiny_local else ("function_calling" if capable else "react")
    else:
        effective_mode = mode

    if effective_mode == "function_calling":
        return _run_function_calling(task, cfg)
    else:
        return _run_react(task, cfg)


def step_to_dict(step: "AgentStep", result_limit: int = 2000) -> dict:
    """Serialize một AgentStep thành dict (cho JSON/SSE)."""
    return {
        "step": step.step,
        "thought": step.thought,
        "is_final": step.is_final,
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result[:result_limit],
            }
            for tc in step.tool_calls
        ],
    }


def format_result(result: AgentResult, verbose: bool = False) -> str:
    """Format AgentResult thành text đẹp cho hiển thị."""
    lines = []

    if not result.success:
        lines.append(f"❌ Lỗi: {result.error}")
        return "\n".join(lines)

    if verbose:
        for step in result.steps:
            lines.append(f"\n── Bước {step.step} ──")
            if step.thought:
                lines.append(f"💭 Suy nghĩ: {step.thought}")
            for tc in step.tool_calls:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                lines.append(f"🔧 Gọi: {tc.name}({args_str})")
                lines.append(f"📊 Kết quả: {tc.result[:500]}{'...' if len(tc.result) > 500 else ''}")
            if step.is_final:
                lines.append("✅ Hoàn thành")

    lines.append(f"\n{'─'*50}")
    lines.append(f"🤖 Câu trả lời ({result.model}, {len(result.steps)} bước, {result.elapsed:.1f}s):")
    lines.append(result.answer)
    return "\n".join(lines)
