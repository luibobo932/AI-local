"""
MCP (Model Context Protocol) client cho AI-local.

Cho phép AI-local kết nối tới bất kỳ MCP server nào
(như Claude Code làm) để sử dụng thêm tools.

Hỗ trợ:
- MCP server qua HTTP/SSE (Streamable HTTP Transport)
- Tool discovery (tools/list)
- Tool execution (tools/call)
- Lưu kết quả vào tool registry

Sử dụng:
    from mcp_client import MCPClient

    client = MCPClient("http://localhost:3001")
    tools = await client.list_tools()
    result = await client.call_tool("read_file", {"path": "README.md"})
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict

    def to_openai_schema(self) -> dict:
        """Chuyển thành OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPServer:
    name: str
    url: str
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False
    error: str = ""


class MCPClient:
    """
    MCP client kết nối tới MCP servers.

    Hỗ trợ JSON-RPC 2.0 over HTTP.
    """

    def __init__(self, timeout: float = 30.0):
        self._servers: dict[str, MCPServer] = {}
        self._timeout = timeout
        self._http = httpx.AsyncClient(timeout=timeout)

    async def add_server(self, name: str, url: str) -> MCPServer:
        """Thêm và kết nối tới MCP server."""
        server = MCPServer(name=name, url=url.rstrip("/"))
        self._servers[name] = server

        try:
            tools = await self._list_tools(server)
            server.tools = tools
            server.connected = True
        except Exception as e:
            server.error = str(e)
            server.connected = False

        return server

    def remove_server(self, name: str):
        self._servers.pop(name, None)

    def list_servers(self) -> list[MCPServer]:
        return list(self._servers.values())

    def all_tools(self) -> list[MCPTool]:
        """Lấy tất cả tools từ tất cả servers đang kết nối."""
        tools = []
        for server in self._servers.values():
            if server.connected:
                tools.extend(server.tools)
        return tools

    def all_openai_schemas(self) -> list[dict]:
        """Lấy tất cả tools dưới dạng OpenAI schema."""
        return [t.to_openai_schema() for t in self.all_tools()]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Gọi tool theo tên (tìm trong tất cả servers)."""
        for server in self._servers.values():
            if not server.connected:
                continue
            if any(t.name == tool_name for t in server.tools):
                return await self._call_tool(server, tool_name, arguments)
        return f"[ERROR] Tool '{tool_name}' không tìm thấy trong bất kỳ MCP server nào."

    # ─── Internal ──────────────────────────────────────────────────────────────

    async def _rpc(self, server: MCPServer, method: str, params: dict = None) -> Any:
        """Gửi JSON-RPC 2.0 request tới MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = await self._http.post(
            f"{server.url}/mcp",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result")

    async def _list_tools(self, server: MCPServer) -> list[MCPTool]:
        """Lấy danh sách tools từ server."""
        result = await self._rpc(server, "tools/list")
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            ))
        return tools

    async def _call_tool(self, server: MCPServer, name: str, arguments: dict) -> str:
        """Gọi tool trên server."""
        result = await self._rpc(server, "tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # MCP trả về list content items
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[image: {item.get('url', '')}]")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else str(result)

    async def close(self):
        await self._http.aclose()


# ─── Global instance ──────────────────────────────────────────────────────────

_global_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    global _global_client
    if _global_client is None:
        _global_client = MCPClient()
    return _global_client


# ─── Config file support ──────────────────────────────────────────────────────

def load_mcp_config(config_path: str = ".mcp_servers.json") -> dict:
    """
    Đọc file config MCP servers.

    Format:
    {
      "servers": {
        "filesystem": {"url": "http://localhost:3001"},
        "github":     {"url": "http://localhost:3002"}
      }
    }
    """
    import os
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


async def init_mcp_from_config(config_path: str = ".mcp_servers.json") -> MCPClient:
    """Khởi tạo MCP client từ file config và kết nối tất cả servers."""
    config = load_mcp_config(config_path)
    client = get_mcp_client()
    servers = config.get("servers", {})
    for name, cfg in servers.items():
        url = cfg.get("url", "")
        if url:
            await client.add_server(name, url)
    return client


# ─── CLI mini ─────────────────────────────────────────────────────────────────

async def _cli_main():
    """Test MCP client từ command line."""
    import sys
    if len(sys.argv) < 2:
        print("Dùng: python mcp_client.py <server_url> [tool_name] [json_args]")
        return

    url = sys.argv[1]
    client = MCPClient()
    server = await client.add_server("test", url)

    if not server.connected:
        print(f"Lỗi kết nối: {server.error}")
        return

    print(f"✅ Kết nối tới {url}")
    print(f"Tools có sẵn ({len(server.tools)}):")
    for t in server.tools:
        print(f"  - {t.name}: {t.description}")

    if len(sys.argv) >= 3:
        tool_name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) >= 4 else {}
        print(f"\nGọi {tool_name}({args})...")
        result = await client.call_tool(tool_name, args)
        print(f"Kết quả:\n{result}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(_cli_main())
