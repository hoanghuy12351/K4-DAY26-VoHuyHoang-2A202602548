"""Tool Registry Client — agent khám phá tool từ danh mục trung tâm.

Thay vì hard-code "server nào có tool nào", agent hỏi Tool Registry:
  "Tôi cần tool có tag 'weather'" → Registry trả về get_weather, server weather, stdio
  "Tôi cần tool có tag 'forecast'" → Registry trả về get_weather_v2, server weather-v2

Agent tự quyết định dùng tool nào, kết nối tới đúng server, và gọi tool —
tất cả tại runtime mà không cần hard-code.

Production thường dùng:
  - Tool registry tập trung (DB/API) với search full-text hoặc semantic
  - Kubernetes + CRD (Custom Resource Definition) cho MCP servers
  - API Gateway (Kong, Envoy) làm single entry point

Ví dụ này minh hoạ ý tưởng với một file JSON đơn giản.

Cách chạy:
    pip install -r ../requirements.txt     # từ thư mục gốc repo
    cd 03-production
    python registry_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from packaging.version import Version

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

REGISTRY_PATH = Path(__file__).parent / "registry.json"
# Đường dẫn tương đối trong registry.json được tính từ thư mục chứa file registry,
# nên script chạy đúng dù bạn đứng ở thư mục nào.
REGISTRY_DIR = REGISTRY_PATH.parent


class ToolRegistry:
    """Danh mục trung tâm — agent tra cứu tool theo tag, tên, hoặc mô tả."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.tools: dict[str, dict] = data["tools"]
        self.servers: dict[str, dict] = data["servers"]

    def search(self, *, tag: str | None = None, keyword: str | None = None) -> list[dict]:
        """Tìm tool theo tag hoặc keyword trong description.

        Trả về danh sách kết quả, mỗi phần tử gồm thông tin tool + server.
        """
        results = []
        for tool_name, tool_cfg in self.tools.items():
            match = False
            if tag and tag in tool_cfg.get("tags", []):
                match = True
            if keyword and keyword.lower() in tool_cfg.get("description", "").lower():
                match = True
            if match:
                server_cfg = self.servers.get(tool_cfg["server"], {})
                results.append({
                    "tool": tool_name,
                    "description": tool_cfg["description"],
                    "version": tool_cfg["version"],
                    "deprecated": tool_cfg.get("deprecated", False),
                    "parameters": tool_cfg.get("parameters", {}),
                    "server_name": tool_cfg["server"],
                    "server": server_cfg,
                })
        return results

    def best_match(self, *, tag: str | None = None, keyword: str | None = None) -> dict:
        """Trả về tool phù hợp nhất (ưu tiên version cao, không deprecated)."""
        results = self.search(tag=tag, keyword=keyword)
        if not results:
            raise KeyError(f"Không tìm thấy tool (tag={tag}, keyword={keyword})")
        active = [r for r in results if not r["deprecated"]]
        candidates = active or results
        # So sánh theo semantic version, không so chuỗi ("10.0.0" > "2.0.0")
        return max(candidates, key=lambda r: Version(r["version"]))


async def connect_and_call(match: dict, tool_args: dict) -> str:
    """Kết nối tới server phù hợp và gọi tool đã tìm được."""
    server = match["server"]
    tool_name = match["tool"]

    if server.get("transport") == "stdio":
        # "python" trong registry → dùng đúng interpreter hiện tại (tránh lỗi venv)
        command = server.get("command", "python")
        if command in ("python", "python3"):
            command = sys.executable
        args = [
            str((REGISTRY_DIR / a).resolve()) if a.endswith(".py") else a
            for a in server.get("args", [])
        ]
        params = StdioServerParameters(command=command, args=args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                return result.content[0].text

    elif server.get("transport") == "streamable-http":
        headers = {}
        auth_cfg = server.get("auth")
        if auth_cfg and auth_cfg["type"] == "bearer":
            token = os.environ.get(auth_cfg["token_env"], "dev-token-abc123")
            headers["Authorization"] = f"Bearer {token}"

        async with create_mcp_http_client(headers=headers) as http_client:
            async with streamable_http_client(server["url"], http_client=http_client) as (
                read, write,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_args)
                    return result.content[0].text

    raise ValueError(f"Transport không được hỗ trợ: {server.get('transport')}")


async def main() -> None:
    registry = ToolRegistry()

    # ── 1. Agent liệt kê toàn bộ tool trong registry ─────────────────
    print("=== Tool Registry ===\n")
    for name, cfg in registry.tools.items():
        status = " (deprecated)" if cfg.get("deprecated") else ""
        print(f"  {name} v{cfg['version']}{status}")
        print(f"    {cfg['description']}")
        print(f"    tags: {cfg['tags']}  →  server: {cfg['server']}")
    print()

    # ── 2. Agent tìm tool theo tag (giống agent nhận task "lấy thời tiết") ─
    print('--- Agent cần tool có tag "weather" ---')
    results = registry.search(tag="weather")
    print(f"Tìm thấy {len(results)} tool(s):")
    for r in results:
        print(f"  • {r['tool']} v{r['version']} (server: {r['server_name']})")

    # ── 3. Agent chọn tool phù hợp nhất và gọi ──────────────────────
    best = registry.best_match(tag="weather")
    print(f"\nBest match: {best['tool']} v{best['version']}")
    print(f"Kết nối tới server [{best['server_name']}]...")

    output = await connect_and_call(best, {"city": "Hanoi"})
    print(f"Kết quả: {output}\n")

    # ── 4. Agent tìm tool theo keyword ───────────────────────────────
    print('--- Agent tìm tool có keyword "forecast" ---')
    results = registry.search(keyword="forecast")
    for r in results:
        print(f"  • {r['tool']}: {r['description']}")


if __name__ == "__main__":
    asyncio.run(main())
