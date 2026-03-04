"""MCP host — manages client connections to MCP servers."""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client
from fastmcp.client.client import CallToolResult


@dataclass(frozen=True)
class ToolPolicy:
    allowlist: frozenset[str] | None = None
    denylist: frozenset[str] | None = None

    def allows(self, tool_name: str) -> bool:
        if self.denylist and tool_name in self.denylist:
            return False
        if self.allowlist is not None:
            return tool_name in self.allowlist
        return True


@dataclass(frozen=True)
class ServerRegistration:
    name: str
    transport: Any  # FastMCP instance, URL, or command
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    alias_map: dict[str, str] = field(default_factory=dict)


@dataclass
class DiscoveredTool:
    server_name: str
    mcp_name: str
    copilot_name: str
    description: str
    input_schema: dict[str, Any]


class MCPHost:
    def __init__(self) -> None:
        self._servers: dict[str, ServerRegistration] = {}
        self._clients: dict[str, Client] = {}
        self._discovered: dict[str, DiscoveredTool] = {}
        self._exit_stack: AsyncExitStack | None = None

    def register_server(self, registration: ServerRegistration) -> None:
        self._servers[registration.name] = registration

    async def connect_all(self) -> None:
        if self._exit_stack is not None:
            raise RuntimeError("Already connected — call disconnect_all() first")
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            for name, reg in self._servers.items():
                client = Client(reg.transport)
                await stack.enter_async_context(client)
                self._clients[name] = client
        except Exception:
            await stack.__aexit__(*sys.exc_info())
            self._clients.clear()
            raise
        self._exit_stack = stack

    async def disconnect_all(self) -> None:
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
        self._clients.clear()
        self._discovered.clear()
        self._exit_stack = None

    async def discover_tools(
        self,
        native_names: frozenset[str] = frozenset(),
    ) -> dict[str, DiscoveredTool]:
        self._discovered.clear()

        for server_name, client in self._clients.items():
            reg = self._servers[server_name]
            tools = await client.list_tools()
            mcp_names = {t.name for t in tools}

            reverse_alias: dict[str, str] = {}  # mcp_name -> copilot_name
            for copilot_name, mcp_name in reg.alias_map.items():
                if mcp_name not in mcp_names:
                    raise ValueError(
                        f"Alias '{copilot_name}' -> '{mcp_name}': "
                        f"MCP tool '{mcp_name}' not found on server '{server_name}'"
                    )
                if mcp_name in reverse_alias:
                    raise ValueError(
                        f"Duplicate alias target: both '{copilot_name}' and "
                        f"'{reverse_alias[mcp_name]}' alias '{mcp_name}'"
                    )
                reverse_alias[mcp_name] = copilot_name

            for copilot_name, mcp_name in reg.alias_map.items():
                if not reg.policy.allows(mcp_name):
                    raise ValueError(
                        f"Alias '{copilot_name}' -> '{mcp_name}': "
                        f"tool is filtered out by policy on server '{server_name}'"
                    )

            for tool in tools:
                if not reg.policy.allows(tool.name):
                    continue
                copilot_name = reverse_alias.get(tool.name, tool.name)

                if copilot_name in native_names:
                    raise ValueError(f"MCP tool '{copilot_name}' (from '{server_name}') collides with native tool")
                if copilot_name in self._discovered:
                    existing = self._discovered[copilot_name]
                    raise ValueError(
                        f"Tool name collision: '{copilot_name}' from '{server_name}' and '{existing.server_name}'"
                    )
                self._discovered[copilot_name] = DiscoveredTool(
                    server_name=server_name,
                    mcp_name=tool.name,
                    copilot_name=copilot_name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )

        return dict(self._discovered)

    def get_discovered_tools(self) -> dict[str, DiscoveredTool]:
        return dict(self._discovered)

    async def call_tool(
        self,
        copilot_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        tool = self._discovered[copilot_name]
        client = self._clients[tool.server_name]
        return await client.call_tool(
            tool.mcp_name,
            arguments,
            raise_on_error=False,
        )
