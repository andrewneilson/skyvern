"""Tests for MCPHost."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from skyvern.forge.sdk.copilot.mcp_host import (
    DiscoveredTool,
    MCPHost,
    ServerRegistration,
    ToolPolicy,
)


def _make_tool(name: str, description: str = "", input_schema: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = input_schema or {"properties": {}, "required": []}
    return t


def _make_client(tools: list[MagicMock] | None = None) -> MagicMock:
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=tools or [])
    client.call_tool = AsyncMock()
    # Support async context manager protocol
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestToolPolicy:
    def test_allow_all_by_default(self) -> None:
        policy = ToolPolicy()
        assert policy.allows("anything") is True

    def test_allowlist_filters(self) -> None:
        policy = ToolPolicy(allowlist=frozenset({"a", "b"}))
        assert policy.allows("a") is True
        assert policy.allows("c") is False

    def test_denylist_filters(self) -> None:
        policy = ToolPolicy(denylist=frozenset({"bad"}))
        assert policy.allows("good") is True
        assert policy.allows("bad") is False

    def test_denylist_overrides_allowlist(self) -> None:
        policy = ToolPolicy(
            allowlist=frozenset({"a", "b"}),
            denylist=frozenset({"b"}),
        )
        assert policy.allows("a") is True
        assert policy.allows("b") is False


class TestMCPHostLifecycle:
    @pytest.mark.asyncio
    async def test_connect_all_when_already_connected_raises(self) -> None:
        host = MCPHost()
        host._exit_stack = MagicMock()  # Simulate already connected
        with pytest.raises(RuntimeError, match="Already connected"):
            await host.connect_all()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self) -> None:
        host = MCPHost()
        # Manually populate state
        host._clients["test"] = _make_client()
        host._discovered["tool"] = DiscoveredTool(
            server_name="test",
            mcp_name="tool",
            copilot_name="tool",
            description="",
            input_schema={},
        )
        host._exit_stack = MagicMock()
        host._exit_stack.__aexit__ = AsyncMock()

        await host.disconnect_all()

        assert host._clients == {}
        assert host._discovered == {}
        assert host._exit_stack is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_safe(self) -> None:
        host = MCPHost()
        # Should not raise
        await host.disconnect_all()


class TestMCPHostDiscovery:
    @pytest.mark.asyncio
    async def test_basic_discovery(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(name="s1", transport=None)
        host._servers["s1"] = reg

        client = _make_client(
            [
                _make_tool("tool_a", "desc A"),
                _make_tool("tool_b", "desc B"),
            ]
        )
        host._clients["s1"] = client

        discovered = await host.discover_tools()

        assert "tool_a" in discovered
        assert "tool_b" in discovered
        assert discovered["tool_a"].server_name == "s1"
        assert discovered["tool_a"].mcp_name == "tool_a"
        assert discovered["tool_a"].copilot_name == "tool_a"

    @pytest.mark.asyncio
    async def test_allowlist_filtering(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            policy=ToolPolicy(allowlist=frozenset({"tool_a"})),
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client(
            [
                _make_tool("tool_a"),
                _make_tool("tool_b"),
            ]
        )

        discovered = await host.discover_tools()

        assert "tool_a" in discovered
        assert "tool_b" not in discovered

    @pytest.mark.asyncio
    async def test_denylist_filtering(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            policy=ToolPolicy(denylist=frozenset({"tool_b"})),
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client(
            [
                _make_tool("tool_a"),
                _make_tool("tool_b"),
            ]
        )

        discovered = await host.discover_tools()

        assert "tool_a" in discovered
        assert "tool_b" not in discovered

    @pytest.mark.asyncio
    async def test_alias_mapping(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            alias_map={"my_navigate": "skyvern_navigate"},
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client(
            [
                _make_tool("skyvern_navigate", "Navigate"),
                _make_tool("skyvern_click", "Click"),
            ]
        )

        discovered = await host.discover_tools()

        assert "my_navigate" in discovered
        assert discovered["my_navigate"].mcp_name == "skyvern_navigate"
        # Un-aliased tool should appear under its own name
        assert "skyvern_click" in discovered

    @pytest.mark.asyncio
    async def test_alias_target_not_found_raises(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            alias_map={"my_tool": "nonexistent_tool"},
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client([_make_tool("real_tool")])

        with pytest.raises(ValueError, match="not found on server"):
            await host.discover_tools()

    @pytest.mark.asyncio
    async def test_duplicate_alias_target_raises(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            alias_map={
                "alias_a": "skyvern_navigate",
                "alias_b": "skyvern_navigate",
            },
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client([_make_tool("skyvern_navigate")])

        with pytest.raises(ValueError, match="Duplicate alias target"):
            await host.discover_tools()

    @pytest.mark.asyncio
    async def test_alias_filtered_by_policy_raises(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(
            name="s1",
            transport=None,
            policy=ToolPolicy(denylist=frozenset({"blocked_tool"})),
            alias_map={"my_tool": "blocked_tool"},
        )
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client([_make_tool("blocked_tool")])

        with pytest.raises(ValueError, match="filtered out by policy"):
            await host.discover_tools()

    @pytest.mark.asyncio
    async def test_native_name_collision_raises(self) -> None:
        host = MCPHost()
        reg = ServerRegistration(name="s1", transport=None)
        host._servers["s1"] = reg
        host._clients["s1"] = _make_client([_make_tool("update_workflow")])

        with pytest.raises(ValueError, match="collides with native tool"):
            await host.discover_tools(
                native_names=frozenset({"update_workflow"}),
            )

    @pytest.mark.asyncio
    async def test_mcp_to_mcp_name_collision_raises(self) -> None:
        host = MCPHost()
        # Two servers exposing the same tool name
        host._servers["s1"] = ServerRegistration(name="s1", transport=None)
        host._servers["s2"] = ServerRegistration(name="s2", transport=None)
        host._clients["s1"] = _make_client([_make_tool("shared_tool")])
        host._clients["s2"] = _make_client([_make_tool("shared_tool")])

        with pytest.raises(ValueError, match="Tool name collision"):
            await host.discover_tools()


class TestMCPHostCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self) -> None:
        host = MCPHost()
        host._servers["s1"] = ServerRegistration(name="s1", transport=None)
        client = _make_client()
        host._clients["s1"] = client

        # Manually add a discovered tool
        host._discovered["my_tool"] = DiscoveredTool(
            server_name="s1",
            mcp_name="real_mcp_tool",
            copilot_name="my_tool",
            description="",
            input_schema={},
        )

        await host.call_tool("my_tool", {"arg": "val"})

        client.call_tool.assert_awaited_once_with(
            "real_mcp_tool",
            {"arg": "val"},
            raise_on_error=False,
        )

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises_key_error(self) -> None:
        host = MCPHost()
        with pytest.raises(KeyError):
            await host.call_tool("nonexistent", {})
