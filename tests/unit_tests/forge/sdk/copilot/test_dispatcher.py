"""Tests for ToolDispatcher."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyvern.forge.sdk.copilot.dispatcher import (
    NativeTool,
    SchemaOverlay,
    ToolDispatcher,
)
from skyvern.forge.sdk.copilot.mcp_host import DiscoveredTool, MCPHost
from skyvern.forge.sdk.copilot.runtime import AgentContext


def _make_ctx(**overrides: Any) -> AgentContext:
    defaults = dict(
        organization_id="org-1",
        workflow_id="wf-1",
        workflow_permanent_id="wfp-1",
        workflow_yaml="",
        browser_session_id=None,
        stream=MagicMock(),
        api_key=None,
    )
    defaults.update(overrides)
    return AgentContext(**defaults)


def _make_host(discovered: dict[str, DiscoveredTool] | None = None) -> MCPHost:
    host = MagicMock(spec=MCPHost)
    host.get_discovered_tools.return_value = discovered or {}
    host.call_tool = AsyncMock()
    return host


def _make_call_result(
    structured_content: dict[str, Any] | None = None,
    is_error: bool = False,
    content: list[Any] | None = None,
) -> MagicMock:
    result = MagicMock()
    result.structured_content = structured_content
    result.is_error = is_error
    result.content = content or []
    return result


def _disc_tool(
    copilot_name: str,
    mcp_name: str | None = None,
    props: dict | None = None,
    required: list[str] | None = None,
) -> DiscoveredTool:
    return DiscoveredTool(
        server_name="test",
        mcp_name=mcp_name or copilot_name,
        copilot_name=copilot_name,
        description=f"Description for {copilot_name}",
        input_schema={
            "properties": props or {},
            "required": required or [],
        },
    )


class TestSchemaOverlay:
    @pytest.mark.asyncio
    async def test_hide_params(self) -> None:
        discovered = {
            "tool_a": _disc_tool(
                "tool_a",
                props={
                    "session_id": {"type": "string"},
                    "url": {"type": "string"},
                },
                required=["session_id", "url"],
            ),
        }
        host = _make_host(discovered)
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "tool_a",
            SchemaOverlay(hide_params=frozenset({"session_id"})),
        )

        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        tool_def = next(d for d in defs if d["name"] == "tool_a")
        assert "session_id" not in tool_def["input_schema"]["properties"]
        assert "url" in tool_def["input_schema"]["properties"]
        assert "session_id" not in tool_def["input_schema"]["required"]

    @pytest.mark.asyncio
    async def test_required_overrides(self) -> None:
        discovered = {
            "tool_a": _disc_tool(
                "tool_a",
                props={"a": {}, "b": {}, "c": {}},
                required=["a", "b", "c"],
            ),
        }
        host = _make_host(discovered)
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "tool_a",
            SchemaOverlay(required_overrides=["a"]),
        )

        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        tool_def = next(d for d in defs if d["name"] == "tool_a")
        assert tool_def["input_schema"]["required"] == ["a"]

    @pytest.mark.asyncio
    async def test_arg_transforms(self) -> None:
        discovered = {
            "type_text": _disc_tool(
                "type_text",
                props={
                    "selector": {"type": "string"},
                    "clear": {"type": "boolean"},
                },
                required=["selector", "clear"],
            ),
        }
        host = _make_host(discovered)
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "type_text",
            SchemaOverlay(arg_transforms={"clear_first": "clear"}),
        )

        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        tool_def = next(d for d in defs if d["name"] == "type_text")
        props = tool_def["input_schema"]["properties"]
        assert "clear_first" in props
        assert "clear" not in props
        assert "clear_first" in tool_def["input_schema"]["required"]

    @pytest.mark.asyncio
    async def test_forced_args_hidden_from_schema(self) -> None:
        discovered = {
            "screenshot": _disc_tool(
                "screenshot",
                props={
                    "inline": {"type": "boolean"},
                    "full_page": {"type": "boolean"},
                },
                required=["inline"],
            ),
        }
        host = _make_host(discovered)
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "screenshot",
            SchemaOverlay(forced_args={"inline": True}),
        )

        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        tool_def = next(d for d in defs if d["name"] == "screenshot")
        assert "inline" not in tool_def["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_description_override(self) -> None:
        discovered = {"tool_a": _disc_tool("tool_a")}
        host = _make_host(discovered)
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "tool_a",
            SchemaOverlay(description="Custom description"),
        )

        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        tool_def = next(d for d in defs if d["name"] == "tool_a")
        assert tool_def["description"] == "Custom description"


class TestNativeDispatch:
    @pytest.mark.asyncio
    async def test_native_tool_called(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)

        handler = AsyncMock(return_value={"ok": True, "data": "result"})
        dispatcher.register_native(
            "my_native",
            NativeTool(handler=handler, schema={"description": "test"}),
        )

        ctx = _make_ctx()
        result = await dispatcher.dispatch("my_native", {"arg": "val"}, ctx)

        handler.assert_awaited_once_with({"arg": "val"}, ctx)
        assert result.copilot_result == {"ok": True, "data": "result"}
        assert result.raw_mcp_result is None

    def test_has_tool_native(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_native(
            "my_native",
            NativeTool(handler=AsyncMock(), schema={}),
        )
        assert dispatcher.has_tool("my_native") is True
        assert dispatcher.has_tool("unknown") is False

    def test_get_native_tool_names(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_native("a", NativeTool(handler=AsyncMock(), schema={}))
        dispatcher.register_native("b", NativeTool(handler=AsyncMock(), schema={}))
        assert dispatcher.get_native_tool_names() == frozenset({"a", "b"})


class TestMCPDispatch:
    @pytest.mark.asyncio
    async def test_mcp_tool_called_with_transformed_args(self) -> None:
        discovered = {
            "type_text": _disc_tool("type_text", mcp_name="skyvern_type"),
        }
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={"ok": True, "data": {"selector": "#email"}},
        )
        host.call_tool.return_value = call_result

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "type_text",
            SchemaOverlay(arg_transforms={"clear_first": "clear"}),
        )

        ctx = _make_ctx()
        result = await dispatcher.dispatch(
            "type_text",
            {"selector": "#email", "text": "hello", "clear_first": True},
            ctx,
        )

        # Verify arg transform: clear_first -> clear
        call_args = host.call_tool.call_args
        mcp_params = call_args[0][1]
        assert "clear" in mcp_params
        assert "clear_first" not in mcp_params
        assert result.copilot_result["ok"] is True

    @pytest.mark.asyncio
    async def test_forced_args_injected(self) -> None:
        discovered = {
            "screenshot": _disc_tool("screenshot", mcp_name="skyvern_screenshot"),
        }
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={"ok": True, "data": {"data": "base64..."}},
        )
        host.call_tool.return_value = call_result

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "screenshot",
            SchemaOverlay(forced_args={"inline": True}),
        )

        ctx = _make_ctx()
        await dispatcher.dispatch("screenshot", {"full_page": False}, ctx)

        call_args = host.call_tool.call_args
        mcp_params = call_args[0][1]
        assert mcp_params["inline"] is True

    @pytest.mark.asyncio
    async def test_hidden_params_stripped(self) -> None:
        discovered = {
            "navigate": _disc_tool("navigate"),
        }
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={"ok": True, "data": {"url": "https://example.com"}},
        )
        host.call_tool.return_value = call_result

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "navigate",
            SchemaOverlay(hide_params=frozenset({"session_id"})),
        )

        ctx = _make_ctx()
        # Pass session_id even though it should be stripped
        await dispatcher.dispatch(
            "navigate",
            {"url": "https://example.com", "session_id": "should_be_removed"},
            ctx,
        )

        call_args = host.call_tool.call_args
        mcp_params = call_args[0][1]
        assert "session_id" not in mcp_params

    @pytest.mark.asyncio
    async def test_is_error_with_structured_content(self) -> None:
        discovered = {"tool_a": _disc_tool("tool_a")}
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={"ok": False, "error": "Something broke"},
            is_error=True,
        )
        host.call_tool.return_value = call_result

        dispatcher = ToolDispatcher(host)
        ctx = _make_ctx()
        result = await dispatcher.dispatch("tool_a", {}, ctx)

        assert result.copilot_result["ok"] is False
        assert "Something broke" in result.copilot_result["error"]

    @pytest.mark.asyncio
    async def test_is_error_without_structured_content(self) -> None:
        discovered = {"tool_a": _disc_tool("tool_a")}
        host = _make_host(discovered)

        text_content = MagicMock()
        text_content.text = "Server error occurred"

        call_result = _make_call_result(
            structured_content=None,
            is_error=True,
            content=[text_content],
        )
        host.call_tool.return_value = call_result

        dispatcher = ToolDispatcher(host)
        ctx = _make_ctx()
        result = await dispatcher.dispatch("tool_a", {}, ctx)

        assert result.copilot_result["ok"] is False
        assert "Server error occurred" in result.copilot_result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_exception_returns_error(self) -> None:
        discovered = {"tool_a": _disc_tool("tool_a")}
        host = _make_host(discovered)
        host.call_tool.side_effect = RuntimeError("Connection lost")

        dispatcher = ToolDispatcher(host)
        ctx = _make_ctx()
        result = await dispatcher.dispatch("tool_a", {}, ctx)

        assert result.copilot_result["ok"] is False
        assert "Connection lost" in result.copilot_result["error"]


class TestPreHooks:
    @pytest.mark.asyncio
    async def test_pre_hook_blocks_dispatch(self) -> None:
        discovered = {"evaluate": _disc_tool("evaluate")}
        host = _make_host(discovered)

        async def block_clicks(params: dict, ctx: AgentContext) -> dict | None:
            if ".click()" in params.get("expression", ""):
                return {"ok": False, "error": "No clicking!"}
            return None

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "evaluate",
            SchemaOverlay(pre_hook=block_clicks),
        )

        ctx = _make_ctx()
        result = await dispatcher.dispatch(
            "evaluate",
            {"expression": "document.querySelector('#btn').click()"},
            ctx,
        )

        assert result.copilot_result["ok"] is False
        assert "No clicking" in result.copilot_result["error"]
        # MCP call should NOT have been made
        host.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pre_hook_allows_dispatch(self) -> None:
        discovered = {"evaluate": _disc_tool("evaluate")}
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={"ok": True, "data": {"result": 42}},
        )
        host.call_tool.return_value = call_result

        async def block_clicks(params: dict, ctx: AgentContext) -> dict | None:
            if ".click()" in params.get("expression", ""):
                return {"ok": False, "error": "No clicking!"}
            return None

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "evaluate",
            SchemaOverlay(pre_hook=block_clicks),
        )

        ctx = _make_ctx()
        result = await dispatcher.dispatch(
            "evaluate",
            {"expression": "document.title"},
            ctx,
        )

        assert result.copilot_result["ok"] is True
        host.call_tool.assert_awaited_once()


class TestPostHooks:
    @pytest.mark.asyncio
    async def test_post_hook_transforms_result(self) -> None:
        discovered = {"navigate": _disc_tool("navigate")}
        host = _make_host(discovered)
        call_result = _make_call_result(
            structured_content={
                "ok": True,
                "data": {"url": "https://example.com"},
            },
        )
        host.call_tool.return_value = call_result

        async def nav_hook(
            result: dict[str, Any],
            raw: dict[str, Any],
            ctx: AgentContext,
        ) -> dict[str, Any]:
            if result.get("ok"):
                data = result.pop("data", {})
                result["url"] = data.get("url", "")
                result["next_step"] = "Page loaded."
            return result

        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "navigate",
            SchemaOverlay(post_hook=nav_hook),
        )

        ctx = _make_ctx()
        result = await dispatcher.dispatch("navigate", {"url": "https://example.com"}, ctx)

        assert result.copilot_result["url"] == "https://example.com"
        assert result.copilot_result["next_step"] == "Page loaded."
        assert "data" not in result.copilot_result


class TestToolDefinitions:
    @pytest.mark.asyncio
    async def test_resolve_schemas_required_first(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        with pytest.raises(RuntimeError, match="Call resolve_schemas"):
            dispatcher.get_tool_definitions()

    @pytest.mark.asyncio
    async def test_anthropic_format(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_native(
            "test_tool",
            NativeTool(
                handler=AsyncMock(),
                schema={
                    "description": "A test tool",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            ),
        )
        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="anthropic")

        assert len(defs) == 1
        assert defs[0]["name"] == "test_tool"
        assert defs[0]["description"] == "A test tool"
        assert defs[0]["input_schema"]["type"] == "object"
        assert "x" in defs[0]["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_openai_format(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_native(
            "test_tool",
            NativeTool(
                handler=AsyncMock(),
                schema={
                    "description": "A test tool",
                    "properties": {},
                    "required": [],
                },
            ),
        )
        await dispatcher.resolve_schemas()
        defs = dispatcher.get_tool_definitions(tool_format="openai")

        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "test_tool"


class TestTimeout:
    def test_native_timeout(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_native(
            "slow_tool",
            NativeTool(handler=AsyncMock(), schema={}, timeout=120),
        )
        assert dispatcher.get_timeout("slow_tool") == 120

    def test_overlay_timeout(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        dispatcher.register_overlay(
            "evaluate",
            SchemaOverlay(timeout=30),
        )
        assert dispatcher.get_timeout("evaluate") == 30

    def test_no_timeout(self) -> None:
        host = _make_host()
        dispatcher = ToolDispatcher(host)
        assert dispatcher.get_timeout("unknown") is None
