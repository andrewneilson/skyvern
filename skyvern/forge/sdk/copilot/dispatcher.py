"""Unified tool dispatcher — routes calls to native handlers or MCP servers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import structlog

from skyvern.forge.sdk.copilot.mcp_host import MCPHost
from skyvern.forge.sdk.copilot.runtime import (
    AgentContext,
    ensure_browser_session,
    mcp_browser_context,
    mcp_to_copilot,
)

LOG = structlog.get_logger()

PreHook = Callable[[dict[str, Any], AgentContext], Awaitable[dict[str, Any] | None]]
PostHook = Callable[[dict[str, Any], dict[str, Any], AgentContext], Awaitable[dict[str, Any]]]


@dataclass
class SchemaOverlay:
    description: str | None = None
    hide_params: frozenset[str] = frozenset()
    required_overrides: list[str] | None = None
    arg_transforms: dict[str, str] = field(default_factory=dict)
    forced_args: dict[str, Any] = field(default_factory=dict)
    requires_browser: bool = False
    timeout: int | None = None
    pre_hook: PreHook | None = None
    post_hook: PostHook | None = None


@dataclass
class NativeTool:
    handler: Callable[[dict[str, Any], AgentContext], Awaitable[dict[str, Any]]]
    schema: dict[str, Any]
    timeout: int | None = None


@dataclass
class DispatchResult:
    copilot_result: dict[str, Any]
    raw_mcp_result: dict[str, Any] | None = None


class ToolDispatcher:
    def __init__(self, host: MCPHost) -> None:
        self._host = host
        self._native: dict[str, NativeTool] = {}
        self._overlays: dict[str, SchemaOverlay] = {}
        self._resolved_schemas: dict[str, dict[str, Any]] | None = None

    def register_native(self, name: str, tool: NativeTool) -> None:
        self._native[name] = tool

    def register_overlay(self, copilot_name: str, overlay: SchemaOverlay) -> None:
        self._overlays[copilot_name] = overlay

    def get_native_tool_names(self) -> frozenset[str]:
        return frozenset(self._native.keys())

    async def resolve_schemas(self) -> None:
        schemas: dict[str, dict[str, Any]] = {}

        for name, tool in self._native.items():
            schemas[name] = tool.schema

        discovered = self._host.get_discovered_tools()
        for copilot_name, disc_tool in discovered.items():
            overlay = self._overlays.get(copilot_name, SchemaOverlay())
            props = dict(disc_tool.input_schema.get("properties", {}))
            required = list(disc_tool.input_schema.get("required", []))

            for p in overlay.hide_params | frozenset(overlay.forced_args):
                props.pop(p, None)
                if p in required:
                    required.remove(p)

            for copilot_param, mcp_param in overlay.arg_transforms.items():
                if mcp_param in props:
                    props[copilot_param] = props.pop(mcp_param)
                if mcp_param in required:
                    required.remove(mcp_param)
                    required.append(copilot_param)

            if overlay.required_overrides is not None:
                required = overlay.required_overrides

            schemas[copilot_name] = {
                "description": overlay.description or disc_tool.description,
                "properties": props,
                "required": required,
            }
        self._resolved_schemas = schemas

    def get_tool_definitions(self, tool_format: str = "anthropic") -> list[dict[str, Any]]:
        if self._resolved_schemas is None:
            raise RuntimeError("Call resolve_schemas() before get_tool_definitions()")

        if tool_format == "anthropic":
            return [
                {
                    "name": name,
                    "description": schema.get("description", ""),
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    },
                }
                for name, schema in self._resolved_schemas.items()
            ]
        else:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", []),
                        },
                    },
                }
                for name, schema in self._resolved_schemas.items()
            ]

    def get_timeout(self, name: str) -> int | None:
        if name in self._native:
            return self._native[name].timeout
        overlay = self._overlays.get(name)
        return overlay.timeout if overlay else None

    def has_tool(self, name: str) -> bool:
        return name in self._native or name in self._host.get_discovered_tools()

    async def dispatch(
        self,
        name: str,
        params: dict[str, Any],
        ctx: AgentContext,
    ) -> DispatchResult:
        if name in self._native:
            result = await self._native[name].handler(params, ctx)
            return DispatchResult(copilot_result=result)

        overlay = self._overlays.get(name, SchemaOverlay())

        if overlay.pre_hook:
            hook_result = await overlay.pre_hook(params, ctx)
            if hook_result is not None:
                return DispatchResult(copilot_result=hook_result)

        mcp_params = {k: v for k, v in params.items() if k not in overlay.hide_params}

        for copilot_param, mcp_param in overlay.arg_transforms.items():
            if copilot_param in mcp_params:
                mcp_params[mcp_param] = mcp_params.pop(copilot_param)

        mcp_params.update(overlay.forced_args)

        if overlay.requires_browser:
            err = await ensure_browser_session(ctx)
            if err:
                return DispatchResult(copilot_result=err)
            mcp_params["session_id"] = ctx.browser_session_id

        try:
            if overlay.requires_browser:
                async with mcp_browser_context(ctx):
                    call_result = await self._host.call_tool(name, mcp_params)
            else:
                call_result = await self._host.call_tool(name, mcp_params)
        except Exception as e:
            LOG.warning("MCP tool call failed", tool=name, error=str(e), exc_info=True)
            return DispatchResult(copilot_result={"ok": False, "error": f"{name} failed: {e}"})

        raw_mcp = call_result.structured_content or {}
        if call_result.is_error:
            raw_mcp["ok"] = False
            if not call_result.structured_content and call_result.content:
                text_parts = [c.text for c in call_result.content if hasattr(c, "text")]
                raw_mcp["error"] = " ".join(text_parts) if text_parts else "Unknown MCP error"
            else:
                raw_mcp["error"] = raw_mcp.get("error") or "Unknown MCP error"
        copilot_result = mcp_to_copilot(raw_mcp)

        if overlay.post_hook:
            copilot_result = await overlay.post_hook(copilot_result, raw_mcp, ctx)

        return DispatchResult(copilot_result=copilot_result, raw_mcp_result=raw_mcp)
