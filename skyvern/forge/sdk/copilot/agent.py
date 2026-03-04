"""Copilot agent loop — multi-turn tool-use agent for workflow building."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from skyvern.config import settings
from skyvern.exceptions import SkyvernContextWindowExceededError
from skyvern.forge.prompts import prompt_engine
from skyvern.forge.sdk.api.llm.api_handler_factory import LLMCaller
from skyvern.forge.sdk.api.llm.config_registry import LLMConfigRegistry
from skyvern.forge.sdk.copilot.context import StructuredContext
from skyvern.forge.sdk.copilot.dispatcher import ToolDispatcher
from skyvern.forge.sdk.copilot.mcp_host import MCPHost, ServerRegistration, ToolPolicy
from skyvern.forge.sdk.copilot.runtime import AgentContext
from skyvern.forge.sdk.copilot.tools import (
    get_skyvern_mcp_alias_map,
    process_workflow_yaml,
    register_all_tools,
)
from skyvern.forge.sdk.routes.event_source_stream import EventSourceStream
from skyvern.forge.sdk.schemas.workflow_copilot import (
    WorkflowCopilotChatHistoryMessage,
    WorkflowCopilotCondensingUpdate,
    WorkflowCopilotStreamMessageType,
    WorkflowCopilotToolCallUpdate,
    WorkflowCopilotToolResultUpdate,
)
from skyvern.forge.sdk.workflow.exceptions import BaseWorkflowHTTPException
from skyvern.forge.sdk.workflow.models.workflow import Workflow

LOG = structlog.get_logger()

WORKFLOW_KNOWLEDGE_BASE_PATH = Path("skyvern/forge/prompts/skyvern/workflow_knowledge_base.txt")

MAX_ITERATIONS = 25
TOOL_TIMEOUT_SECONDS = 30
TOTAL_TIMEOUT_SECONDS = 600
COMPACTION_THRESHOLD = 0.75
DEFAULT_CONTEXT_WINDOW = 128_000
MAX_CONSECUTIVE_SAME_TOOL = 3
MAX_POST_UPDATE_NUDGES = 2


@dataclass
class AgentResult:
    user_response: str
    updated_workflow: Workflow | None
    global_llm_context: str | None
    response_type: str = "REPLY"
    workflow_yaml: str | None = None
    workflow_was_persisted: bool = False


def _format_chat_history(chat_history: list[WorkflowCopilotChatHistoryMessage]) -> str:
    if not chat_history:
        return ""
    lines = [f"{msg.sender}: {msg.content}" for msg in chat_history]
    return "\n".join(lines)


def _build_system_prompt(
    workflow_yaml: str,
    chat_history_text: str,
    global_llm_context: str,
    debug_run_info_text: str,
    tool_usage_guide: str,
) -> str:
    workflow_knowledge_base = WORKFLOW_KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    return prompt_engine.load_prompt(
        template="workflow-copilot-agent",
        workflow_knowledge_base=workflow_knowledge_base,
        workflow_yaml=workflow_yaml or "",
        chat_history=chat_history_text,
        global_llm_context=global_llm_context or "",
        current_datetime=datetime.now(timezone.utc).isoformat(),
        debug_run_info=debug_run_info_text,
        tool_usage_guide=tool_usage_guide,
    )


def _extract_text_from_content(content: list[dict[str, Any]]) -> str:
    return "\n".join(block["text"] for block in content if block.get("type") == "text" and block.get("text"))


def _build_tool_usage_guide(tool_defs: list[dict[str, Any]]) -> str:
    """Build a short tool catalog from runtime tool definitions."""
    seen: set[str] = set()
    lines: list[str] = []

    for tool_def in tool_defs:
        # Anthropic format: top-level name/description
        # OpenAI format: nested under "function"
        source = tool_def if "name" in tool_def else tool_def.get("function", {})
        if not isinstance(source, dict):
            continue

        name = str(source.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)

        description = " ".join(str(source.get("description", "")).split()).strip()
        lines.append(f"- **{name}** — {description or 'No description provided.'}")

    return "\n".join(lines) if lines else "- No tool metadata available."


def _parse_final_response(text: str) -> dict[str, Any]:
    """Parse the agent's final text response, stripping markdown fences if present."""
    cleaned = text.strip()
    for prefix in ("```json", "```"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return {"type": "REPLY", "user_response": text}


def _summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Create a brief human-readable summary of a tool result."""
    if not result.get("ok", False):
        return f"Failed: {result.get('error', 'Unknown error')[:200]}"

    data = result.get("data") or {}

    if tool_name == "update_workflow":
        return f"Workflow updated ({data.get('block_count', '?')} blocks)"
    if tool_name == "list_credentials":
        return f"Found {data.get('count', 0)} credential(s)"
    if tool_name == "get_block_schema":
        if "block_types" in data:
            return f"Listed {data.get('count', '?')} block types"
        return f"Schema for {data.get('block_type', '?')}"
    if tool_name == "validate_block":
        if data.get("valid"):
            return f"Block '{data.get('label', '?')}' is valid"
        return "Block validation failed"
    if tool_name == "run_blocks_and_collect_debug":
        labels = [b.get("label", "?") for b in data.get("blocks", [])]
        return f"Run {', '.join(labels)}: {data.get('overall_status', '?')}"
    if tool_name == "get_browser_screenshot":
        return f"Screenshot taken ({data.get('url', '?')[:80]})"
    if tool_name == "navigate_browser":
        url = result.get("url") or data.get("url", "?")
        return f"Navigated to {url[:80]}"
    if tool_name == "evaluate":
        result_val = data.get("result")
        preview = str(result_val)[:100] if result_val is not None else "undefined"
        return f"JS result: {preview}"
    if tool_name == "click":
        return f"Clicked '{data.get('selector', '?')}'"
    if tool_name == "type_text":
        length = data.get("typed_length") or data.get("text_length", "?")
        return f"Typed {length} chars into '{data.get('selector', '?')}'"
    return "OK"


def _sanitize_tool_result_for_llm(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Strip large/binary fields from tool results before sending to the LLM."""
    sanitized = dict(result)
    for key in ("action", "browser_context", "artifacts", "timing_ms"):
        sanitized.pop(key, None)

    data = sanitized.get("data")
    if isinstance(data, dict):
        data = dict(data)
        if "screenshot_base64" in data:
            data["screenshot_base64"] = "[base64 image omitted — screenshot was taken successfully]"
        if "visible_elements_html" in data and data["visible_elements_html"]:
            html = data["visible_elements_html"]
            if len(html) > 3000:
                data["visible_elements_html"] = html[:3000] + "\n... [truncated]"
        if "schema" in data and isinstance(data["schema"], dict):
            schema_str = json.dumps(data["schema"])
            if len(schema_str) > 2000:
                data["schema"] = {
                    "_truncated": True,
                    "message": f"Schema too large ({len(schema_str)} chars). Use get_block_schema for the specific block type.",
                }
        data.pop("sdk_equivalent", None)
        sanitized["data"] = data
    sanitized.pop("_workflow", None)
    return sanitized


def _get_llm_key_for_copilot(llm_api_handler: Any) -> str:
    return getattr(llm_api_handler, "llm_key", None) or settings.LLM_KEY


def _get_context_window(llm_key: str) -> int:
    try:
        import litellm

        config = LLMConfigRegistry.get_config(llm_key)
        info = litellm.get_model_info(config.model_name)
        return info.get("max_input_tokens", DEFAULT_CONTEXT_WINDOW)
    except Exception:
        return DEFAULT_CONTEXT_WINDOW


def _estimate_tokens(messages: list[dict], model_name: str) -> int:
    try:
        import litellm

        return litellm.token_counter(model=model_name, messages=messages)
    except Exception:
        total_chars = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(json.dumps(block))
                    elif isinstance(block, str):
                        total_chars += len(block)
        return total_chars // 4


def _is_orphaned_tool_message(msg: dict) -> bool:
    role = msg.get("role", "")
    if role == "tool":
        return True
    if role == "user":
        content = msg.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "tool_result":
                return True
    return False


async def _condense_message_history(
    messages: list[dict],
    is_anthropic: bool,
    stream: EventSourceStream,
) -> list[dict]:
    await stream.send(
        WorkflowCopilotCondensingUpdate(
            type=WorkflowCopilotStreamMessageType.CONDENSING,
            status="started",
        )
    )

    if len(messages) <= 5:
        await stream.send(
            WorkflowCopilotCondensingUpdate(
                type=WorkflowCopilotStreamMessageType.CONDENSING,
                status="completed",
            )
        )
        return messages

    first = messages[0]
    keep_count = 4
    while keep_count < len(messages) - 1 and _is_orphaned_tool_message(messages[-keep_count]):
        keep_count += 1
    keep_recent = messages[-keep_count:]
    middle = messages[1:-keep_count]

    summary_lines = ["[Condensed conversation history]"]
    for msg in middle:
        role = msg.get("role", "?")
        content = msg.get("content")
        if role == "assistant":
            if isinstance(content, list):
                tool_names = [b.get("name", "?") for b in content if b.get("type") == "tool_use"]
                if tool_names:
                    summary_lines.append(f"Called: {', '.join(tool_names)}")
            elif isinstance(content, str) and content:
                summary_lines.append(f"Assistant: {content[:100]}...")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                if tc_names and not (isinstance(content, list)):
                    summary_lines.append(f"Called: {', '.join(tc_names)}")
        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        try:
                            parsed = json.loads(result_content)
                            ok = parsed.get("ok", False)
                            status = "OK" if ok else f"Failed: {parsed.get('error', '?')[:80]}"
                            summary_lines.append(f"  Result: {status}")
                        except json.JSONDecodeError:
                            summary_lines.append(f"  Result: {result_content[:80]}")
        elif role == "tool":
            content_str = msg.get("content", "")
            try:
                parsed = json.loads(content_str)
                ok = parsed.get("ok", False)
                status = "OK" if ok else f"Failed: {parsed.get('error', '?')[:80]}"
                summary_lines.append(f"  Tool result: {status}")
            except (json.JSONDecodeError, TypeError):
                summary_lines.append(f"  Tool result: {str(content_str)[:80]}")

    condensed_text = "\n".join(summary_lines)

    if is_anthropic:
        condensed_msg: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": condensed_text}],
        }
    else:
        condensed_msg = {"role": "user", "content": condensed_text}

    result = [first, condensed_msg] + keep_recent

    await stream.send(
        WorkflowCopilotCondensingUpdate(
            type=WorkflowCopilotStreamMessageType.CONDENSING,
            status="completed",
        )
    )

    return result


async def run_copilot_agent(
    stream: EventSourceStream,
    organization_id: str,
    chat_request: Any,
    chat_history: list[WorkflowCopilotChatHistoryMessage],
    global_llm_context: str | None,
    debug_run_info_text: str,
    llm_api_handler: Any,
    api_key: str | None = None,
) -> AgentResult:
    chat_history_text = _format_chat_history(chat_history)

    ctx = AgentContext(
        organization_id=organization_id,
        workflow_id=chat_request.workflow_id,
        workflow_permanent_id=chat_request.workflow_permanent_id,
        workflow_yaml=chat_request.workflow_yaml or "",
        browser_session_id=None,
        stream=stream,
        api_key=api_key,
    )

    llm_key = _get_llm_key_for_copilot(llm_api_handler)
    is_anthropic = "ANTHROPIC" in llm_key
    tool_format = "anthropic" if is_anthropic else "openai"

    from skyvern.cli.mcp_tools import mcp as skyvern_mcp

    skyvern_alias_map = get_skyvern_mcp_alias_map()

    host = MCPHost()
    host.register_server(
        ServerRegistration(
            name="skyvern",
            transport=skyvern_mcp,
            policy=ToolPolicy(
                allowlist=frozenset(skyvern_alias_map.values()),
            ),
            alias_map=skyvern_alias_map,
        )
    )

    dispatcher = ToolDispatcher(host)
    register_all_tools(dispatcher)

    try:
        await host.connect_all()
        await host.discover_tools(
            native_names=dispatcher.get_native_tool_names(),
        )
        await dispatcher.resolve_schemas()
        tool_defs = dispatcher.get_tool_definitions(tool_format=tool_format)
    except Exception:
        await host.disconnect_all()
        raise

    caller = LLMCaller(llm_key=llm_key)
    model_name = LLMConfigRegistry.get_config(llm_key).model_name
    context_window = _get_context_window(llm_key)
    token_threshold = int(context_window * COMPACTION_THRESHOLD)
    tool_usage_guide = _build_tool_usage_guide(tool_defs)
    system_prompt = _build_system_prompt(
        workflow_yaml=chat_request.workflow_yaml or "",
        chat_history_text=chat_history_text,
        global_llm_context=global_llm_context or "",
        debug_run_info_text=debug_run_info_text,
        tool_usage_guide=tool_usage_guide,
    )

    initial_content = system_prompt + "\n\nUser message:\n" + chat_request.message
    caller.message_history = [
        {"role": "user", "content": [{"type": "text", "text": initial_content}]},
    ]

    LOG.info(
        "Starting copilot agent loop",
        workflow_permanent_id=chat_request.workflow_permanent_id,
        user_message_len=len(chat_request.message),
        llm_key=llm_key,
    )

    try:
        return await _run_agent_loop(
            caller=caller,
            ctx=ctx,
            dispatcher=dispatcher,
            tool_defs=tool_defs,
            stream=stream,
            organization_id=organization_id,
            chat_request=chat_request,
            global_llm_context=global_llm_context,
            is_anthropic=is_anthropic,
            model_name=model_name,
            token_threshold=token_threshold,
        )
    finally:
        await host.disconnect_all()


def _append_nudge(
    caller: LLMCaller,
    content: list[dict[str, Any]],
    nudge: str,
    is_anthropic: bool,
) -> None:
    if is_anthropic:
        caller.message_history.append({"role": "assistant", "content": content})
        caller.message_history.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
    else:
        text_content = _extract_text_from_content(content)
        caller.message_history.append({"role": "assistant", "content": text_content or ""})
        caller.message_history.append({"role": "user", "content": nudge})


async def _run_agent_loop(
    *,
    caller: LLMCaller,
    ctx: AgentContext,
    dispatcher: ToolDispatcher,
    tool_defs: list[dict[str, Any]],
    stream: EventSourceStream,
    organization_id: str,
    chat_request: Any,
    global_llm_context: str | None,
    is_anthropic: bool,
    model_name: str,
    token_threshold: int,
) -> AgentResult:
    last_workflow: Workflow | None = None
    last_workflow_yaml: str | None = None
    start_time = time.monotonic()
    consecutive_tool_tracker: list[str] = []
    tool_activity: list[dict[str, Any]] = []
    navigate_called = False
    observation_after_navigate = False
    _OBSERVATION_TOOLS = {"evaluate", "get_browser_screenshot", "click", "type_text", "run_blocks_and_collect_debug"}
    navigate_enforcement_done = False
    update_workflow_called = False
    workflow_persisted = False
    test_after_update_done = False
    post_update_nudge_count = 0

    for iteration in range(MAX_ITERATIONS):
        if await stream.is_disconnected():
            LOG.info("Client disconnected, stopping agent loop", iteration=iteration)
            return AgentResult(
                user_response="Request cancelled.",
                updated_workflow=last_workflow,
                global_llm_context=global_llm_context,
                workflow_yaml=last_workflow_yaml,
            )

        elapsed = time.monotonic() - start_time
        if elapsed > TOTAL_TIMEOUT_SECONDS:
            LOG.warning("Agent loop total timeout", elapsed=elapsed)
            return AgentResult(
                user_response="I ran out of time processing your request. Here's what I have so far.",
                updated_workflow=last_workflow,
                global_llm_context=global_llm_context,
                workflow_yaml=last_workflow_yaml,
            )

        LOG.info(
            "Agent loop iteration",
            iteration=iteration,
            message_count=len(caller.message_history),
            elapsed=round(elapsed, 1),
        )

        llm_start = time.monotonic()
        try:
            raw_response = await caller.call(
                prompt=None,
                prompt_name="workflow-copilot-agent",
                organization_id=organization_id,
                tools=tool_defs,
                raw_response=True,
                use_message_history=True,
            )
        except SkyvernContextWindowExceededError:
            LOG.warning("Context window exceeded, attempting emergency condensing")
            caller.message_history = await _condense_message_history(caller.message_history, is_anthropic, stream)
            try:
                raw_response = await caller.call(
                    prompt=None,
                    prompt_name="workflow-copilot-agent",
                    organization_id=organization_id,
                    tools=tool_defs,
                    raw_response=True,
                    use_message_history=True,
                )
            except SkyvernContextWindowExceededError:
                LOG.error("Context window still exceeded after condensing")
                return AgentResult(
                    user_response="The conversation became too long. Please start a new chat.",
                    updated_workflow=last_workflow,
                    global_llm_context=global_llm_context,
                    workflow_yaml=last_workflow_yaml,
                )
        llm_duration = time.monotonic() - llm_start
        LOG.info("LLM response received", iteration=iteration, duration=round(llm_duration, 1))

        content = raw_response.get("content", [])
        stop_reason = raw_response.get("stop_reason", "")

        if not content and "choices" in raw_response:
            choices = raw_response["choices"]
            if choices:
                msg = choices[0].get("message", {})
                text_content = msg.get("content", "")
                tool_calls_openai = msg.get("tool_calls", [])
                content = []
                if text_content:
                    content.append({"type": "text", "text": text_content})
                for tc in tool_calls_openai:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    )
                stop_reason = choices[0].get("finish_reason", "stop")
                if stop_reason == "tool_calls":
                    stop_reason = "tool_use"

        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            if navigate_called and not observation_after_navigate and not navigate_enforcement_done:
                navigate_enforcement_done = True
                LOG.info("Post-navigate enforcement: agent responded without observing page")
                nudge = (
                    "You navigated to a page but did not observe its content. "
                    "You MUST use evaluate, get_browser_screenshot, click, or type_text "
                    "to inspect the page before responding. Do NOT answer from memory."
                )
                _append_nudge(caller, content, nudge, is_anthropic)
                continue

            if update_workflow_called and not test_after_update_done:
                if post_update_nudge_count < MAX_POST_UPDATE_NUDGES:
                    post_update_nudge_count += 1
                    LOG.info(
                        "Post-update enforcement: agent responded without testing workflow",
                        nudge_count=post_update_nudge_count,
                    )
                    nudge = (
                        "You updated the workflow but did not test it. "
                        "You MUST call run_blocks_and_collect_debug to test at least the first block "
                        "before responding to the user. This verifies the workflow actually works."
                    )
                    _append_nudge(caller, content, nudge, is_anthropic)
                    continue
                LOG.warning(
                    "Post-update enforcement exhausted nudges, allowing response", nudge_count=post_update_nudge_count
                )
                update_workflow_called = False
                post_update_nudge_count = 0

            text = _extract_text_from_content(content)
            if not text:
                text = '{"type": "REPLY", "user_response": "I\'m not sure how to help with that. Could you rephrase?"}'
            action_data = _parse_final_response(text)
            user_response = action_data.get("user_response") or "Done."

            if action_data.get("type") == "REPLACE_WORKFLOW":
                LOG.warning("Agent used inline REPLACE_WORKFLOW instead of update_workflow tool")
                workflow_yaml = action_data.get("workflow_yaml", "")
                if workflow_yaml:
                    try:
                        last_workflow = process_workflow_yaml(
                            workflow_id=chat_request.workflow_id,
                            workflow_permanent_id=chat_request.workflow_permanent_id,
                            organization_id=organization_id,
                            workflow_yaml=workflow_yaml,
                        )
                    except (yaml.YAMLError, ValidationError, BaseWorkflowHTTPException) as e:
                        LOG.warning("Failed to process final workflow YAML", error=str(e))
                        user_response = (
                            f"{user_response}\n\n"
                            f"(Note: The proposed workflow had a validation error: {str(e)[:200]}. "
                            f"Please ask me to fix it.)"
                        )

            resp_type = action_data.get("type", "REPLY")
            if resp_type not in ("REPLY", "ASK_QUESTION", "REPLACE_WORKFLOW"):
                resp_type = "REPLY"

            llm_context_raw = action_data.get("global_llm_context")
            if isinstance(llm_context_raw, dict):
                try:
                    structured = StructuredContext.model_validate(llm_context_raw)
                except Exception:
                    structured = StructuredContext.from_json_str(global_llm_context)
            elif isinstance(llm_context_raw, str):
                structured = StructuredContext.from_json_str(llm_context_raw)
            else:
                structured = StructuredContext.from_json_str(global_llm_context)
            structured.merge_turn_summary(tool_activity)
            enriched_context = structured.to_json_str()

            return AgentResult(
                user_response=str(user_response),
                updated_workflow=last_workflow,
                global_llm_context=enriched_context or None,
                response_type=resp_type,
                workflow_yaml=last_workflow_yaml,
                workflow_was_persisted=workflow_persisted,
            )

        if is_anthropic:
            caller.message_history.append({"role": "assistant", "content": content})
        else:
            openai_msg: dict[str, Any] = {"role": "assistant", "content": None}
            text_parts = [b["text"] for b in content if b.get("type") == "text"]
            if text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            openai_tool_calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    openai_tool_calls.append(
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        }
                    )
            if openai_tool_calls:
                openai_msg["tool_calls"] = openai_tool_calls
            caller.message_history.append(openai_msg)

        tool_results: list[dict[str, Any]] = []
        screenshots_for_llm: list[str] = []
        for tool_block in tool_use_blocks:
            if await stream.is_disconnected():
                LOG.info("Client disconnected during tool execution", iteration=iteration)
                return AgentResult(
                    user_response="Request cancelled.",
                    updated_workflow=last_workflow,
                    global_llm_context=global_llm_context,
                    workflow_yaml=last_workflow_yaml,
                )

            tool_name = tool_block["name"]
            tool_input = tool_block.get("input", {})
            tool_id = tool_block["id"]

            LOG.info(
                "Executing tool",
                tool_name=tool_name,
                iteration=iteration,
                tool_id=tool_id,
            )

            await stream.send(
                WorkflowCopilotToolCallUpdate(
                    type=WorkflowCopilotStreamMessageType.TOOL_CALL,
                    tool_name=tool_name,
                    tool_input={k: v for k, v in tool_input.items() if k != "workflow_yaml"},
                    iteration=iteration,
                    tool_call_id=tool_id,
                )
            )

            consecutive_tool_tracker.append(tool_name)
            is_looping = (
                len(consecutive_tool_tracker) >= MAX_CONSECUTIVE_SAME_TOOL
                and len(set(consecutive_tool_tracker[-MAX_CONSECUTIVE_SAME_TOOL:])) == 1
            )

            if is_looping:
                LOG.warning(
                    "Tool loop detected, skipping execution",
                    tool_name=tool_name,
                    count=MAX_CONSECUTIVE_SAME_TOOL,
                    iteration=iteration,
                )
                result = {
                    "ok": False,
                    "error": (
                        f"LOOP DETECTED: '{tool_name}' has been called "
                        f"{MAX_CONSECUTIVE_SAME_TOOL} times consecutively. "
                        f"This tool will not run again. Use a DIFFERENT tool "
                        f"to continue, or produce your final JSON response."
                    ),
                }
            elif not dispatcher.has_tool(tool_name):
                result = {"ok": False, "error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_timeout = dispatcher.get_timeout(tool_name) or TOOL_TIMEOUT_SECONDS
                    dispatch_result = await asyncio.wait_for(
                        dispatcher.dispatch(tool_name, tool_input, ctx),
                        timeout=tool_timeout,
                    )
                    result = dispatch_result.copilot_result
                except asyncio.TimeoutError:
                    result = {"ok": False, "error": f"Tool {tool_name} timed out after {tool_timeout}s"}
                except Exception as e:
                    LOG.error("Tool execution error", tool_name=tool_name, error=str(e), exc_info=True)
                    result = {"ok": False, "error": f"Tool error: {e}"}

            if tool_name == "update_workflow" and result.get("ok") and "_workflow" in result:
                wf_obj = result["_workflow"]
                if not isinstance(wf_obj, Workflow):
                    raise TypeError(f"Expected Workflow, got {type(wf_obj)}")
                last_workflow = wf_obj
                last_workflow_yaml = ctx.workflow_yaml or None
                wf = wf_obj
                has_blocks = bool(wf.workflow_definition and wf.workflow_definition.blocks)
                update_workflow_called = has_blocks
                if has_blocks:
                    workflow_persisted = True
                test_after_update_done = False
                post_update_nudge_count = 0

            if (
                tool_name == "run_blocks_and_collect_debug"
                and not result.get("ok", False)
                and last_workflow is not None
            ):
                last_workflow = None

            if tool_name == "navigate_browser" and result.get("ok"):
                navigate_called = True
                observation_after_navigate = False
            elif tool_name in _OBSERVATION_TOOLS:
                observation_after_navigate = True

            if tool_name == "run_blocks_and_collect_debug":
                test_after_update_done = True
                update_workflow_called = False
                post_update_nudge_count = 0

            summary = _summarize_tool_result(tool_name, result)
            success = result.get("ok", False)

            activity_entry: dict[str, Any] = {"tool": tool_name, "summary": summary}
            if tool_name in ("run_blocks_and_collect_debug", "get_run_results") and result.get("ok"):
                data = result.get("data") or {}
                blocks = data.get("blocks", []) if isinstance(data, dict) else []
                output_parts = []
                for b in blocks:
                    if b.get("output") or b.get("extracted_data"):
                        out = b.get("output") or b.get("extracted_data")
                        out_str = json.dumps(out, default=str) if not isinstance(out, str) else out
                        if len(out_str) > 500:
                            out_str = out_str[:500] + "..."
                        output_parts.append(f"{b.get('label', '?')}: {out_str}")
                if output_parts:
                    activity_entry["output_preview"] = "; ".join(output_parts)
            tool_activity.append(activity_entry)

            await stream.send(
                WorkflowCopilotToolResultUpdate(
                    type=WorkflowCopilotStreamMessageType.TOOL_RESULT,
                    tool_name=tool_name,
                    success=success,
                    summary=summary,
                    iteration=iteration,
                    tool_call_id=tool_id,
                )
            )

            screenshot_b64 = None
            data = result.get("data")
            if isinstance(data, dict):
                screenshot_b64 = data.get("screenshot_base64")

            sanitized = _sanitize_tool_result_for_llm(tool_name, result)
            text_content = json.dumps(sanitized)

            if screenshot_b64 and is_anthropic:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {"type": "text", "text": text_content},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            },
                        ],
                    }
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": text_content,
                    }
                )
                if screenshot_b64:
                    screenshots_for_llm.append(screenshot_b64)

        if is_anthropic:
            caller.message_history.append({"role": "user", "content": tool_results})
        else:
            for tr in tool_results:
                caller.message_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"],
                    }
                )
            if screenshots_for_llm:
                image_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": "Browser screenshot from the tool call above:"},
                ]
                for b64 in screenshots_for_llm:
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                caller.message_history.append({"role": "user", "content": image_parts})
                screenshots_for_llm.clear()

        if len(consecutive_tool_tracker) >= 2 and consecutive_tool_tracker[-1] != consecutive_tool_tracker[-2]:
            consecutive_tool_tracker = [consecutive_tool_tracker[-1]]

        estimated_tokens = _estimate_tokens(caller.message_history, model_name)
        if estimated_tokens > token_threshold:
            LOG.info(
                "Condensing conversation",
                estimated_tokens=estimated_tokens,
                threshold=token_threshold,
                message_count=len(caller.message_history),
            )
            caller.message_history = await _condense_message_history(caller.message_history, is_anthropic, stream)

    LOG.warning("Agent loop exhausted max iterations", max_iterations=MAX_ITERATIONS)
    return AgentResult(
        user_response="I've reached the maximum number of steps. Here's what I have so far.",
        updated_workflow=last_workflow,
        global_llm_context=global_llm_context,
        workflow_yaml=last_workflow_yaml,
    )
