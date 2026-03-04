"""Tests for post-update_workflow test enforcement logic."""

from __future__ import annotations

from typing import Any

MAX_POST_UPDATE_NUDGES = 2


def _simulate_enforcement(
    events: list[dict[str, Any]],
) -> list[str]:
    """Replay a sequence of tool/response events through the enforcement logic.

    Each event is a dict with:
      - type: "tool_result" | "final_response"
      - tool_name: str (for tool_result)
      - ok: bool (for tool_result)
      - has_blocks: bool (for update_workflow results)

    Returns a list of actions taken: "nudge" or "respond".
    """
    update_workflow_called = False
    test_after_update_done = False
    post_update_nudge_count = 0
    actions: list[str] = []

    for event in events:
        if event["type"] == "tool_result":
            tool_name = event["tool_name"]
            ok = event.get("ok", True)

            if tool_name == "update_workflow" and ok:
                has_blocks = event.get("has_blocks", True)
                update_workflow_called = has_blocks
                test_after_update_done = False
                post_update_nudge_count = 0

            if tool_name == "run_blocks_and_collect_debug":
                test_after_update_done = True
                update_workflow_called = False
                post_update_nudge_count = 0

        elif event["type"] == "final_response":
            if update_workflow_called and not test_after_update_done:
                if post_update_nudge_count < MAX_POST_UPDATE_NUDGES:
                    post_update_nudge_count += 1
                    actions.append("nudge")
                else:
                    update_workflow_called = False
                    post_update_nudge_count = 0
                    actions.append("respond")
            else:
                actions.append("respond")

    return actions


class TestPostUpdateEnforcement:
    """Test the post-update_workflow enforcement state machine."""

    def test_nudge_when_skipping_test(self) -> None:
        """Agent is nudged when it tries to respond after update_workflow
        without calling run_blocks_and_collect_debug."""
        events = [
            {"type": "tool_result", "tool_name": "get_block_schema", "ok": True},
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["nudge"]

    def test_no_nudge_when_no_blocks(self) -> None:
        """No nudge when update_workflow succeeds but workflow has no blocks."""
        events = [
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": False},
            {"type": "final_response"},
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["respond"]

    def test_no_nudge_when_test_called(self) -> None:
        """No nudge when run_blocks_and_collect_debug is called after update."""
        events = [
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "tool_result", "tool_name": "run_blocks_and_collect_debug", "ok": True},
            {"type": "final_response"},
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["respond"]

    def test_re_enforcement_on_second_update(self) -> None:
        """Nudge fires again after a second update_workflow without testing."""
        events = [
            # First update + test (no nudge)
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "tool_result", "tool_name": "run_blocks_and_collect_debug", "ok": True},
            {"type": "final_response"},
            # Second update without test (nudge)
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["respond", "nudge"]

    def test_no_nudge_on_failed_update(self) -> None:
        """No nudge when update_workflow fails (ok=False)."""
        events = [
            {"type": "tool_result", "tool_name": "update_workflow", "ok": False, "has_blocks": True},
            {"type": "final_response"},
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["respond"]

    def test_nudge_only_fires_once_per_update(self) -> None:
        """After a nudge, agent calls test, then responds successfully."""
        events = [
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},  # nudge
            # Agent calls run_blocks_and_collect_debug after nudge
            {"type": "tool_result", "tool_name": "run_blocks_and_collect_debug", "ok": True},
            {"type": "final_response"},  # respond
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["nudge", "respond"]

    def test_nudge_then_respond_without_test(self) -> None:
        """After MAX_POST_UPDATE_NUDGES nudges without test, agent is allowed
        to respond (bounded to avoid infinite loops)."""
        events = [
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},  # nudge 1
            {"type": "final_response"},  # nudge 2
            {"type": "final_response"},  # respond (nudges exhausted)
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["nudge", "nudge", "respond"]

    def test_nudge_counter_resets_on_test(self) -> None:
        """After 1 nudge, agent calls test, then second update without test
        gets nudged again (counter was reset)."""
        events = [
            # First update, 1 nudge, then test
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},  # nudge 1
            {"type": "tool_result", "tool_name": "run_blocks_and_collect_debug", "ok": True},
            {"type": "final_response"},  # respond
            # Second update without test — counter was reset, so nudge fires
            {"type": "tool_result", "tool_name": "update_workflow", "ok": True, "has_blocks": True},
            {"type": "final_response"},  # nudge 1 (fresh counter)
        ]
        actions = _simulate_enforcement(events)
        assert actions == ["nudge", "respond", "nudge"]
