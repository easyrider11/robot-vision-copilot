"""LLMPlanner contract tests - no network, no SDK. A fake `complete` stands in
for the model so the *boundaries* get tested: what the planner accepts, what it
rejects, and that every rejection lands on the rule-based fallback."""

from __future__ import annotations

import json

from rvc.agent.planner import (
    PLAN_SCHEMA,
    REPLAN_SCHEMA,
    LLMPlanner,
    RuleBasedPlanner,
)
from rvc.types import FailureKind

RULE = RuleBasedPlanner()
RULE_PLAN = RULE.plan("pick up the red block and place it in the blue box")


def _fake(reply):
    """Build a schema-aware fake completer; records what it was asked."""
    calls = []

    def complete(prompt, schema=None):
        calls.append((prompt, schema))
        return reply(prompt, schema) if callable(reply) else reply

    complete.calls = calls
    return complete


# --- plan() -----------------------------------------------------------------


def test_no_completer_means_rule_based_and_says_so():
    p = LLMPlanner(complete=None)
    assert not p.enabled and "fallback" in p.name
    goals = p.plan("x")
    assert [g.id for g in goals] == [g.id for g in RULE_PLAN]
    assert "no LLM callable" in p.last_error


def test_valid_llm_plan_is_used_and_schema_is_passed():
    reply = json.dumps({"subgoals": [
        {"id": "a", "text": "move above the red block", "budget": 30},
        {"id": "b", "text": "descend to the red block", "budget": 20},
    ]})
    fake = _fake(reply)
    p = LLMPlanner(complete=fake)
    goals = p.plan("pick up the red block")
    assert [g.id for g in goals] == ["a", "b"]
    assert goals[0].note == "planned by LLM"
    assert fake.calls[0][1] is PLAN_SCHEMA, "structured-output schema must reach the backend"
    assert p.last_error == ""


def test_plain_str_to_str_completer_still_works():
    def old_style(prompt):  # no schema kwarg
        return '{"subgoals": [{"id": "only", "text": "open the gripper", "budget": 5}]}'

    goals = LLMPlanner(complete=old_style).plan("x")
    assert [g.id for g in goals] == ["only"]


def test_garbage_json_falls_back_to_rule_based():
    p = LLMPlanner(complete=_fake("sure! here is a plan: approach then grasp"))
    goals = p.plan("x")
    assert [g.id for g in goals] == [g.id for g in RULE_PLAN]
    assert "rejected" in p.last_error


def test_empty_or_duplicate_plan_is_rejected():
    p = LLMPlanner(complete=_fake('{"subgoals": []}'))
    assert [g.id for g in p.plan("x")] == [g.id for g in RULE_PLAN]
    dup = json.dumps({"subgoals": [
        {"id": "a", "text": "t", "budget": 1}, {"id": "a", "text": "u", "budget": 1}]})
    p = LLMPlanner(complete=_fake(dup))
    assert [g.id for g in p.plan("x")] == [g.id for g in RULE_PLAN]
    assert "duplicate" in p.last_error


def test_backend_exception_falls_back_to_rule_based():
    def boom(prompt, schema=None):
        raise ConnectionError("network down")

    p = LLMPlanner(complete=boom)
    assert [g.id for g in p.plan("x")] == [g.id for g in RULE_PLAN]
    assert "ConnectionError" in p.last_error


# --- replan() ---------------------------------------------------------------


def test_replan_accepts_an_earlier_existing_subgoal():
    fake = _fake('{"resume_at": "descend", "why": "the block is still under the gripper"}')
    p = LLMPlanner(complete=fake)
    idx, why = p.replan(RULE_PLAN, failed_index=2, failure=FailureKind.GRASP_FAILED, attempt=1)
    assert idx == 1 and why.startswith("LLM:")
    assert fake.calls[-1][1] is REPLAN_SCHEMA


def test_replan_never_skips_ahead_of_the_failure():
    fake = _fake('{"resume_at": "release", "why": "just finish"}')  # index 6 > failed 2
    p = LLMPlanner(complete=fake)
    idx, why = p.replan(RULE_PLAN, failed_index=2, failure=FailureKind.GRASP_FAILED, attempt=1)
    rule_idx, _ = RULE.replan(RULE_PLAN, 2, FailureKind.GRASP_FAILED, 1)
    assert idx == rule_idx and "skip ahead" in p.last_error


def test_replan_rejects_invented_subgoal_ids():
    fake = _fake('{"resume_at": "teleport_to_goal", "why": "faster"}')
    p = LLMPlanner(complete=fake)
    idx, _ = p.replan(RULE_PLAN, failed_index=3, failure=FailureKind.GRASP_SLIP, attempt=1)
    rule_idx, _ = RULE.replan(RULE_PLAN, 3, FailureKind.GRASP_SLIP, 1)
    assert idx == rule_idx and "unknown subgoal" in p.last_error


def test_replan_prompt_carries_the_rule_based_suggestion():
    fake = _fake('{"resume_at": "approach", "why": "ok"}')
    p = LLMPlanner(complete=fake)
    p.replan(RULE_PLAN, failed_index=4, failure=FailureKind.TARGET_LOST, attempt=2)
    prompt = fake.calls[-1][0]
    assert "target_lost" in prompt and "attempt 2" in prompt and '"approach"' in prompt
