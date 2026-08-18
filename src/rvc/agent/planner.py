"""Task decomposition.

Two planners, one interface:

  * `RuleBasedPlanner`  - the default. Deterministic, no network, no API key.
  * `LLMPlanner`        - the OPTIONAL high-level planner the brief asks for.
                          It is a strict add-on: nothing in Stage 1 or Stage 2
                          requires it, and if no `complete` callable is wired
                          in it silently defers to the rule-based plan.

Both emit the same thing: a list of `Subgoal`s whose `.text` is a natural
sentence. That sentence is what gets handed to the action model, so the exact
same plan drives the scripted mock and real OpenVLA.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from rvc.types import FailureKind


@dataclass(slots=True)
class Subgoal:
    id: str
    text: str  # fed verbatim to the policy as the instruction
    budget: int = 25  # max env steps before we call it STALLED
    note: str = ""


# The canonical pick-and-place decomposition, matching the brief's example
# ("定位 -> 接近 -> 抓取 -> 移动 -> 放置").
PICK_PLACE_PLAN: list[Subgoal] = [
    Subgoal("approach", "move above the red block", 30, "定位 + 接近：对齐 xy，保持安全高度"),
    Subgoal("descend", "descend to the red block", 20, "下降到抓取高度"),
    Subgoal("grasp", "close the gripper on the red block", 6, "闭合夹爪"),
    Subgoal("lift", "lift the red block", 14, "抬起，离开桌面"),
    Subgoal("transport", "move above the blue box", 30, "搬运到目标上方"),
    Subgoal("lower", "lower the red block into the blue box", 16, "下降到放置高度"),
    Subgoal("release", "open the gripper to release the red block", 6, "松开夹爪"),
]


class RuleBasedPlanner:
    """Deterministic decomposition + deterministic recovery re-planning."""

    name = "rule-based"

    def plan(self, instruction: str) -> list[Subgoal]:
        return [Subgoal(s.id, s.text, s.budget, s.note) for s in PICK_PLACE_PLAN]

    def replan(
        self, plan: list[Subgoal], failed_index: int, failure: FailureKind, attempt: int
    ) -> tuple[int, str]:
        """Return (index to resume from, human-readable rationale).

        This is the whole recovery policy in one readable table. It is
        deliberately not clever: every branch is something you can reason about
        when a rollout goes wrong at 2am.
        """
        if failure is FailureKind.TARGET_LOST:
            return 0, "目标丢失 -> 退回第 0 步重新定位（先抬升视野，再重新感知）"
        if failure is FailureKind.GRASP_FAILED:
            return 0, "抓取失败 -> 张开夹爪、抬升，重新对齐后再抓一次"
        if failure is FailureKind.GRASP_SLIP:
            return 0, "运输中滑落 -> 物体已落回桌面，从定位重新开始"
        if failure is FailureKind.STALLED:
            back = max(0, failed_index - 1)
            return back, f"子目标 {failed_index} 超预算 -> 回退一步到 {back} 重试"
        if failure is FailureKind.UNSAFE_ACTION:
            return failed_index, "动作被安全校验拒绝 -> 原地重试（已夹取到安全范围）"
        return failed_index, "未知失败 -> 原地重试"


SYSTEM_PROMPT = """You are the high-level task planner for a robot arm.
Decompose the user's instruction into an ordered list of short, imperative
sub-instructions. Each one must be executable by a vision-language-action model
that only sees one camera image and one sentence.

Reply with JSON only:
{"subgoals": [{"id": "approach", "text": "move above the red block", "budget": 30}]}
"""

# JSON schemas handed to the LLM backend (structured outputs). Keeping them here,
# next to the parsers that consume them, is what makes "the LLM can only choose
# among typed, pre-approved skills" a checkable claim rather than a hope.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "budget": {"type": "integer"},
                },
                "required": ["id", "text", "budget"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}

REPLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "resume_at": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["resume_at", "why"],
    "additionalProperties": False,
}

REPLAN_PROMPT = """A robot arm is executing this ordered plan (subgoal ids in order):
{plan}

Subgoal #{failed_index} ("{failed_id}") just failed with failure = {failure}
(recovery attempt {attempt}). The rule-based policy suggests resuming at
"{suggested_id}" because: {suggested_why}

Choose the subgoal id to resume from. You may only pick an id from the plan
above, and it must be at or before the failed subgoal - never skip ahead.
Prefer resuming as late as is safe. Reply with JSON only:
{{"resume_at": "<id>", "why": "<one sentence>"}}
"""


def _accepts_kw(fn: Callable[..., str], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class LLMPlanner:
    """Optional LLM front-end. Never required; degrades to rule-based.

    `complete` is any callable `(prompt: str, schema: dict | None) -> str`; a
    plain `str -> str` callable also works (schema is then not passed). Wire it
    to the Anthropic SDK via `rvc.agent.llm_anthropic.make_anthropic_completer`,
    a local model, whatever - this module has no vendor dependency.

    Two jobs, both bounded:
      * plan()   - decompose an instruction into subgoals
      * replan() - after a failure, choose WHERE IN THE EXISTING PLAN to resume.
                   The LLM cannot invent actions here: it may only pick an
                   existing subgoal id at or before the failure point. Anything
                   else is rejected and the rule-based table decides.
    """

    def __init__(
        self,
        complete: Callable[..., str] | None = None,
        fallback: RuleBasedPlanner | None = None,
    ) -> None:
        self.complete = complete
        self.fallback = fallback or RuleBasedPlanner()
        self.last_error: str = ""
        self._pass_schema = complete is not None and _accepts_kw(complete, "schema")

    @property
    def enabled(self) -> bool:
        return self.complete is not None

    @property
    def name(self) -> str:
        return "llm" if self.enabled else "llm(fallback:rule-based)"

    # -- helpers --------------------------------------------------------------

    def _ask(self, prompt: str, schema: dict) -> dict:
        assert self.complete is not None
        raw = self.complete(prompt, schema=schema) if self._pass_schema else self.complete(prompt)
        return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])

    # -- planning -------------------------------------------------------------

    def plan(self, instruction: str) -> list[Subgoal]:
        if self.complete is None:
            self.last_error = "no LLM callable wired; using rule-based plan"
            return self.fallback.plan(instruction)
        try:
            data = self._ask(f"{SYSTEM_PROMPT}\n\nInstruction: {instruction}", PLAN_SCHEMA)
            goals = [
                Subgoal(
                    str(g["id"]), str(g["text"]), int(g.get("budget", 25)), "planned by LLM"
                )
                for g in data["subgoals"]
            ]
            if not goals:
                raise ValueError("LLM returned an empty plan")
            if len({g.id for g in goals}) != len(goals):
                raise ValueError("LLM plan has duplicate subgoal ids")
            self.last_error = ""
            return goals
        except Exception as exc:
            # A planner that fails must not take the robot down with it.
            self.last_error = f"LLM plan rejected ({type(exc).__name__}: {exc}); using rule-based"
            return self.fallback.plan(instruction)

    def replan(
        self, plan: list[Subgoal], failed_index: int, failure: FailureKind, attempt: int
    ) -> tuple[int, str]:
        idx, why = self.fallback.replan(plan, failed_index, failure, attempt)
        if self.complete is None:
            return idx, why
        ids = [s.id for s in plan]
        prompt = REPLAN_PROMPT.format(
            plan=" -> ".join(f"{i}:{s}" for i, s in enumerate(ids)),
            failed_index=failed_index,
            failed_id=ids[failed_index] if 0 <= failed_index < len(ids) else "?",
            failure=failure.value,
            attempt=attempt,
            suggested_id=ids[idx],
            suggested_why=why,
        )
        try:
            data = self._ask(prompt, REPLAN_SCHEMA)
            chosen = str(data["resume_at"])
            if chosen not in ids:
                raise ValueError(f"unknown subgoal id {chosen!r}")
            new_idx = ids.index(chosen)
            if new_idx > failed_index:
                raise ValueError(f"tried to skip ahead to {chosen!r} past the failure")
            self.last_error = ""
            return new_idx, f"LLM: {str(data.get('why', '')).strip()[:160]}"
        except Exception as exc:
            self.last_error = f"LLM replan rejected ({type(exc).__name__}: {exc}); using rule-based"
            return idx, why


@dataclass
class PlannerInfo:
    name: str
    note: str = ""
    subgoals: list[str] = field(default_factory=list)
