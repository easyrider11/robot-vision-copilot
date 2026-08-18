"""The ROS 2-style agent layer.

    IDLE -> PERCEIVE -> PLAN -> EXECUTE -> VERIFY -> RECOVER -> SUCCEEDED / FAILED

This is the half of the system that is NOT the neural network. OpenVLA answers
one question - "given this image and this sentence, what is the next 7-DoF
delta?" - and nothing else. Task decomposition, safety, failure detection,
retry budgets, logging and termination all live here.

In Stage 3 this exact class becomes the body of a ROS 2 node: PERCEIVE
subscribes to `/camera/image_raw`, EXECUTE publishes to the controller topic,
and the state enum is published on `/agent/state`. Nothing about the logic
changes, which is the point of keeping it framework-free here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from rvc.agent.planner import RuleBasedPlanner, Subgoal
from rvc.agent.validators import ActionValidator, SafetyLimits
from rvc.agent.verifier import TabletopVerifier, VerifyResult
from rvc.perception.detector import ColorDetector, Detection
from rvc.types import (
    Action,
    AgentState,
    EpisodeResult,
    FailureKind,
    Observation,
    StepRecord,
)


@dataclass
class AgentConfig:
    max_recoveries: int = 3
    max_total_steps: int = 220
    mode: str = "subgoal"  # "subgoal" (planner-decomposed) | "e2e" (raw instruction)
    target_label: str = "red_block"
    recover_lift_steps: int = 4
    perceive_required: bool = True


@dataclass
class Transition:
    step: int
    frm: AgentState
    to: AgentState
    reason: str


@dataclass
class AgentTrace:
    records: list[StepRecord] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    frames: list[tuple[int, np.ndarray]] = field(default_factory=list)
    recoveries: int = 0
    detections_missed: int = 0


class RobotAgent:
    """Framework-free agent loop. `on_event` receives every interesting moment
    so the terminal reporter (or a ROS 2 publisher, or a WebSocket) can render
    it without this class knowing anything about presentation."""

    def __init__(
        self,
        env: Any,
        policy: Any,
        *,
        planner: Any | None = None,
        validator: ActionValidator | None = None,
        verifier: Any | None = None,
        detector: Any | None = None,
        config: AgentConfig | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        collect_frames: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.planner = planner or RuleBasedPlanner()
        self.validator = validator or ActionValidator(SafetyLimits.for_env(env))
        self.verifier = verifier or TabletopVerifier()
        self.detector = detector or ColorDetector()
        self.cfg = config or AgentConfig()
        self.on_event = on_event or (lambda kind, payload: None)
        self.collect_frames = collect_frames

        self.state = AgentState.IDLE
        self.trace = AgentTrace()

    # -- helpers -------------------------------------------------------------

    def _goto(self, to: AgentState, reason: str) -> None:
        if to is not self.state:
            self.trace.transitions.append(Transition(self.total_steps, self.state, to, reason))
            self.on_event("transition", {"frm": self.state, "to": to, "reason": reason,
                                         "step": self.total_steps})
            self.state = to

    def _ee(self, obs: Observation) -> np.ndarray | None:
        p = obs.privileged.get("ee")
        return None if p is None else np.asarray(p, dtype=np.float32)

    def _holding(self, obs: Observation) -> bool:
        return bool(obs.privileged.get("holding", False))

    # -- main loop -----------------------------------------------------------

    def run(self, run_dir: str = "") -> EpisodeResult:
        t_start = time.time()
        obs = self.env.reset()
        self.validator.reset()

        instruction = self.env.instruction
        self.total_steps = 0
        self.plan: list[Subgoal] = []
        self.idx = 0
        self.steps_in_subgoal = 0
        self._plan_dirty = True
        self._last_info: dict = {"event": None}
        self._prev_holding = False
        self._last_execute_t = 0.0
        failure = FailureKind.NONE
        success = False

        self.on_event("episode_start", {
            "instruction": instruction,
            "env": getattr(self.env, "name", "?"),
            "policy": getattr(self.policy, "name", "?"),
            "mode": self.cfg.mode,
        })

        self._goto(AgentState.PERCEIVE, "episode start")

        while not self.state.terminal and self.total_steps < self.cfg.max_total_steps:
            if self.state is AgentState.PERCEIVE:
                obs, failure = self._do_perceive(obs)

            elif self.state is AgentState.PLAN:
                self._do_plan(instruction)

            elif self.state is AgentState.EXECUTE:
                obs, failure, success = self._do_execute(obs)

            elif self.state is AgentState.VERIFY:
                obs, failure, success = self._do_verify(obs)

            elif self.state is AgentState.RECOVER:
                obs, failure = self._do_recover(obs, failure)

        if not self.state.terminal:
            failure = FailureKind.TIMEOUT
            self._goto(AgentState.FAILED, f"达到全局步数上限 {self.cfg.max_total_steps}")

        result = EpisodeResult(
            task_id=getattr(self.env, "task_id", getattr(self.env, "name", "task")),
            instruction=instruction,
            backend=getattr(self.policy, "name", "?"),
            degraded=bool(getattr(self.policy, "degraded", False))
            or bool(getattr(self.env, "degraded", False)),
            degraded_reason="; ".join(
                r
                for r in (
                    getattr(self.policy, "degraded_reason", ""),
                    getattr(self.env, "degraded_reason", ""),
                )
                if r
            ),
            success=success or self.state is AgentState.SUCCEEDED,
            steps=self.total_steps,
            recoveries=self.trace.recoveries,
            final_state=self.state,
            failure=failure if self.state is AgentState.FAILED else FailureKind.NONE,
            wall_time_s=round(time.time() - t_start, 3),
            run_dir=run_dir,
            injected_fault=getattr(self.env, "inject", "none"),
        )
        self.on_event("episode_end", {"result": result})
        return result

    # -- states --------------------------------------------------------------

    def _do_perceive(self, obs: Observation) -> tuple[Observation, FailureKind]:
        det: Detection | None = self.detector.find(obs.image, self.cfg.target_label)
        holding = self._holding(obs)
        self._last_detection = det
        self.on_event("perceive", {"detection": det, "step": self.total_steps, "holding": holding})

        # A held object is expected to be occluded by the gripper, so only treat
        # a miss as TARGET_LOST when we are not carrying anything.
        if det is None and not holding and self.cfg.perceive_required:
            self.trace.detections_missed += 1
            self._goto(AgentState.RECOVER, f"检测不到 {self.cfg.target_label}")
            return obs, FailureKind.TARGET_LOST

        self._goto(AgentState.PLAN if self._plan_dirty else AgentState.EXECUTE,
                   "需要(重新)规划" if self._plan_dirty else "沿用现有计划")
        return obs, FailureKind.NONE

    def _do_plan(self, instruction: str) -> None:
        if self.cfg.mode == "e2e":
            self.plan = [Subgoal("task", instruction, self.cfg.max_total_steps,
                                 "端到端：整句指令直接交给动作模型")]
            self.idx = 0
        elif not self.plan:
            self.plan = self.planner.plan(instruction)
            self.idx = 0
        self._plan_dirty = False
        self.steps_in_subgoal = 0
        self.on_event("plan", {
            "planner": getattr(self.planner, "name", "?"),
            "subgoals": [(s.id, s.text) for s in self.plan],
            "index": self.idx,
        })
        self._goto(AgentState.EXECUTE, f"子目标 {self.idx}/{len(self.plan)}")

    def _do_execute(self, obs: Observation) -> tuple[Observation, FailureKind, bool]:
        sub = self.plan[self.idx]

        # Full-cycle latency: time since the previous EXECUTE tick began covers
        # one complete PERCEIVE -> PLAN -> EXECUTE -> VERIFY loop.
        now = time.perf_counter()
        cycle_ms = (now - self._last_execute_t) * 1000 if self._last_execute_t else 0.0
        self._last_execute_t = now

        # This is the ONLY place the action model is consulted. It receives an
        # image and a sentence - nothing else.
        policy_obs = replace(obs, instruction=sub.text) if self.cfg.mode == "subgoal" else obs
        t0 = time.perf_counter()
        raw_action = self.policy.predict(policy_obs)
        latency_ms = (time.perf_counter() - t0) * 1000

        ee = self._ee(obs)
        action, ok, note = self.validator.validate(raw_action, ee)

        self.on_event("action", {
            "step": self.total_steps, "subgoal": sub, "raw": raw_action, "action": action,
            "ok": ok, "note": note, "latency_ms": latency_ms,
        })

        if not ok:
            self._record(sub, action, False, note, 0.0, False, FailureKind.UNSAFE_ACTION,
                         obs, latency_ms)
            self._goto(AgentState.RECOVER, note)
            return obs, FailureKind.UNSAFE_ACTION, False

        self._prev_holding = self._holding(obs)
        obs, reward, done, info = self.env.step(action)
        self._last_info = info or {}
        self.total_steps += 1
        self.steps_in_subgoal += 1

        self._record(sub, action, True, note, reward, done, FailureKind.NONE, obs, latency_ms,
                     cycle_ms=cycle_ms)
        self._capture(obs, sub)

        self.on_event("step_result", {
            "step": self.total_steps, "reward": reward, "done": done,
            "event": self._last_info.get("event"), "obs": obs,
        })

        self._goto(AgentState.VERIFY, "执行完一步，检查子目标")
        return obs, FailureKind.NONE, bool(reward > 0)

    def _do_verify(self, obs: Observation) -> tuple[Observation, FailureKind, bool]:
        sub = self.plan[self.idx]
        if bool(obs.privileged.get("success")) or self._last_info.get("event") == "success":
            self.on_event("verify", {"subgoal": sub,
                                     "result": VerifyResult(True, FailureKind.NONE, "任务成功")})
            self._goto(AgentState.SUCCEEDED, "环境判定任务完成")
            return obs, FailureKind.NONE, True

        res: VerifyResult = self.verifier.check(
            sub, obs, self._last_info, self.steps_in_subgoal, self._prev_holding
        )
        self.on_event("verify", {"subgoal": sub, "result": res, "index": self.idx,
                                 "total": len(self.plan)})

        if res.failure is not FailureKind.NONE:
            self._goto(AgentState.RECOVER, f"{res.failure.value}: {res.note}")
            return obs, res.failure, False

        if res.complete:
            self.idx += 1
            self.steps_in_subgoal = 0
            if self.idx >= len(self.plan):
                # Plan exhausted but the env never declared success.
                if self._last_info.get("event") == "timeout":
                    self._goto(AgentState.FAILED, "环境超时")
                    return obs, FailureKind.TIMEOUT, False
                self.idx = len(self.plan) - 1
                self._goto(AgentState.RECOVER, "计划已执行完但任务未成功")
                return obs, FailureKind.STALLED, False
            self._goto(AgentState.PERCEIVE, f"子目标完成 -> 进入 {self.plan[self.idx].id}")
            return obs, FailureKind.NONE, False

        if self._last_info.get("event") == "timeout":
            self._goto(AgentState.FAILED, "环境超时")
            return obs, FailureKind.TIMEOUT, False

        self._goto(AgentState.PERCEIVE, "子目标未完成，继续")
        return obs, FailureKind.NONE, False

    def _do_recover(
        self, obs: Observation, failure: FailureKind
    ) -> tuple[Observation, FailureKind]:
        if self.trace.recoveries >= self.cfg.max_recoveries:
            self._goto(AgentState.FAILED, f"恢复次数用尽 ({self.cfg.max_recoveries})")
            return obs, failure

        self.trace.recoveries += 1
        attempt = self.trace.recoveries

        # 1. Physically make the robot safe first: open the jaws and back off
        #    upward. These are AGENT actions, not model actions - logged as such.
        for _ in range(self.cfg.recover_lift_steps):
            if self.total_steps >= self.cfg.max_total_steps:
                break
            retreat = Action(np.array([0, 0, 1.0, 0, 0, 0, 0.0], dtype=np.float32))
            safe, ok, note = self.validator.validate(retreat, self._ee(obs))
            if not ok:
                break
            obs, reward, done, info = self.env.step(safe)
            self._last_info = info or {}
            self.total_steps += 1
            self._record(
                Subgoal("recover", "retreat and re-perceive", 0),
                safe, True, note or "recovery retreat", reward, done, failure, obs, 0.0,
                source="recovery",
            )
            self._capture(obs, Subgoal("recover", "retreat and re-perceive", 0))

        # 2. Ask the planner where to resume from.
        self.idx, why = self.planner.replan(self.plan, self.idx, failure, attempt)
        self.steps_in_subgoal = 0
        self._plan_dirty = False
        self.validator.reset()

        self.on_event("recover", {
            "attempt": attempt, "max": self.cfg.max_recoveries,
            "failure": failure, "resume_index": self.idx, "why": why,
        })
        self._goto(AgentState.PERCEIVE, f"第 {attempt} 次恢复：{why}")
        return obs, FailureKind.NONE

    # -- logging -------------------------------------------------------------

    def _record(
        self, sub: Subgoal, action: Action, ok: bool, note: str, reward: float,
        done: bool, failure: FailureKind, obs: Observation, latency_ms: float,
        source: str = "", cycle_ms: float = 0.0,
    ) -> None:
        ee = self._ee(obs)
        self.trace.records.append(StepRecord(
            step=self.total_steps,
            state=self.state,
            subgoal=sub.id,
            instruction=sub.text,
            action=action.to_list(),
            action_source=source or getattr(self.policy, "name", "?"),
            validated=ok,
            validator_note=note,
            reward=float(reward),
            done=bool(done),
            failure=failure,
            ee_xyz=[] if ee is None else [round(float(v), 4) for v in ee],
            holding=self._holding(obs),
            latency_ms=round(latency_ms, 2),
            cycle_ms=round(cycle_ms, 3),
            frame=f"frames/step_{self.total_steps:04d}.png" if self.collect_frames else None,
        ))

    def _capture(self, obs: Observation, sub: Subgoal) -> None:
        if not self.collect_frames:
            return
        from rvc.perception.detector import draw_overlay

        dets = self.detector.detect(obs.image, ("red_block", "blue_box"))
        header = f"{self.total_steps:03d} {self.state.value} | {sub.id}"
        footer = f"hold={self._holding(obs)} | {sub.text[:44]}"
        self.trace.frames.append((self.total_steps, draw_overlay(obs.image, dets, header, footer)))
