"""Terminal reporting.

The brief asks the demo to "explain每一步：任务、观测、预测动作、执行结果" in a
human-readable way. That is what this file does and all it does - the agent
emits events, this renders them. Swap it for a ROS 2 publisher or a WebSocket
feed without touching any logic.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from rvc.types import AgentState, EpisodeResult, FailureKind

_ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _paint(code: str) -> Callable[[str], str]:
    def wrap(s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if _ENABLED else s

    return wrap


DIM = _paint("2")
B = _paint("1")
RED = _paint("31")
GRN = _paint("32")
YLW = _paint("33")
BLU = _paint("36")
MAG = _paint("35")

STATE_COLOR = {
    AgentState.IDLE: DIM,
    AgentState.PERCEIVE: BLU,
    AgentState.PLAN: MAG,
    AgentState.EXECUTE: GRN,
    AgentState.VERIFY: YLW,
    AgentState.RECOVER: RED,
    AgentState.SUCCEEDED: GRN,
    AgentState.FAILED: RED,
}
W = 78


def rule(title: str = "", ch: str = "─") -> str:
    if not title:
        return DIM(ch * W)
    pad = max(0, W - len(title) - 3)
    return DIM(f"{ch * 2} ") + B(title) + DIM(" " + ch * pad)


def banner(lines: list[str], title: str) -> str:
    out = [rule(title, "═")]
    out += [f"  {ln}" for ln in lines]
    out.append(rule("", "═"))
    return "\n".join(out)


class Reporter:
    """Consumes agent events. `style` is 'full' (default) or 'compact'."""

    def __init__(self, style: str = "full") -> None:
        self.style = style
        self._pending: dict[str, Any] = {}
        self._last_detection = None
        self._plan_len = 0
        self._idx = 0

    # -- event sink ----------------------------------------------------------

    def __call__(self, kind: str, p: dict) -> None:
        fn = getattr(self, f"_on_{kind}", None)
        if fn:
            fn(p)

    def _on_episode_start(self, p: dict) -> None:
        print()
        print(rule("EPISODE 开始", "═"))
        print(f"  任务 task        : {B(p['instruction'])}")
        print(f"  环境 env         : {p['env']}")
        print(f"  策略 policy      : {p['policy']}")
        print(f"  模式 mode        : {p['mode']}  "
              + DIM("(subgoal=规划器拆解子目标 / e2e=整句直接给动作模型)"))
        print(rule("", "═"))

    def _on_transition(self, p: dict) -> None:
        if self.style != "full":
            return
        frm, to = p["frm"], p["to"]
        if to in (AgentState.SUCCEEDED, AgentState.FAILED, AgentState.RECOVER):
            col = STATE_COLOR[to]
            print(f"  {DIM('状态')} {frm.value} {DIM('->')} {col(B(to.value))}  {DIM(p['reason'])}")

    def _on_plan(self, p: dict) -> None:
        self._plan_len = len(p["subgoals"])
        print(rule(f"PLAN · {p['planner']} · {self._plan_len} 个子目标"))
        for i, (sid, text) in enumerate(p["subgoals"]):
            mark = "▶" if i == p["index"] else " "
            print(f"  {mark} {i}. {B(sid):<24} {DIM(text)}")

    def _on_perceive(self, p: dict) -> None:
        self._last_detection = p["detection"]

    def _on_action(self, p: dict) -> None:
        self._pending = p

    def _on_verify(self, p: dict) -> None:
        self._pending["verify"] = p["result"]
        self._idx = p.get("index", self._idx)
        self._flush()

    def _on_step_result(self, p: dict) -> None:
        self._pending["step_result"] = p

    def _on_recover(self, p: dict) -> None:
        print()
        print(rule(f"RECOVER · 第 {p['attempt']}/{p['max']} 次", "─"))
        print(f"  失败类型 failure : {RED(B(p['failure'].value))}")
        print(f"  恢复策略 action  : {p['why']}")
        print(f"  恢复后从子目标   : #{p['resume_index']} 重新开始")
        print(rule("", "─"))

    def _on_episode_end(self, p: dict) -> None:
        self.summary(p["result"])

    # -- per-step rendering --------------------------------------------------

    def _flush(self) -> None:
        pend, self._pending = self._pending, {}
        if not pend or "subgoal" not in pend:
            return
        sr = pend.get("step_result") or {}
        obs = sr.get("obs")
        sub = pend["subgoal"]
        act = pend["action"]
        vr = pend.get("verify")
        step = sr.get("step", pend.get("step", 0))

        det = self._last_detection
        det_s = (
            f"{det.label} conf {det.confidence:.2f} @ ({det.center_world[0]:+.3f},"
            f"{det.center_world[1]:+.3f})"
            if det
            else RED("未检测到目标")
        )
        priv = (obs.privileged if obs is not None else {}) or {}
        ee = priv.get("ee", [0, 0, 0])
        hold = "✓" if priv.get("holding") else "✗"

        if self.style == "compact":
            ok = GRN("✓") if (vr and vr.complete) else DIM("·")
            print(
                f"  {ok} {step:>3}  {sub.id:<10} "
                f"a=[{' '.join(f'{v:+.2f}' for v in act.vector[:3])}|g{act.gripper:.0f}]  "
                f"ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f}) hold={hold}  "
                f"{DIM(vr.note if vr else '')}"
            )
            return

        idx_s = f"{self._idx + 1}/{self._plan_len}" if self._plan_len else "-"
        print()
        print(rule(f"step {step:03d} │ EXECUTE │ 子目标 {idx_s} · {sub.id}"))
        print(f"  {BLU('观测 obs')}    : 256×256 RGB · {det_s}")
        print(f"  {MAG('指令 instr')}  : \"{sub.text}\"   {DIM('← 交给动作模型的全部语言输入')}")
        print(
            f"  {GRN('动作 act')}    : "
            + "  ".join(f"{n}{v:+.2f}" for n, v in zip("xyz", act.vector[:3], strict=True))
            + f"  rpy[{','.join(f'{v:+.2f}' for v in act.vector[3:6])}]"
            + f"  grip={'CLOSE' if act.gripper < 0.5 else 'OPEN '}"
            + DIM(f"   ← {pend.get('note') or 'no clamp'} · {pend['latency_ms']:.1f}ms")
        )
        res = (
            f"ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) hold={hold} "
            f"r={sr.get('reward', 0.0):.2f}"
        )
        if vr:
            tag = GRN("子目标完成") if vr.complete else (
                RED(vr.failure.value) if vr.failure is not FailureKind.NONE else DIM("进行中")
            )
            res += f" │ {tag} {DIM(vr.note)}"
        ev = sr.get("event")
        if ev:
            res += f" │ {YLW('event=' + str(ev))}"
        print(f"  {YLW('结果 result')} : {res}")

    # -- final ---------------------------------------------------------------

    def summary(self, r: EpisodeResult) -> None:
        ok = r.success
        head = GRN(B(" SUCCEEDED ")) if ok else RED(B(" FAILED "))
        print()
        print(rule("EPISODE 结束", "═"))
        state_s = STATE_COLOR[r.final_state](r.final_state.value)
        print(f"  结果 outcome     : {head}   最终状态 {state_s}")
        print(f"  任务 task        : {r.instruction}")
        print(f"  步数 steps       : {r.steps}    恢复次数 recoveries: {r.recoveries}")
        print(f"  失败 failure     : {r.failure.value}")
        print(f"  注入故障 inject  : {r.injected_fault}")
        print(f"  耗时 wall time   : {r.wall_time_s}s")
        print(f"  动作来源 backend : {r.backend}")
        if r.degraded:
            print()
            print("  " + RED(B("⚠ 降级演示 DEGRADED DEMO — 这不是真实 OpenVLA 推理结果")))
            for line in r.degraded_reason.split("; "):
                if line.strip():
                    print("    " + YLW("· " + line.strip()))
        else:
            print("  " + GRN("✓ 真实 VLA 推理 (not degraded)"))
        print(f"  产物 artifacts   : {r.run_dir}")
        print(rule("", "═"))
        print()
