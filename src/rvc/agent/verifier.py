"""VERIFY - did the sub-goal actually happen?

Separated from the state machine on purpose. "Did the grasp succeed?" is a
perception/state question; "what do I do about it?" is a control-flow question.
Keeping them apart is what lets you swap in a learned success classifier later
without touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvc.agent.planner import Subgoal
from rvc.types import FailureKind, Observation

Z_TRAVEL = 0.185
Z_GRASP_OK = 0.050
Z_PLACE_OK = 0.105
XY_OK = 0.020
XY_BOX_OK = 0.024


@dataclass(slots=True)
class VerifyResult:
    complete: bool
    failure: FailureKind
    note: str


class TabletopVerifier:
    """Uses simulator ground truth. In the real world this is where a success
    classifier, force/torque sensing or a wrist camera check would live."""

    name = "tabletop-groundtruth"

    def check(
        self,
        subgoal: Subgoal,
        obs: Observation,
        info: dict,
        steps_in_subgoal: int,
        prev_holding: bool,
    ) -> VerifyResult:
        p = obs.privileged
        ee = np.asarray(p["ee"], dtype=np.float32)
        block = np.asarray(p["block"], dtype=np.float32)
        box = np.asarray(p["box"], dtype=np.float32)
        holding = bool(p["holding"])
        event = info.get("event")

        xy_block = float(np.linalg.norm(ee[:2] - block[:2]))
        xy_box = float(np.linalg.norm(ee[:2] - box))

        # --- hard failure signals from the environment ----------------------
        if event == "grasp_slip":
            return VerifyResult(False, FailureKind.GRASP_SLIP, "物体在运输途中滑落")
        if event == "dropped":
            return VerifyResult(False, FailureKind.GRASP_SLIP, "在过高位置松开，物体掉落")
        if prev_holding and not holding and subgoal.id in ("lift", "transport", "lower"):
            return VerifyResult(False, FailureKind.GRASP_SLIP, "夹持状态意外丢失")
        if subgoal.id == "grasp" and event == "grasp_failed":
            return VerifyResult(False, FailureKind.GRASP_FAILED, "夹爪闭合但未夹到物体")

        # --- per-subgoal completion predicates -------------------------------
        done, why = {
            "approach": (xy_block < XY_OK and abs(ee[2] - Z_TRAVEL) < 0.025,
                         f"xy误差={xy_block:.3f}m z={ee[2]:.3f}m"),
            "descend": (xy_block < XY_OK and ee[2] < Z_GRASP_OK,
                        f"xy误差={xy_block:.3f}m z={ee[2]:.3f}m"),
            "grasp": (holding, f"holding={holding}"),
            "lift": (holding and ee[2] > 0.15, f"holding={holding} z={ee[2]:.3f}m"),
            "transport": (holding and xy_box < XY_BOX_OK,
                          f"holding={holding} 到目标xy误差={xy_box:.3f}m"),
            "lower": (holding and ee[2] < Z_PLACE_OK, f"holding={holding} z={ee[2]:.3f}m"),
            "release": (not holding, f"holding={holding}"),
        }.get(subgoal.id, (bool(p.get("success")), "自定义子目标：以任务成功为准"))

        if done:
            return VerifyResult(True, FailureKind.NONE, why)

        # --- budget exhausted ------------------------------------------------
        if steps_in_subgoal >= subgoal.budget:
            kind = (
                FailureKind.GRASP_FAILED
                if subgoal.id == "grasp"
                else FailureKind.STALLED
            )
            return VerifyResult(False, kind, f"超出 {subgoal.budget} 步预算 ({why})")

        return VerifyResult(False, FailureKind.NONE, why)


class RewardVerifier:
    """Fallback for envs (LIBERO) that only expose scalar reward + done."""

    name = "reward-only"

    def check(
        self,
        subgoal: Subgoal,
        obs: Observation,
        info: dict,
        steps_in_subgoal: int,
        prev_holding: bool,
    ) -> VerifyResult:
        if info.get("reward", 0.0) > 0 or info.get("success"):
            return VerifyResult(True, FailureKind.NONE, "环境返回成功信号")
        if steps_in_subgoal >= subgoal.budget:
            return VerifyResult(False, FailureKind.STALLED, f"超出 {subgoal.budget} 步预算")
        return VerifyResult(False, FailureKind.NONE, "等待环境成功信号")
