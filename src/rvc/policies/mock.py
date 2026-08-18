"""Scripted mock policy - the DEGRADED stand-in for OpenVLA.

READ THIS BEFORE TRUSTING ANY NUMBER THIS PRODUCES
--------------------------------------------------
This is NOT a vision-language-action model. It does not look at pixels. It is a
proportional controller that reads privileged simulator state and dispatches on
keywords in the instruction string. It exists so the *system* - state machine,
validators, failure detection, recovery, logging, HTTP service - can be built
and understood today, on a laptop that cannot host a 7B VLA.

Every run that uses it is stamped `degraded=true` with this reason, in the
terminal banner, in `summary.json`, and in the `/infer` HTTP response. Swapping
in `OpenVLALocalPolicy` or `OpenVLARemotePolicy` changes this one object and
nothing else.

Why keyword dispatch on the instruction? Because it keeps the interface honest:
the mock receives exactly what OpenVLA would receive (an image and a sentence)
and returns exactly what OpenVLA would return (a 7-vector). The planner can
therefore drive either one without modification.
"""

from __future__ import annotations

import numpy as np

from rvc.envs.tabletop import STEP_SCALE
from rvc.types import Action, Gripper, Observation

# Servo set-points, in metres.
Z_TRAVEL = 0.185
Z_GRASP = 0.032
Z_PLACE = 0.085
XY_DEADBAND = 0.006
Z_DEADBAND = 0.006
KP = 0.9  # fraction of the residual to request per step


def _servo(current: float, target: float, deadband: float) -> float:
    err = target - current
    if abs(err) < deadband:
        return 0.0
    return float(np.clip(err / STEP_SCALE * KP, -1.0, 1.0))


class ScriptedMockPolicy:
    """Privileged-state P controller for `TabletopSim`."""

    name = "scripted-mock"
    degraded = True
    degraded_reason = (
        "OpenVLA-7B cannot run on this host (no CUDA GPU / insufficient VRAM+disk). "
        "Actions come from a scripted proportional controller with privileged simulator "
        "state, NOT from a vision-language-action model."
    )

    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def describe(self) -> str:
        return f"{self.name} (P-controller, kp={KP}, noise={self.noise})"

    # -- Policy protocol -----------------------------------------------------

    #: Privileged keys this controller servos on. Only TabletopSim supplies
    #: them; LIBERO deliberately exposes reward-based success instead.
    REQUIRED_KEYS = ("ee", "block", "box", "holding")

    def can_drive(self, obs: Observation) -> bool:
        return all(k in obs.privileged for k in self.REQUIRED_KEYS)

    def predict(self, obs: Observation) -> Action:
        p = obs.privileged
        if not self.can_drive(obs):
            # Nothing to servo on - most commonly because the env is LIBERO,
            # which has no notion of "the block" or "the box". Returning zeros
            # is the honest answer; `rvc.compat` warns about this pairing up
            # front so a 0% success rate is never mistaken for a model result.
            return Action.zeros()

        ee = np.asarray(p["ee"], dtype=np.float32)
        block = np.asarray(p["block"], dtype=np.float32)[:2]
        box = np.asarray(p["box"], dtype=np.float32)
        holding = bool(p["holding"])
        text = obs.instruction.lower()

        a = np.zeros(7, dtype=np.float32)
        a[6] = Gripper.CLOSED if holding else Gripper.OPEN

        # --- keyword dispatch, ordered most-specific first ------------------
        if "close the gripper" in text or "grasp" in text:
            a[0] = _servo(ee[0], float(block[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(block[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_GRASP, Z_DEADBAND)
            a[6] = Gripper.CLOSED

        elif "open the gripper" in text or "release" in text:
            a[6] = Gripper.OPEN

        elif "lift" in text:
            a[2] = _servo(ee[2], Z_TRAVEL, Z_DEADBAND)
            a[6] = Gripper.CLOSED

        elif "into the blue box" in text or ("lower" in text and "box" in text):
            a[0] = _servo(ee[0], float(box[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(box[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_PLACE, Z_DEADBAND)
            a[6] = Gripper.CLOSED

        elif "above the blue box" in text or "move to the blue box" in text:
            a[0] = _servo(ee[0], float(box[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(box[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_TRAVEL, Z_DEADBAND)
            a[6] = Gripper.CLOSED

        elif "descend" in text or "lower" in text:
            a[0] = _servo(ee[0], float(block[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(block[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_GRASP, Z_DEADBAND)
            a[6] = Gripper.OPEN

        elif "above the red block" in text or "move to the red block" in text:
            a[0] = _servo(ee[0], float(block[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(block[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_TRAVEL, Z_DEADBAND)
            a[6] = Gripper.OPEN

        else:
            # End-to-end fallback: no sub-goal given, so run the whole task as
            # one implicit script. This is the closest analogue to handing the
            # full instruction straight to OpenVLA.
            a = self._end_to_end(ee, block, box, holding)

        if self.noise > 0:
            a[:6] += self.rng.normal(0.0, self.noise, size=6).astype(np.float32)
            a[:6] = np.clip(a[:6], -1.0, 1.0)
        return Action(a)

    # -- implicit whole-task script -----------------------------------------

    @staticmethod
    def _end_to_end(
        ee: np.ndarray, block: np.ndarray, box: np.ndarray, holding: bool
    ) -> np.ndarray:
        a = np.zeros(7, dtype=np.float32)
        if not holding:
            over_block = float(np.linalg.norm(ee[:2] - block)) < 0.012
            a[0] = _servo(ee[0], float(block[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(block[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_GRASP if over_block else Z_TRAVEL, Z_DEADBAND)
            a[6] = Gripper.CLOSED if (over_block and ee[2] < 0.05) else Gripper.OPEN
        else:
            over_box = float(np.linalg.norm(ee[:2] - box)) < 0.012
            a[0] = _servo(ee[0], float(box[0]), XY_DEADBAND)
            a[1] = _servo(ee[1], float(box[1]), XY_DEADBAND)
            a[2] = _servo(ee[2], Z_PLACE if over_box else Z_TRAVEL, Z_DEADBAND)
            a[6] = Gripper.OPEN if (over_box and ee[2] < 0.10) else Gripper.CLOSED
        return a
