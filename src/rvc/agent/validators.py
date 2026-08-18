"""Action validation - the safety boundary between the model and the robot.

A VLA is a neural network. It can emit NaN, a 40x-too-large delta, or a
gripper value that oscillates every step. None of that should ever reach an
actuator. Everything here is cheap, deterministic and logged.

`validate` returns (safe_action, ok, note):
  * ok=False means the action was REJECTED outright -> the agent raises
    FailureKind.UNSAFE_ACTION and re-plans.
  * ok=True with a non-empty note means the action was CLAMPED and is safe to
    execute; the clamp is still recorded.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from rvc.types import Action


@dataclass
class SafetyLimits:
    max_delta: float = 1.0  # per-dim magnitude ceiling (normalized units)
    max_norm: float = 1.75  # ceiling on ||dxyz||, stops diagonal over-speed
    max_rate: float = 1.2  # max change vs previous action, per dim
    gripper_chatter_window: int = 6
    gripper_chatter_max_flips: int = 4

    # Workspace bounds are ENV-SPECIFIC and must never be hardcoded here.
    # Hardcoding TabletopSim's box shipped a real bug: on LIBERO the Panda's
    # end-effector sits at z=0.91, far outside a 0.35 m ceiling, so the clip
    # asked for dz = (0.35 - 0.91)/0.035 = -16 every single step.
    # None means "this env does not declare bounds - skip the check".
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None

    @classmethod
    def for_env(cls, env: object, **kw) -> SafetyLimits:
        """Build limits from an env's declared `workspace_bounds`, if any."""
        bounds = getattr(env, "workspace_bounds", None)
        if bounds is None:
            return cls(**kw)
        lo, hi = bounds
        return cls(workspace_min=tuple(lo), workspace_max=tuple(hi), **kw)


class ActionValidator:
    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self._prev: np.ndarray | None = None
        self._grip_history: list[float] = []
        self.rejections = 0
        self.clamps = 0

    def reset(self) -> None:
        self._prev = None
        self._grip_history.clear()

    def validate(
        self, action: Action, ee_xyz: np.ndarray | None = None, step_scale: float = 0.035
    ) -> tuple[Action, bool, str]:
        v = np.asarray(action.vector, dtype=np.float32).copy()
        notes: list[str] = []

        # 1. Non-finite values are never recoverable - reject.
        if not np.all(np.isfinite(v)):
            self.rejections += 1
            return Action.zeros(), False, "REJECT: action contains NaN/Inf"

        # 2. Per-dimension magnitude.
        over = np.abs(v[:6]) > self.limits.max_delta
        if over.any():
            v[:6] = np.clip(v[:6], -self.limits.max_delta, self.limits.max_delta)
            notes.append(f"clamped {int(over.sum())} dim(s) to +-{self.limits.max_delta}")

        # 3. Translation speed ceiling.
        n = float(np.linalg.norm(v[:3]))
        if n > self.limits.max_norm:
            v[:3] *= self.limits.max_norm / n
            notes.append(f"scaled ||dxyz|| {n:.2f} -> {self.limits.max_norm}")

        # 4. Rate limit vs the previous command (jerk guard).
        if self._prev is not None:
            d = v[:6] - self._prev[:6]
            hot = np.abs(d) > self.limits.max_rate
            if hot.any():
                v[:6] = self._prev[:6] + np.clip(d, -self.limits.max_rate, self.limits.max_rate)
                notes.append(f"rate-limited {int(hot.sum())} dim(s)")

        # 5. Gripper must be a clean binary command.
        g = float(np.clip(v[6], 0.0, 1.0))
        v[6] = 1.0 if g > 0.5 else 0.0
        self._grip_history.append(v[6])
        w = self._grip_history[-self.limits.gripper_chatter_window :]
        flips = sum(1 for a, b in itertools.pairwise(w) if a != b)
        if flips >= self.limits.gripper_chatter_max_flips:
            self.rejections += 1
            return Action.zeros(), False, f"REJECT: gripper chattering ({flips} flips in {len(w)})"

        # 6. Predicted pose must stay inside the workspace box - but only if
        #    this env actually declared one. Skipping is safer than guessing:
        #    a wrong box turns every step into a full-scale correction toward
        #    an imaginary boundary.
        if (
            ee_xyz is not None
            and self.limits.workspace_min is not None
            and self.limits.workspace_max is not None
        ):
            ee = np.asarray(ee_xyz, dtype=np.float32)
            nxt = ee + v[:3] * step_scale
            lo = np.asarray(self.limits.workspace_min, dtype=np.float32)
            hi = np.asarray(self.limits.workspace_max, dtype=np.float32)
            if np.any(nxt < lo - 1e-6) or np.any(nxt > hi + 1e-6):
                allowed = (np.clip(nxt, lo, hi) - ee) / step_scale
                # Re-clamp: if the EE is already outside the box, the correction
                # needed to get back can exceed the per-step magnitude ceiling.
                # Requesting a 16x action to "fix" that would be worse than the
                # violation. Move as far back as one legal step allows.
                v[:3] = np.clip(allowed, -self.limits.max_delta, self.limits.max_delta)
                notes.append("clipped to workspace bounds")

        # Final invariant: whatever the earlier stages did, nothing leaves this
        # function outside the declared magnitude ceiling.
        v[:6] = np.clip(v[:6], -self.limits.max_delta, self.limits.max_delta)

        self._prev = v.copy()
        if notes:
            self.clamps += 1
        return Action(v), True, "; ".join(notes)
