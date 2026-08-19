"""Core data types shared by every layer.

The whole project is built around three contracts:

    Observation  --(Policy)-->  Action  --(Env)-->  Observation'

    AgentState machine wraps that loop and decides when to PERCEIVE / PLAN /
    EXECUTE / VERIFY / RECOVER.

Keeping these types dependency-free (stdlib + numpy only) is what lets the
Stage-1 demo run on a laptop with no torch, no CUDA and no simulator installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

# OpenVLA (and LIBERO / RLDS-style manipulation datasets) use a 7-DoF action:
#   [dx, dy, dz, droll, dpitch, dyaw, gripper]
# The first six are normalized end-effector deltas in [-1, 1]; the last is the
# gripper command. OpenVLA's RLDS dataloader standardises the gripper to [0, 1]
# with **1 = open, 0 = closed** (see openvla `run_libero_eval.py`: "the
# dataloader flips the sign ... (0 = close, 1 = open)"). LIBERO's OSC
# controller instead wants [-1, +1] with -1 = open, +1 = closed. `rvc.envs`
# owns that conversion so policies never have to care.
#
# HISTORY: this contract originally said "0 = open, 1 = closed" - backwards.
# The tabletop sim, mock policy and visual servo were self-consistent with the
# wrong sign while `LiberoEnv._to_libero` was consistent with the right one,
# so a real OpenVLA output would have been inverted on the tabletop sim. Found
# 2026-08-19 while writing the LIBERO behaviour-cloning baseline, whose hdf5
# actions forced the convention to be written down precisely. The single
# source of truth is now the `Gripper` enum + `tests/test_contract.py`.
ACTION_DIM = 7
ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


class Gripper(float, Enum):
    OPEN = 1.0  # OpenVLA / RLDS convention
    CLOSED = 0.0


@dataclass(slots=True)
class Action:
    """A single 7-DoF end-effector delta command."""

    vector: np.ndarray  # shape (7,), float32

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if self.vector.shape[0] != ACTION_DIM:
            raise ValueError(f"action must have {ACTION_DIM} dims, got {self.vector.shape[0]}")

    @property
    def delta_xyz(self) -> np.ndarray:
        return self.vector[0:3]

    @property
    def delta_rpy(self) -> np.ndarray:
        return self.vector[3:6]

    @property
    def gripper(self) -> float:
        return float(self.vector[6])

    @classmethod
    def zeros(cls) -> Action:
        """All-zero vector. NOTE: under the contract gripper=0 means CLOSED, so
        this is not a neutral no-op - use `hold()` when you want "stop moving,
        keep the gripper as it is"."""
        return cls(np.zeros(ACTION_DIM, dtype=np.float32))

    @classmethod
    def hold(cls, gripper: float = 1.0) -> Action:
        """Zero motion with an explicit gripper command (default OPEN = 1.0)."""
        v = np.zeros(ACTION_DIM, dtype=np.float32)
        v[6] = float(gripper)
        return cls(v)

    def to_list(self) -> list[float]:
        return [round(float(v), 5) for v in self.vector]

    def pretty(self) -> str:
        parts = [f"{lbl}={v:+.3f}" for lbl, v in zip(ACTION_LABELS, self.vector, strict=True)]
        return "  ".join(parts)


@dataclass(slots=True)
class Observation:
    """What the robot sees and knows at one timestep."""

    image: np.ndarray  # (H, W, 3) uint8 RGB - the primary camera
    instruction: str  # natural-language task, fed to the VLA
    step: int = 0
    wrist_image: np.ndarray | None = None
    proprio: np.ndarray | None = None  # optional joint / EE state
    # Privileged simulator state. Used ONLY by the mock policy and by the
    # verifier; a real VLA never sees this. Kept separate on purpose so the
    # boundary between "what the model knows" and "what we know" stays visible.
    privileged: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent state machine
# ---------------------------------------------------------------------------


class AgentState(str, Enum):
    IDLE = "IDLE"
    PERCEIVE = "PERCEIVE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        return self in (AgentState.SUCCEEDED, AgentState.FAILED)


class FailureKind(str, Enum):
    NONE = "none"
    TARGET_LOST = "target_lost"  # perception can no longer find the object
    GRASP_FAILED = "grasp_failed"  # closed the gripper but nothing is held
    GRASP_SLIP = "grasp_slip"  # held it, then dropped it
    UNSAFE_ACTION = "unsafe_action"  # validator rejected the policy output
    STALLED = "stalled"  # no progress for N steps
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Logging records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepRecord:
    """One row of the episode log. Written to actions.jsonl."""

    step: int
    state: AgentState
    subgoal: str
    instruction: str
    action: list[float]
    action_source: str  # which policy produced it
    validated: bool
    validator_note: str
    reward: float
    done: bool
    failure: FailureKind
    ee_xyz: list[float]
    holding: bool
    latency_ms: float  # policy inference only
    cycle_ms: float = 0.0  # full control cycle (perceive->verify), wall clock
    frame: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["state"] = self.state.value
        d["failure"] = self.failure.value
        return json.dumps(d, ensure_ascii=False)


@dataclass
class EpisodeResult:
    task_id: str
    instruction: str
    backend: str
    degraded: bool
    degraded_reason: str
    success: bool
    steps: int
    recoveries: int
    final_state: AgentState
    failure: FailureKind
    wall_time_s: float
    run_dir: str
    injected_fault: str = "none"
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["final_state"] = self.final_state.value
        d["failure"] = self.failure.value
        return d
