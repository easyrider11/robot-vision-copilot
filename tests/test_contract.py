"""The 7-DoF action contract, pinned against OpenVLA's documented convention.

OpenVLA's RLDS dataloader standardises the gripper dimension to [0, 1] with
1 = open and 0 = closed; LIBERO's OSC controller wants [-1, +1] with -1 = open.
(openvla/experiments/robot/libero/run_libero_eval.py: "The dataloader flips the
sign of the gripper action to align with other datasets (0 = close, 1 = open),
so flip it back (-1 = open, +1 = close) before executing the action".)

This file exists because the contract was once written backwards and nothing
caught it: every component was internally consistent with *some* sign.
"""

from __future__ import annotations

import numpy as np

from rvc.agent.validators import ActionValidator, SafetyLimits
from rvc.envs.libero_env import LiberoEnv
from rvc.envs.tabletop import TabletopSim
from rvc.policies.mock import ScriptedMockPolicy
from rvc.types import Action, Gripper


def test_gripper_enum_follows_openvla():
    assert Gripper.OPEN == 1.0 and Gripper.CLOSED == 0.0


def test_openvla_open_maps_to_libero_open():
    # 1 (open) -> LIBERO -1 (open); 0 (closed) -> LIBERO +1 (closed)
    assert LiberoEnv._to_libero(Action.hold(Gripper.OPEN))[-1] == -1.0
    assert LiberoEnv._to_libero(Action.hold(Gripper.CLOSED))[-1] == +1.0


def test_tabletop_reads_the_same_convention():
    env = TabletopSim()
    env.step(Action.hold(Gripper.CLOSED))
    assert env.grip_closed
    env.step(Action.hold(Gripper.OPEN))
    assert not env.grip_closed


def test_mock_policy_emits_the_contract_convention():
    """The mock closes on the block and opens over the box - check the VALUES
    it emits, not just that the sim happens to agree with it."""
    env = TabletopSim()
    pol = ScriptedMockPolicy()
    obs = env.reset()
    # same subgoal order the planner uses: get there first, then close
    for _ in range(40):
        obs.instruction = "descend to the red block"
        a = pol.predict(obs)
        assert a.gripper == Gripper.OPEN == 1.0
        obs, *_ = env.step(a)
    info = {}
    for _ in range(5):
        obs.instruction = "close the gripper on the red block"
        a = pol.predict(obs)
        assert a.gripper == Gripper.CLOSED == 0.0
        obs, _, _, info = env.step(a)
        if info.get("event") == "grasped":
            break
    assert info.get("event") == "grasped"


def test_validator_rejection_keeps_last_gripper():
    v = ActionValidator(SafetyLimits())
    ee = np.zeros(3)
    closed = Action.hold(Gripper.CLOSED)
    v.validate(closed, ee)  # accepted, recorded
    bad = Action(np.array([np.nan, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))
    out, ok, note = v.validate(bad, ee)
    assert not ok and out.gripper == Gripper.CLOSED, "rejection must not re-open the gripper"
