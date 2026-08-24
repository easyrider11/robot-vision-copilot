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


def test_smolvla_gripper_roundtrip_is_lossless():
    """SmolVLA emits LIBERO env-space gripper (+1 close / -1 open). The client
    converts to the repo contract ((1-g)/2, OpenVLA convention 1=open) and
    LiberoEnv._to_libero converts back. The full chain must be the identity on
    the saturated values the model actually emits."""
    import numpy as np

    from rvc.envs.libero_env import LiberoEnv
    from rvc.types import Action

    for g_model, g_env_expected in ((1.0, 1.0), (-1.0, -1.0), (0.999, 1.0), (-0.98, -1.0)):
        g_contract = float(np.clip((1.0 - g_model) / 2.0, 0.0, 1.0))
        vec = np.zeros(7, dtype=np.float32)
        vec[6] = g_contract
        out = LiberoEnv._to_libero(Action(vec))
        assert out[6] == g_env_expected, (g_model, g_contract, out[6])


def test_smolvla_state8_layout_matches_libero_dataset():
    """The lerobot/libero training set built its 8-dim state as
    eef_pos(3) + robosuite quat2axisangle(eef_quat)(3) + gripper_qpos(2).
    Serving the model anything else silently shifts its input distribution."""
    import math

    import numpy as np

    from rvc.policies.smolvla_remote import libero_state8, quat_to_axisangle
    from rvc.types import Observation

    # identity quat -> zero rotation; 90 deg about z -> [0, 0, pi/2]
    assert np.allclose(quat_to_axisangle(np.array([0.0, 0.0, 0.0, 1.0])), 0.0)
    half = math.sqrt(0.5)
    aa = quat_to_axisangle(np.array([0.0, 0.0, half, half]))
    assert np.allclose(aa, [0.0, 0.0, math.pi / 2], atol=1e-6)

    obs = Observation(
        image=np.zeros((256, 256, 3), np.uint8), instruction="x", step=0,
        proprio=np.array([0.1, -0.2, 0.9], np.float32),
        privileged={"ee_quat": [0.0, 0.0, half, half], "gripper_qpos": [0.03, -0.03]},
    )
    s = libero_state8(obs)
    assert s.shape == (8,)
    assert np.allclose(s[:3], [0.1, -0.2, 0.9])
    assert np.allclose(s[3:6], [0.0, 0.0, math.pi / 2], atol=1e-6)
    assert np.allclose(s[6:], [0.03, -0.03])


def test_smolvla_client_refuses_when_server_is_down():
    """No silent degradation: an unreachable server must raise PolicyUnavailable
    with a fix hint, never fall back to something that isn't the model."""
    import pytest

    from rvc.policies.base import PolicyUnavailable
    from rvc.policies.smolvla_remote import SmolVLARemotePolicy

    with pytest.raises(PolicyUnavailable, match="smolvla-serve"):
        SmolVLARemotePolicy(url="http://127.0.0.1:1")  # nothing listens on port 1
