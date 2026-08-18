"""LIBERO integration tests.

Skipped automatically when LIBERO is not installed, so `make test` stays green
on a machine that only did `make setup`. Run `make setup-libero` to enable.

These are slow by unit-test standards (env construction is ~2.3 s), so the
module builds one env and shares it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rvc.envs.libero_env import probe
from rvc.types import ACTION_DIM, Action

_ok, _why = probe()
pytestmark = pytest.mark.skipif(not _ok, reason=f"LIBERO unavailable: {_why}")

# Imported lazily: `LiberoEnv` itself is importable without LIBERO, but every
# test below is skipped when it is missing, so keep the module cheap.
if _ok:
    from rvc.envs.libero_env import LiberoEnv, list_tasks


@pytest.fixture(scope="module")
def env():
    e = LiberoEnv(task_suite="libero_spatial", task_index=0, max_steps=30)
    yield e
    e.close()


def test_suite_exposes_ten_spatial_tasks():
    tasks = list_tasks("libero_spatial")
    assert len(tasks) == 10
    assert all(isinstance(lang, str) and lang for _, lang in tasks)


def test_instruction_is_the_dataset_wording(env):
    # Not our phrasing - this is what OpenVLA must be fed verbatim.
    assert env.instruction == (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )


def test_reset_renders_two_real_cameras(env):
    obs = env.reset()
    assert obs.image.shape == (256, 256, 3) and obs.image.dtype == np.uint8
    assert obs.image.std() > 5, "blank frame - offscreen rendering is broken"
    assert obs.wrist_image is not None and obs.wrist_image.shape == (256, 256, 3)
    assert obs.instruction == env.instruction


def test_step_advances_and_reports_reward(env):
    env.reset()
    obs, reward, done, info = env.step(Action.zeros())
    assert obs.step == 1
    assert isinstance(reward, float) and "reward" in info and "success" in info
    assert not done


def test_timeout_terminates_and_is_labelled():
    e = LiberoEnv(task_index=0, max_steps=3)
    e.reset()
    for _ in range(3):
        obs, reward, done, info = e.step(Action.zeros())
    assert done and info.get("event") == "timeout"
    e.close()


def test_gripper_convention_matches_openvla_eval():
    """OpenVLA emits gripper in [0,1]; LIBERO's OSC wants [-1,+1] inverted.

    Composition of normalize_gripper_action + invert_gripper_action:
        0.0 (OpenVLA open)   -> 2*0-1 = -1 -> sign -1 -> invert -> +1
        1.0 (OpenVLA closed) -> 2*1-1 = +1 -> sign +1 -> invert -> -1
    Getting this backwards is the classic "the arm never grasps" bug.
    """
    open_cmd = LiberoEnv._to_libero(Action(np.array([0, 0, 0, 0, 0, 0, 0.0])))
    close_cmd = LiberoEnv._to_libero(Action(np.array([0, 0, 0, 0, 0, 0, 1.0])))
    assert open_cmd.shape == (ACTION_DIM,)
    assert open_cmd[-1] == 1.0
    assert close_cmd[-1] == -1.0
    assert open_cmd[-1] == -close_cmd[-1]


def test_libero_declares_no_workspace_box(env):
    """Regression: hardcoding TabletopSim's 0.35 m ceiling made the validator
    emit dz = -16 on every LIBERO step (the Panda EE sits near z = 0.91)."""
    from rvc.agent.validators import ActionValidator, SafetyLimits

    assert env.workspace_bounds is None
    limits = SafetyLimits.for_env(env)
    assert limits.workspace_min is None and limits.workspace_max is None

    obs = env.reset()
    ee = np.asarray(obs.privileged["ee"], dtype=np.float32)
    assert ee[2] > 0.5, "sanity: LIBERO's EE really is far above a tabletop box"
    a, ok, note = ActionValidator(limits).validate(Action.zeros(), ee)
    assert ok and np.all(np.abs(a.vector[:6]) <= 1.0) and "workspace" not in note


def test_mock_policy_refuses_to_drive_libero(env):
    """The scripted controller must return zeros, not crash, and not pretend."""
    from rvc.compat import check
    from rvc.policies.mock import ScriptedMockPolicy

    obs = env.reset()
    pol = ScriptedMockPolicy()
    assert not pol.can_drive(obs)
    assert np.all(pol.predict(obs).vector == 0.0)

    notes = check("mock", "libero")
    assert any(n.level == "error" for n in notes), "this pairing must be flagged"
