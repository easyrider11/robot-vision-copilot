"""Stage 1 unit tests. Run with `make test`.

These pin down the properties that must not silently break: the action
contract, the safety validator, the honesty of the degradation reporting, and
the fact that every injected fault actually reaches RECOVER.
"""

from __future__ import annotations

import numpy as np
import pytest

from rvc.agent.state_machine import AgentConfig, RobotAgent
from rvc.agent.validators import ActionValidator, SafetyLimits
from rvc.envs.tabletop import TabletopSim
from rvc.perception.detector import ColorDetector
from rvc.policies.mock import ScriptedMockPolicy
from rvc.policies.registry import resolve_policy
from rvc.types import Action, AgentState, FailureKind

# --- action contract --------------------------------------------------------


def test_action_must_be_7dof():
    with pytest.raises(ValueError):
        Action(np.zeros(6))
    assert Action.zeros().vector.shape == (7,)


# --- validator --------------------------------------------------------------


def test_validator_rejects_nan():
    v = ActionValidator()
    a, ok, note = v.validate(Action(np.array([np.nan, 0, 0, 0, 0, 0, 0])))
    assert not ok and "NaN" in note and v.rejections == 1


def test_validator_clamps_oversized_action():
    v = ActionValidator()
    a, ok, note = v.validate(Action(np.array([9.0, -9.0, 0, 0, 0, 0, 0])))
    assert ok and "clamped" in note
    assert np.all(np.abs(a.vector[:6]) <= 1.0)


def test_validator_binarizes_gripper():
    v = ActionValidator()
    assert v.validate(Action(np.array([0, 0, 0, 0, 0, 0, 0.7])))[0].gripper == 1.0
    v.reset()
    assert v.validate(Action(np.array([0, 0, 0, 0, 0, 0, 0.2])))[0].gripper == 0.0


def test_validator_rejects_gripper_chatter():
    v = ActionValidator(SafetyLimits(gripper_chatter_max_flips=3))
    last_ok = True
    for i in range(8):
        _, last_ok, _ = v.validate(Action(np.array([0, 0, 0, 0, 0, 0, float(i % 2)])))
        if not last_ok:
            break
    assert not last_ok, "alternating gripper commands must eventually be rejected"


TABLETOP_LIMITS = SafetyLimits(
    workspace_min=(-0.30, -0.30, 0.0), workspace_max=(0.30, 0.30, 0.35)
)


def test_validator_keeps_ee_inside_workspace():
    v = ActionValidator(TABLETOP_LIMITS)
    ee = np.array([0.0, 0.0, 0.34], dtype=np.float32)  # ceiling is 0.35
    a, ok, note = v.validate(Action(np.array([0, 0, 1.0, 0, 0, 0, 0])), ee)
    assert ok and "workspace" in note
    assert ee[2] + a.vector[2] * 0.035 <= 0.35 + 1e-6


def test_workspace_check_is_opt_in():
    """No declared bounds -> no check. Guessing a box is worse than no box."""
    v = ActionValidator()
    ee = np.array([0.0, 0.0, 99.0], dtype=np.float32)
    a, ok, note = v.validate(Action(np.array([0, 0, 1.0, 0, 0, 0, 0])), ee)
    assert ok and "workspace" not in note
    assert a.vector[2] == 1.0


def test_workspace_clip_never_exceeds_magnitude_ceiling():
    """Regression: LIBERO's EE sits at z=0.91, far outside TabletopSim's 0.35 m
    ceiling. Applying that box produced dz = (0.35-0.91)/0.035 = -16 every
    step - a 16x out-of-range command emitted by the *safety* layer."""
    v = ActionValidator(TABLETOP_LIMITS)
    ee = np.array([0.0, 0.0, 0.91], dtype=np.float32)  # way above the ceiling
    a, ok, note = v.validate(Action(np.array([0, 0, 0, 0, 0, 0, 0])), ee)
    assert ok and "workspace" in note
    assert np.all(np.abs(a.vector[:6]) <= 1.0), f"validator emitted {a.vector}"


def test_limits_are_derived_from_the_env():
    from rvc.envs.tabletop import TabletopSim

    lim = SafetyLimits.for_env(TabletopSim())
    assert lim.workspace_max == (0.30, 0.30, 0.35)
    # An env that declares nothing must yield no bounds, not defaults.
    class Bare:
        pass

    assert SafetyLimits.for_env(Bare()).workspace_max is None


# --- environment ------------------------------------------------------------


def test_env_renders_a_real_image():
    env = TabletopSim()
    obs = env.reset()
    assert obs.image.shape == (256, 256, 3) and obs.image.dtype == np.uint8
    assert obs.image.std() > 5, "a blank frame means rendering is broken"


def test_detector_finds_the_block_and_the_box():
    obs = TabletopSim().reset()
    dets = {d.label: d for d in ColorDetector().detect(obs.image, ("red_block", "blue_box"))}
    assert set(dets) == {"red_block", "blue_box"}
    truth = obs.privileged["block"][:2]
    got = dets["red_block"].center_world
    assert abs(got[0] - truth[0]) < 0.02 and abs(got[1] - truth[1]) < 0.02


def test_occluder_actually_hides_the_block():
    env = TabletopSim(inject="target_lost")
    det = ColorDetector()
    hidden = []
    for _ in range(14):
        obs, *_ = env.step(Action.zeros())
        hidden.append(det.find(obs.image, "red_block") is None)
    assert any(hidden), "target_lost injection must make detection genuinely fail"
    assert not all(hidden), "the occluder must pass, otherwise recovery is impossible"


# --- end to end -------------------------------------------------------------


def _run(inject: str, max_recoveries: int = 3):
    env = TabletopSim(inject=inject, max_steps=220)
    agent = RobotAgent(
        env=env,
        policy=ScriptedMockPolicy(),
        config=AgentConfig(max_recoveries=max_recoveries, max_total_steps=220),
        collect_frames=False,
    )
    return agent, agent.run()


def test_clean_run_succeeds_without_recovery():
    agent, r = _run("none")
    assert r.success and r.final_state is AgentState.SUCCEEDED
    assert r.recoveries == 0 and r.failure is FailureKind.NONE


@pytest.mark.parametrize("fault", ["target_lost", "grasp_fail", "grasp_slip"])
def test_every_injected_fault_reaches_recover_and_still_succeeds(fault):
    agent, r = _run(fault)
    assert r.recoveries >= 1, f"{fault} never triggered RECOVER"
    assert any(t.to is AgentState.RECOVER for t in agent.trace.transitions)
    assert r.success, f"{fault} was not recovered from"


def test_recovery_budget_is_enforced():
    agent, r = _run("target_lost", max_recoveries=0)
    assert not r.success and r.final_state is AgentState.FAILED
    assert r.recoveries == 0


def test_state_machine_visits_every_state():
    agent, _ = _run("grasp_slip")
    visited = {t.to for t in agent.trace.transitions} | {AgentState.IDLE}
    for s in (AgentState.PERCEIVE, AgentState.PLAN, AgentState.EXECUTE,
              AgentState.VERIFY, AgentState.RECOVER):
        assert s in visited, f"{s.value} was never entered"


def test_every_logged_action_is_within_limits():
    agent, _ = _run("grasp_slip")
    assert agent.trace.records
    for rec in agent.trace.records:
        v = np.asarray(rec.action)
        assert v.shape == (7,) and np.all(np.isfinite(v))
        assert np.all(np.abs(v[:6]) <= 1.0 + 1e-6)
        assert v[6] in (0.0, 1.0)


# --- honesty ----------------------------------------------------------------


def test_mock_backend_is_always_flagged_degraded():
    res = resolve_policy("mock")
    assert res.degraded and "NOT from a vision-language-action model" in res.degraded_reason
    _, r = _run("none")
    assert r.degraded and r.degraded_reason


def test_no_degraded_flag_refuses_to_silently_fall_back():
    from rvc.policies.base import PolicyUnavailable

    with pytest.raises(PolicyUnavailable):
        resolve_policy("auto", allow_degraded=False)


def test_resolution_records_why_each_backend_was_skipped():
    res = resolve_policy("auto")
    names = dict(res.attempts)
    assert "openvla-local" in names and names["openvla-local"] != "OK"
    assert names.get("mock") == "OK"


# --- auto pairing -----------------------------------------------------------


def test_auto_env_avoids_the_impossible_mock_libero_pairing():
    """`auto` must not pick LIBERO just because it is installed: the mock
    cannot drive it, so the out-of-the-box demo would always fail."""
    from rvc.envs.registry import resolve_env

    res = resolve_env("auto", policy_kind="mock", max_steps=20)
    assert res.chosen == "tabletop"
    skipped = dict(res.attempts).get("libero", "")
    assert "mock" in skipped, "the skip must state its reason"
    res.env.close()


def test_explicit_env_request_is_still_honoured():
    """Asking for tabletop with a real VLA must not be silently overridden."""
    from rvc.envs.registry import resolve_env

    res = resolve_env("tabletop", policy_kind="openvla-remote", max_steps=20)
    assert res.chosen == "tabletop"
    res.env.close()
