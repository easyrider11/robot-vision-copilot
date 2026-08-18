"""Unit tests for the pick-and-place sequencer, driven by a toy kinematic sim.

The toy sim mirrors the Gazebo setup: a marker that moves with the commanded
velocity (through the calibrated axis map), a z that integrates vz, a block
that becomes occluded when the gripper is over it, rides along when attached,
and lands where it is dropped. If the sequencer completes here, the only
things left to go wrong in Gazebo are physics and plumbing - which is the
point of the split.
"""

from __future__ import annotations

import numpy as np

from rvc.agent.pickplace import (
    PLACE_TOL_PX,
    Z_GRASP,
    Z_PLACE,
    PickInput,
    PickPlaceSequencer,
)
from rvc.perception.detector import ColorDetector, Detection
from rvc.policies.visual_servo import VisualServoPolicy

PX_PER_TICK = 8.0  # marker pixels per unit action per tick
Z_PER_TICK = 0.02  # metres per unit vz per tick
OCCLUDE_PX = 14.0  # marker closer than this hides the block from the camera
PAD_PX = (162.0, 95.0)
BLOCK_PX = (89.0, 156.0)


def _det(label: str, u: float, v: float) -> Detection:
    return Detection(
        label=label, confidence=0.9,
        bbox_px=(int(u) - 6, int(v) - 6, int(u) + 6, int(v) + 6),
        center_px=(u, v), center_world=(0.0, 0.0), area_px=144,
    )


class ToySim:
    """Just enough kinematics + occlusion to exercise every phase."""

    def __init__(self, fail_first_grasp: bool = False) -> None:
        self.marker = np.array([128.0, 134.0])
        self.z = 0.95
        self.block = np.array(BLOCK_PX)
        self.attached = False
        self.spawn_attached = True  # DetachableJoint starts attached in gz
        self.fail_first_grasp = fail_first_grasp
        self.attach_count = 0

    def apply(self, cmd) -> None:
        # inverse of the calibrated axis map: du = +ax, dv = -ay
        d = np.array([cmd.vx * PX_PER_TICK, -cmd.vy * PX_PER_TICK])
        self.marker += d
        if self.spawn_attached:  # rigid spawn attachment drags the block too
            self.block += d
        self.z += cmd.vz * Z_PER_TICK
        if cmd.gripper_event == "attach":
            self.attach_count += 1
            near = float(np.hypot(*(self.marker - self.block))) < OCCLUDE_PX
            low = self.z < Z_GRASP + 0.02
            # fail_first_grasp models a persistent fault (dirty suction cup):
            # every attach during the FIRST grasp attempt fails, longer than
            # the confirm timeout, so the sequencer must go through RECOVER.
            broken = self.fail_first_grasp and self.attach_count <= 25
            self.attached = near and low and not broken
        elif cmd.gripper_event == "detach":
            if self.attached:
                self.block = self.marker.copy()  # dropped where we hover
            self.attached = False
            self.spawn_attached = False
        if self.attached:
            self.block = self.marker.copy()  # rides along under the gripper

    def observe(self) -> PickInput:
        hidden = self.attached or float(np.hypot(*(self.marker - self.block))) < OCCLUDE_PX
        joint = "attached" if (self.attached or self.spawn_attached) else "detached"
        return PickInput(
            target=None if hidden else _det("red_block", *self.block),
            pad=_det("blue_box", *PAD_PX),
            marker=_det("gripper_marker", *self.marker),
            z=self.z,
            joint_state=joint,
        )


def _run(sim: ToySim, seq: PickPlaceSequencer, max_ticks: int = 600):
    phases = []
    for _ in range(max_ticks):
        cmd = seq.step(sim.observe())
        sim.apply(cmd)
        if not phases or phases[-1] != cmd.phase:
            phases.append(cmd.phase)
        if cmd.done or cmd.failed:
            return cmd, phases
    raise AssertionError(f"never terminated; phase={seq.phase} z={sim.z:.3f} "
                         f"marker={sim.marker} block={sim.block}")


def _seq(**kw) -> PickPlaceSequencer:
    return PickPlaceSequencer(servo=VisualServoPolicy(ColorDetector()), **kw)


# --- happy path --------------------------------------------------------------


def test_full_pick_and_place_succeeds():
    sim, seq = ToySim(), _seq()
    cmd, phases = _run(sim, seq)
    assert cmd.done and not cmd.failed
    assert seq.recoveries == 0
    # the block must physically end up on the pad, and be SEEN there
    assert float(np.hypot(*(sim.block - np.array(PAD_PX)))) < PLACE_TOL_PX
    assert not sim.attached
    for expected in ("INIT", "APPROACH", "DESCEND", "GRASP", "LIFT",
                     "TRANSPORT", "LOWER", "RELEASE", "RETREAT", "VERIFY", "DONE"):
        assert expected in phases, f"phase {expected} never entered: {phases}"


def test_spawn_attachment_is_released_first():
    sim, seq = ToySim(), _seq()
    cmd = seq.step(sim.observe())
    assert cmd.gripper_event == "detach", "INIT must let go of the spawn attachment"


def test_heights_are_respected():
    sim, seq = ToySim(), _seq()
    grasp_z = place_z = None
    for _ in range(600):
        cmd = seq.step(sim.observe())
        if cmd.gripper_event == "attach":
            grasp_z = sim.z
        if cmd.gripper_event == "detach" and cmd.phase == "RELEASE":
            place_z = sim.z
        sim.apply(cmd)
        if cmd.done or cmd.failed:
            break
    assert grasp_z is not None and abs(grasp_z - Z_GRASP) < 0.03
    assert place_z is not None and abs(place_z - Z_PLACE) < 0.03


# --- failure and recovery ----------------------------------------------------


def test_failed_grasp_recovers_via_confirm_timeout():
    sim, seq = ToySim(fail_first_grasp=True), _seq()
    cmd, phases = _run(sim, seq)
    assert cmd.done, f"should recover and finish, got failure={cmd.failure}"
    assert seq.recoveries == 1
    assert "RECOVER" in phases
    assert sim.attach_count >= 2, "must have re-attempted the grasp"


def test_recovery_budget_exhaustion_fails_closed():
    class NeverGrasp(ToySim):
        def apply(self, cmd):
            super().apply(cmd)
            self.attached = False  # gripper is broken

    sim, seq = NeverGrasp(), _seq(max_recoveries=2)
    cmd, phases = _run(sim, seq)
    assert cmd.failed and not cmd.done
    assert "grasp_failed" in cmd.failure
    assert seq.recoveries == 2
    assert not sim.attached, "must end detached (safe)"


def test_verify_rejects_a_block_dropped_off_the_pad():
    class SlipperyDrop(ToySim):
        def apply(self, cmd):
            dropped = cmd.gripper_event == "detach" and self.attached
            super().apply(cmd)
            if dropped:
                self.block = self.marker + np.array([40.0, 40.0])  # bounced away

    sim, seq = SlipperyDrop(), _seq(max_recoveries=0)
    cmd, phases = _run(sim, seq)
    assert cmd.failed
    assert "place_missed" in cmd.failure


def test_lost_detach_messages_do_not_wave_through():
    """Regression for the first live run: the spawn attachment survived
    because fire-and-forget detach messages were lost in transport discovery.
    INIT must keep commanding until the joint confirms."""
    class LossyStart(ToySim):
        def __init__(self):
            super().__init__()
            self.drops = 4  # first N detach messages vanish

        def apply(self, cmd):
            if cmd.gripper_event == "detach" and self.drops > 0:
                self.drops -= 1
                cmd = type(cmd)(**{**cmd.__dict__, "gripper_event": None})
            super().apply(cmd)

    sim, seq = LossyStart(), _seq()
    cmd, phases = _run(sim, seq)
    assert cmd.done, f"failure={cmd.failure}"
    assert not sim.spawn_attached, "spawn attachment must actually be released"


def test_carried_block_peeking_out_is_not_a_failure():
    """A held block may show a sliver near the marker (grasp offset); only a
    block visible FAR from the marker means it was left behind."""
    class PeekSim(ToySim):
        def observe(self):
            obs = super().observe()
            if self.attached:  # sliver visible right next to the marker
                obs.target = _det("red_block", self.marker[0] + 10, self.marker[1] + 8)
            return obs

    sim, seq = PeekSim(), _seq()
    cmd, phases = _run(sim, seq)
    assert cmd.done, f"peeking carried block was misjudged: {cmd.failure}"
    assert seq.recoveries == 0


def test_no_odometry_means_no_z_motion():
    seq = _seq()
    seq.phase = "DESCEND"
    cmd = seq.step(PickInput(target=None, pad=None, marker=None, z=None))
    assert cmd.vz == 0.0, "without odometry the sequencer must not guess heights"
