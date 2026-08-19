"""Pixel-space visual servoing - the Stage 3 teaching policy.

WHAT THIS IS
------------
A proportional controller that closes the loop from CAMERA PIXELS ALONE:

    error_px = detect(target).center - detect(gripper_marker).center
    action   = clip(Kp * error_px)

Unlike `ScriptedMockPolicy` (which reads the simulator's privileged object
coordinates and therefore cannot run outside TabletopSim), this policy needs
nothing but an image in which both the target and the gripper marker are
visible. That makes it the first policy in this repo that works on *any*
environment with a compatible camera - including the Gazebo world, where the
floating gripper carries a green top-face marker for exactly this purpose.

WHAT THIS IS NOT
----------------
Still not a VLA. It cannot read the instruction beyond picking which color to
chase, it has no notion of grasping strategy, and it is stamped degraded just
like the mock. It exists to teach closed-loop perception -> action, and to
give the Gazebo demo honest, observable motion before a real OpenVLA backend
is attached.

AXIS MAPPING
------------
The mapping from image axes (u right, v down) to world/robot axes depends on
the camera mounting and is the classic source of "it runs away from the
target" bugs. It is therefore an explicit constructor argument instead of a
hidden convention:

    axis_map = ((au, av, bu, bv))  meaning
        action_x = au * eu + av * ev
        action_y = bu * eu + bv * ev
    with (eu, ev) = normalized pixel error in [-1, 1].

Default is the Stage 3 overhead camera in worlds/tabletop.sdf, CALIBRATED
against actual Gazebo renders on 2026-08-14: known world positions
(-0.16,-0.12) and (0.15,0.14) landed at pixels (89,156) and (162,95), giving

    u = 128 + ~235 * world_x        v = 128 - ~235 * world_y

so image-u tracks +world_x and image-v tracks -world_y, hence
axis_map = (1, 0, 0, -1). (The pre-run guess was (0,-1,1,0) - wrong, exactly
the kind of sign bug this parameter exists to make visible.) If the gripper
drives away from the target after a camera change, recalibrate HERE, not in
the node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvc.perception.detector import Detection
from rvc.types import Action, Gripper, Observation


@dataclass
class ServoStatus:
    """What the last predict() saw - published by the ROS node for debugging."""

    target_found: bool = False
    marker_found: bool = False
    error_px: tuple[float, float] = (0.0, 0.0)
    settled: bool = False


class VisualServoPolicy:
    """P-controller on pixel error between a target and a gripper marker."""

    name = "visual-servo"
    degraded = True
    degraded_reason = (
        "Pixel-space proportional servo, NOT a vision-language-action model. "
        "It only chases a color blob; language is ignored except for target choice."
    )

    def __init__(
        self,
        detector,
        target_label: str = "red_block",
        marker_label: str = "gripper_marker",
        kp: float = 3.0,
        deadband_px: float = 4.0,
        image_size: int = 256,
        axis_map: tuple[float, float, float, float] = (1.0, 0.0, 0.0, -1.0),
    ) -> None:
        self.detector = detector
        self.target_label = target_label
        self.marker_label = marker_label
        self.kp = kp
        self.deadband_px = deadband_px
        self.image_size = image_size
        self.axis_map = axis_map
        self.status = ServoStatus()

    def describe(self) -> str:
        return (
            f"{self.name} (kp={self.kp}, deadband={self.deadband_px}px, "
            f"{self.marker_label} -> {self.target_label})"
        )

    # -- core ----------------------------------------------------------------

    def compute(self, target: Detection | None, marker: Detection | None) -> Action:
        """Pure control law - separated from detection so it can be unit-tested."""
        self.status = ServoStatus(
            target_found=target is not None, marker_found=marker is not None
        )
        if target is None or marker is None:
            # Missing either endpoint of the error vector -> the only safe
            # command is "hold still". The agent's PERCEIVE state is what
            # decides whether this is a TARGET_LOST failure.
            return Action.hold(Gripper.OPEN)

        eu = target.center_px[0] - marker.center_px[0]
        ev = target.center_px[1] - marker.center_px[1]
        self.status.error_px = (round(eu, 1), round(ev, 1))

        if abs(eu) < self.deadband_px and abs(ev) < self.deadband_px:
            self.status.settled = True
            return Action.hold(Gripper.OPEN)

        # normalize to [-1, 1] and apply the camera-to-robot axis map
        nu = eu / (self.image_size / 2)
        nv = ev / (self.image_size / 2)
        au, av, bu, bv = self.axis_map
        ax = float(np.clip(self.kp * (au * nu + av * nv), -1.0, 1.0))
        ay = float(np.clip(self.kp * (bu * nu + bv * nv), -1.0, 1.0))

        vec = np.zeros(7, dtype=np.float32)
        vec[0], vec[1] = ax, ay
        vec[6] = Gripper.OPEN
        return Action(vec)

    # -- Policy protocol -----------------------------------------------------

    def predict(self, obs: Observation) -> Action:
        dets = self.detector.detect(obs.image, (self.target_label, self.marker_label))
        by_label = {d.label: d for d in dets}
        return self.compute(by_label.get(self.target_label), by_label.get(self.marker_label))
