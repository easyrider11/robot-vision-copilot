"""Unit tests for the Stage 3 visual-servo policy.

The ROS 2 node itself cannot run on this machine, but the control law and the
detector interaction are pure Python - so the parts most likely to be wrong
(sign conventions, deadband, missing-detection behavior) are pinned here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rvc.perception.detector import ColorDetector, Detection
from rvc.policies.visual_servo import VisualServoPolicy
from rvc.types import Observation


def _det(label: str, cx: float, cy: float) -> Detection:
    return Detection(
        label=label, confidence=0.9,
        bbox_px=(int(cx) - 5, int(cy) - 5, int(cx) + 5, int(cy) + 5),
        center_px=(cx, cy), center_world=(0.0, 0.0), area_px=100,
    )


def _policy(**kw) -> VisualServoPolicy:
    return VisualServoPolicy(ColorDetector(), **kw)


# --- control law -------------------------------------------------------------


def test_missing_target_or_marker_means_hold_still():
    p = _policy()
    assert np.all(p.compute(None, _det("gripper_marker", 128, 128)).vector == 0)
    assert np.all(p.compute(_det("red_block", 40, 40), None).vector == 0)
    assert not p.status.settled


def test_inside_deadband_is_settled_and_still():
    p = _policy(deadband_px=4.0)
    a = p.compute(_det("red_block", 130, 129), _det("gripper_marker", 128, 128))
    assert np.all(a.vector == 0) and p.status.settled


def test_error_direction_follows_axis_map():
    # Default map (calibrated on Gazebo renders 2026-08-14):
    #   action_x = +eu, action_y = -ev.
    p = _policy(kp=1.0)
    # Target 64px to the RIGHT of the marker (eu=+64, ev=0) -> +x motion only.
    a = p.compute(_det("red_block", 192, 128), _det("gripper_marker", 128, 128))
    assert a.vector[0] > 0.0 and a.vector[1] == 0.0
    # Target 64px BELOW the marker (ev=+64) -> -y motion only (image v is -world y).
    a = p.compute(_det("red_block", 128, 192), _det("gripper_marker", 128, 128))
    assert a.vector[0] == 0.0 and a.vector[1] < 0.0


def test_output_is_clipped_and_z_rpy_untouched():
    p = _policy(kp=100.0)  # absurd gain must still clip
    a = p.compute(_det("red_block", 255, 0), _det("gripper_marker", 0, 255))
    assert np.all(np.abs(a.vector[:2]) <= 1.0)
    assert np.all(a.vector[2:6] == 0.0) and a.vector[6] == 0.0


def test_settles_as_error_shrinks():
    p = _policy(kp=2.0, deadband_px=4.0)
    marker = np.array([60.0, 60.0])
    target = _det("red_block", 128, 128)
    for _ in range(200):
        a = p.compute(target, _det("gripper_marker", *marker))
        if p.status.settled:
            break
        # inverse of the calibrated map: du = +ax, dv = -ay (10 px per unit action)
        marker[0] += a.vector[0] * 10
        marker[1] += -a.vector[1] * 10
    assert p.status.settled, f"never settled, marker ended at {marker}"


# --- through real pixels -----------------------------------------------------


def _frame_with(blocks: dict[str, tuple[int, int]]) -> np.ndarray:
    """Render solid color patches the ColorDetector genuinely has to find."""
    im = Image.new("RGB", (256, 256), (90, 90, 95))
    d = ImageDraw.Draw(im)
    colors = {"red_block": (214, 40, 36), "gripper_marker": (30, 220, 40)}
    for label, (cx, cy) in blocks.items():
        d.rectangle([cx - 8, cy - 8, cx + 8, cy + 8], fill=colors[label])
    return np.asarray(im, dtype=np.uint8)


def test_predict_closes_the_loop_from_pixels_alone():
    p = _policy(kp=2.0)
    img = _frame_with({"red_block": (200, 128), "gripper_marker": (60, 128)})
    a = p.predict(Observation(image=img, instruction="move over the red block"))
    assert a.vector[0] > 0.0, "target right of marker must command +x"
    assert p.status.target_found and p.status.marker_found


def test_predict_holds_still_when_marker_absent():
    p = _policy()
    img = _frame_with({"red_block": (200, 128)})
    a = p.predict(Observation(image=img, instruction="x"))
    assert np.all(a.vector == 0) and not p.status.marker_found


def test_visual_servo_is_flagged_degraded():
    p = _policy()
    assert p.degraded and "NOT a vision-language-action model" in p.degraded_reason
