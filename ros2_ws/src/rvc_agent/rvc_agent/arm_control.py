"""Resolved-rate Cartesian control for the Panda - deterministic, in-process.

Replaces MoveIt Servo in the Panda pipeline. Servo's twist mode produced
non-deterministic joint solutions in this container (same -z command: descend
one session, climb the next) and its pose mode drifted even against a frozen
target; docs/09 records the experiments. This module is the same math Servo
promises - damped-least-squares resolved rate - in ~40 lines we can unit-test:

    dq = J^T (J J^T + lambda^2 I)^-1 [v_lin; w_orr]

- J comes from moveit_py's RobotState (KDL under the hood, deterministic)
- v_lin is the commanded world velocity (base frame == world axes here)
- w_corr is a small-angle orientation correction that keeps the suction face
  pointing down (the fixed spawn orientation), so orientation cannot drift
  the way a pure position servo would let it

The output is absolute joint positions for the (already proven) ros2_control
JointGroupPositionController. No open-loop integration against a black box:
q_cmd is re-seeded from the measured joint state every tick.
"""

from __future__ import annotations

import numpy as np

DAMPING = 0.05  # DLS lambda: rank-safe near singularities
MAX_DQ = 0.08  # rad per control step - hard joint-space rate limit
ORI_GAIN = 1.0  # rad/s per rad of orientation error


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def orientation_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector taking r_current to r_target (world frame)."""
    r_err = r_target @ r_current.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])


def resolved_rate_step(
    jacobian: np.ndarray,
    v_lin: np.ndarray,
    r_target: np.ndarray,
    r_current: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One damped-least-squares step. Returns dq (len = jacobian columns)."""
    twist = np.concatenate([v_lin, ORI_GAIN * orientation_error(r_target, r_current)])
    jjt = jacobian @ jacobian.T + (DAMPING ** 2) * np.eye(6)
    dq = jacobian.T @ np.linalg.solve(jjt, twist) * dt
    n = float(np.max(np.abs(dq)))
    if n > MAX_DQ:
        dq = dq * (MAX_DQ / n)
    return dq
