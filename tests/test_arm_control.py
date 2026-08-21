"""The resolved-rate math must be correct before it touches Gazebo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "rvc_agent"))
from rvc_agent.arm_control import (
    MAX_DQ,
    orientation_error,
    quat_to_matrix,
    resolved_rate_step,
)


def test_identity_quaternion_and_zero_error():
    r = quat_to_matrix(0, 0, 0, 1)
    assert np.allclose(r, np.eye(3))
    assert np.allclose(orientation_error(r, r), 0)


def test_orientation_error_direction():
    # current rotated +10deg about world z; correction must be -z
    th = np.radians(10)
    rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    err = orientation_error(np.eye(3), rz)
    assert err[2] < 0 and abs(err[0]) < 1e-9 and abs(err[1]) < 1e-9


def test_resolved_rate_tracks_pure_translation():
    # toy 2-joint planar arm: q -> EE (l1 c1 + l2 c12, l1 s1 + l2 s12)
    l1 = l2 = 0.5

    def fk(q):
        return np.array([
            l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
            l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]),
        ])

    def jac(q):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        j = np.zeros((6, 2))
        j[0] = [-l1 * s1 - l2 * s12, -l2 * s12]
        j[1] = [l1 * c1 + l2 * c12, l2 * c12]
        return j

    q = np.array([0.6, 0.8])
    target = fk(q) + np.array([-0.15, 0.10])
    eye = np.eye(3)
    for _ in range(400):
        err = target - fk(q)
        if np.linalg.norm(err) < 1e-3:
            break
        v = np.clip(err * 4.0, -0.2, 0.2)
        dq = resolved_rate_step(jac(q), np.array([v[0], v[1], 0.0]), eye, eye, dt=0.05)
        q = q + dq
    assert np.linalg.norm(target - fk(q)) < 1e-3, f"did not converge: {fk(q)} vs {target}"


def test_step_is_rate_limited_near_singularity():
    j = np.zeros((6, 7))
    j[0, 0] = 1e-6  # nearly singular
    dq = resolved_rate_step(j, np.array([1.0, 0, 0]), np.eye(3), np.eye(3), dt=0.1)
    assert np.max(np.abs(dq)) <= MAX_DQ + 1e-12
    assert np.all(np.isfinite(dq))
