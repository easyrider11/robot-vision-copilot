"""A dependency-free tabletop pick-and-place simulator.

WHY THIS EXISTS
---------------
Real LIBERO needs robosuite + MuJoCo + offscreen GL, and real OpenVLA needs
15 GB of weights and a CUDA GPU. On a machine that has neither, the honest move
is *not* to fake an OpenVLA rollout - it is to ship a small simulator that
genuinely runs, renders genuine images, and exercises the exact same
Observation -> Policy -> Action -> Env loop the real stack uses.

So this file is a real simulator with a real (if simple) contact model, driven
by the same 7-DoF action vector LIBERO/OpenVLA use. Everything downstream - the
agent state machine, the validators, the recovery logic, the FastAPI service -
is identical whether the env underneath is this or MuJoCo.

It is always reported as `degraded=True`. It is a teaching substitute for
LIBERO, not LIBERO.

WORLD
-----
    x: -0.30 .. +0.30  (left  -> right)
    y: -0.30 .. +0.30  (near  -> far)
    z:  0.00 .. +0.35  (table -> up)

Camera is a top-down orthographic view. Height is conveyed by a drop shadow
whose offset grows with z, plus slight scaling of the gripper - enough that you
can actually *see* the arm descend in the saved frames.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rvc.types import Action, Observation

# --- world constants --------------------------------------------------------
X_MIN, X_MAX = -0.30, 0.30
Y_MIN, Y_MAX = -0.30, 0.30
Z_MIN, Z_MAX = 0.00, 0.35

STEP_SCALE = 0.035  # metres per unit of normalized action
GRASP_XY_TOL = 0.045  # how close in xy the jaws must be to latch
GRASP_Z_TOL = 0.055  # how low the jaws must be to latch
PLACE_Z_TOL = 0.12  # release below this height counts as a controlled place

IMG_SIZE = 256

# --- palette ----------------------------------------------------------------
C_BG = (28, 30, 36)
C_TABLE = (196, 168, 128)
C_TABLE_EDGE = (150, 124, 90)
C_GRID = (182, 155, 118)
C_BLOCK = (214, 62, 58)
C_BLOCK_DK = (150, 38, 36)
C_BOX = (58, 118, 206)
C_BOX_IN = (36, 78, 142)
C_GRIP = (232, 236, 242)
C_GRIP_DK = (150, 158, 170)
C_SHADOW = (120, 100, 74)
C_OCCL = (58, 62, 72)


def _to_px(x: float, y: float) -> tuple[float, float]:
    """World (x, y) -> image pixel. +y is 'far', so it maps to a smaller row."""
    u = (x - X_MIN) / (X_MAX - X_MIN) * IMG_SIZE
    v = (Y_MAX - y) / (Y_MAX - Y_MIN) * IMG_SIZE
    return u, v


class TabletopSim:
    """Pick the red block, place it in the blue box."""

    name = "tabletop-sim"
    #: (min_xyz, max_xyz) in metres - consumed by SafetyLimits.for_env.
    workspace_bounds = ((X_MIN, Y_MIN, Z_MIN), (X_MAX, Y_MAX, Z_MAX))
    degraded = True
    degraded_reason = (
        "Built-in substitute for LIBERO: robosuite/MuJoCo not installed. "
        "Same 7-DoF action interface, simplified contact model."
    )

    #: Task presets mirroring the phrasing style of LIBERO task descriptions.
    TASKS = {
        "pick_place_block": {
            "instruction": "pick up the red block and place it in the blue box",
            "block": (-0.16, -0.12),
            "box": (0.15, 0.14),
        },
        "pick_place_block_far": {
            "instruction": "pick up the red block and put it inside the blue box",
            "block": (0.18, -0.18),
            "box": (-0.14, 0.16),
        },
    }

    def __init__(
        self,
        task_id: str = "pick_place_block",
        max_steps: int = 120,
        inject: str = "none",
        seed: int = 0,
    ) -> None:
        if task_id not in self.TASKS:
            raise ValueError(f"unknown task_id {task_id!r}; choose from {list(self.TASKS)}")
        self.task_id = task_id
        self.max_steps = max_steps
        self.inject = inject
        self.rng = np.random.default_rng(seed)

        cfg = self.TASKS[task_id]
        self._instruction: str = cfg["instruction"]
        self._block_start = np.array(cfg["block"], dtype=np.float32)
        self._box_xy = np.array(cfg["box"], dtype=np.float32)
        self.box_half = 0.055

        # fault-injection bookkeeping
        self._slip_armed = inject == "grasp_slip"
        self._steps_since_grasp = 0
        self._grasp_attempts = 0
        # Window chosen to land during approach/descend, i.e. while the agent is
        # NOT yet holding anything - a held object is legitimately hidden by the
        # gripper, so occluding it then would prove nothing.
        self._occlusion_window: tuple[int, int] = (5, 12) if inject == "target_lost" else (-1, -1)

        self.reset()

    # -- core loop -----------------------------------------------------------

    def reset(self) -> Observation:
        self.t = 0
        self.ee = np.array([0.0, -0.02, 0.26], dtype=np.float32)
        self.grip_closed = False
        self.holding = False
        self.block = self._block_start.copy()
        self.block_z = 0.0
        self.done = False
        self.success = False
        self._steps_since_grasp = 0
        self._grasp_attempts = 0
        self._slip_armed = self.inject == "grasp_slip"
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        info: dict = {"event": None}
        if self.done:
            return self._observe(), 0.0, True, info

        self.t += 1

        # 1. integrate the end-effector delta
        delta = np.clip(action.delta_xyz, -1.0, 1.0) * STEP_SCALE
        self.ee = self.ee + delta.astype(np.float32)
        self.ee[0] = float(np.clip(self.ee[0], X_MIN, X_MAX))
        self.ee[1] = float(np.clip(self.ee[1], Y_MIN, Y_MAX))
        self.ee[2] = float(np.clip(self.ee[2], Z_MIN, Z_MAX))

        # 2. gripper transitions
        want_closed = action.gripper > 0.5
        if want_closed and not self.grip_closed:
            info["event"] = self._close_gripper()
        elif not want_closed and self.grip_closed:
            info["event"] = self._open_gripper()
        self.grip_closed = want_closed

        # 3. carried object follows the jaws
        if self.holding:
            self._steps_since_grasp += 1
            self.block = self.ee[:2].copy()
            self.block_z = max(0.0, float(self.ee[2]) - 0.012)
            # injected slip: the object works loose mid-transit
            if self._slip_armed and self._steps_since_grasp >= 6 and self.ee[2] > 0.10:
                self.holding = False
                self._slip_armed = False
                self.block_z = 0.0
                info["event"] = "grasp_slip"

        # 4. termination
        reward = 0.0
        if self._block_in_box() and not self.holding:
            self.success = True
            self.done = True
            reward = 1.0
            info["event"] = "success"
        elif self.t >= self.max_steps:
            self.done = True
            info["event"] = "timeout"

        return self._observe(), reward, self.done, info

    def _close_gripper(self) -> str | None:
        if self.holding:
            return None
        self._grasp_attempts += 1
        near_xy = float(np.linalg.norm(self.ee[:2] - self.block)) < GRASP_XY_TOL
        low_enough = self.ee[2] < GRASP_Z_TOL
        # injected failure: the very first attempt never latches
        if self.inject == "grasp_fail" and self._grasp_attempts == 1:
            return "grasp_failed"
        if near_xy and low_enough:
            self.holding = True
            self._steps_since_grasp = 0
            return "grasped"
        return "grasp_failed"

    def _open_gripper(self) -> str | None:
        if not self.holding:
            return None
        self.holding = False
        controlled = self.ee[2] < PLACE_Z_TOL
        self.block_z = 0.0
        return "placed" if controlled else "dropped"

    def _block_in_box(self) -> bool:
        return (
            abs(float(self.block[0]) - float(self._box_xy[0])) < self.box_half
            and abs(float(self.block[1]) - float(self._box_xy[1])) < self.box_half
            and self.block_z < 0.02
        )

    # -- dataset generation (used by `make yolo-data`) -----------------------

    def randomize_layout(self, rng: np.random.Generator) -> None:
        """Scatter block, box and gripper for synthetic-data generation.

        Keeps a margin from the table edge and a minimum block-box separation
        so every frame is a plausible task state. Ground-truth boxes for the
        labels come from `ground_truth_boxes()`, which mirrors `render()`.
        """
        m = 0.05
        for _ in range(100):
            block = rng.uniform([X_MIN + m, Y_MIN + m], [X_MAX - m, Y_MAX - m])
            box = rng.uniform([X_MIN + m + 0.02, Y_MIN + m + 0.02],
                              [X_MAX - m - 0.02, Y_MAX - m - 0.02])
            if np.linalg.norm(block - box) > 0.14:
                break
        self.block = block.astype(np.float32)
        self._box_xy = box.astype(np.float32)
        self.block_z = float(rng.choice([0.0, 0.0, 0.0, rng.uniform(0.02, 0.2)]))
        self.ee = np.array([
            rng.uniform(X_MIN + m, X_MAX - m), rng.uniform(Y_MIN + m, Y_MAX - m),
            rng.uniform(0.02, Z_MAX),
        ], dtype=np.float32)
        self.grip_closed = bool(rng.random() < 0.4)
        self.holding = False

    def ground_truth_boxes(self) -> dict[str, tuple[float, float, float, float]]:
        """Pixel bboxes (x0, y0, x1, y1) of the rendered objects - same math as render()."""
        out: dict[str, tuple[float, float, float, float]] = {}
        bx, by = _to_px(float(self._box_xy[0]), float(self._box_xy[1]))
        bh = self.box_half / (X_MAX - X_MIN) * IMG_SIZE
        out["blue_box"] = (bx - bh, by - bh, bx + bh, by + bh)
        if not self._occluded():
            sx, sy = _to_px(float(self.block[0]), float(self.block[1]))
            r = 13 - float(self.block_z) * 8
            out["red_block"] = (sx - r, sy - r, sx + r, sy + r)
        return out

    # -- runtime fault injection (used by `make play`) -----------------------

    def arm_fault(self, kind: str) -> str:
        """Arm a fault mid-episode. Returns a human-readable description of
        what will happen, so interactive tools can echo it honestly."""
        if kind == "grasp_slip":
            self.inject = "grasp_slip"
            self._slip_armed = True
            self._steps_since_grasp = 0
            return "滑落已布防：下次抓取后搬运 6 步且 z>0.10 时物体脱手"
        if kind == "grasp_fail":
            self.inject = "grasp_fail"
            self._grasp_attempts = 0
            return "抓取失败已布防：下一次闭合夹爪不会咬合"
        if kind == "target_lost":
            self._occlusion_window = (self.t + 2, self.t + 10)
            return f"遮挡板将在 step {self.t + 2}..{self.t + 10} 滑过工作区"
        raise ValueError(f"unknown fault {kind!r}; choose grasp_slip/grasp_fail/target_lost")

    # -- observation ---------------------------------------------------------

    @property
    def instruction(self) -> str:
        return self._instruction

    def _occluded(self) -> bool:
        lo, hi = self._occlusion_window
        return lo <= self.t < hi

    def _observe(self) -> Observation:
        img = self.render()
        return Observation(
            image=img,
            instruction=self._instruction,
            step=self.t,
            privileged={
                "ee": self.ee.astype(float).tolist(),
                "block": [float(self.block[0]), float(self.block[1]), float(self.block_z)],
                "box": [float(self._box_xy[0]), float(self._box_xy[1])],
                "box_half": self.box_half,
                "holding": bool(self.holding),
                "grip_closed": bool(self.grip_closed),
                "occluded": self._occluded(),
                "success": bool(self.success),
                "grasp_xy_tol": GRASP_XY_TOL,
                "grasp_z_tol": GRASP_Z_TOL,
            },
        )

    def render(self) -> np.ndarray:
        im = Image.new("RGB", (IMG_SIZE, IMG_SIZE), C_BG)
        d = ImageDraw.Draw(im)

        # table
        d.rectangle([8, 8, IMG_SIZE - 9, IMG_SIZE - 9], fill=C_TABLE, outline=C_TABLE_EDGE, width=3)
        for g in range(1, 6):
            p = 8 + g * (IMG_SIZE - 17) / 6
            d.line([(p, 9), (p, IMG_SIZE - 10)], fill=C_GRID, width=1)
            d.line([(9, p), (IMG_SIZE - 10, p)], fill=C_GRID, width=1)

        # target box (drawn as an open container)
        bx, by = _to_px(float(self._box_xy[0]), float(self._box_xy[1]))
        bh = self.box_half / (X_MAX - X_MIN) * IMG_SIZE
        d.rectangle([bx - bh, by - bh, bx + bh, by + bh], fill=C_BOX_IN, outline=C_BOX, width=5)

        occluded = self._occluded()

        gx, gy = _to_px(float(self.ee[0]), float(self.ee[1]))
        z = float(self.ee[2])
        goff = z * 90
        scale = 1.0 - z * 0.55

        # --- shadows first: they lie on the table, under everything else. -----
        # (Drawing them after the block would make the arm's own shadow occlude
        # the target and trigger spurious TARGET_LOST recoveries.)
        d.ellipse(
            [gx - 17 + goff, gy - 17 + goff, gx + 17 + goff, gy + 17 + goff], fill=C_SHADOW
        )
        hz = float(self.block_z)
        sx, sy = _to_px(float(self.block[0]), float(self.block[1]))
        r = 13 - hz * 8
        if not occluded and hz > 0.01:
            boff = hz * 90
            d.rectangle([sx - r + boff, sy - r + boff, sx + r + boff, sy + r + boff], fill=C_SHADOW)

        # --- the block sits on the table, above the shadows ------------------
        if not occluded:
            d.rectangle([sx - r, sy - r, sx + r, sy + r], fill=C_BLOCK, outline=C_BLOCK_DK, width=2)

        # --- gripper on top: two jaws, spacing = open/closed, size = height ---
        jaw_w, jaw_h = 7 * scale, 22 * scale
        gap = (9 if self.grip_closed else 19) * scale

        d.line([(gx, gy - 34 * scale), (gx, gy - 8 * scale)],
               fill=C_GRIP_DK, width=max(2, int(4 * scale)))
        for sgn in (-1, 1):
            cx = gx + sgn * gap
            d.rectangle(
                [cx - jaw_w / 2, gy - jaw_h / 2, cx + jaw_w / 2, gy + jaw_h / 2],
                fill=C_GRIP,
                outline=C_GRIP_DK,
                width=2,
            )

        # occluder: an opaque panel slides over the workspace, so "target lost"
        # is a real perception failure and not a flag we flipped in code.
        if occluded:
            d.rectangle([12, 96, IMG_SIZE - 13, 172], fill=C_OCCL)
            d.text((22, 126), "OCCLUDER", fill=(200, 205, 215))

        return np.asarray(im, dtype=np.uint8)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
