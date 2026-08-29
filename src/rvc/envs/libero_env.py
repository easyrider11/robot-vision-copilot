"""Real LIBERO wrapper.

VERIFIED WORKING on macOS 15.5 / Apple M3 / arm64, 2026-08-13. Measured on
`libero_spatial` task 0 with two 256x256 offscreen cameras (MuJoCo
auto-selects `MUJOCO_GL=cgl`):

    env construction   2.3 s
    reset              0.55 s
    stepping           24 ms/step steady state
    220-step rollout   ~8 s of simulation

So the simulator is NOT the bottleneck - a real OpenVLA forward pass at
150-400 ms/step dominates by an order of magnitude.

All the version pinning and import glue this needs lives in
`rvc.envs.libero_bootstrap` - read that file before debugging an import error
here, it documents six separate incompatibilities and their symptoms.

Install:  make setup-libero
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

from rvc.envs.libero_bootstrap import bootstrap, torch_load_compat
from rvc.types import Action, Observation


class LiberoUnavailable(RuntimeError):
    """Raised with a precise reason so the runner can report honest degradation."""


def probe() -> tuple[bool, str]:
    """Return (available, reason). Never raises, never renders."""
    ok, why = bootstrap()
    if not ok:
        return False, why
    try:
        import libero  # noqa: F401
        import robosuite  # noqa: F401
        from libero.libero import benchmark  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        return False, f"LIBERO/robosuite not importable: {type(exc).__name__}: {exc}"

    # robosuite 1.4.x reads MjData.qM, removed in MuJoCo 3.3. Catching this
    # here turns a confusing mid-rollout AttributeError into a clear message.
    try:
        import mujoco

        ver = tuple(int(p) for p in mujoco.__version__.split(".")[:2])
        if ver >= (3, 3):
            return False, (
                f"mujoco {mujoco.__version__} removed MjData.qM, which robosuite 1.4.x "
                'still uses. Run: uv pip install "mujoco>=3.2,<3.3"'
            )
    except Exception as exc:
        return False, f"mujoco not importable: {type(exc).__name__}: {exc}"

    return True, ""


class LiberoEnv:
    """Thin adapter from LIBERO's OffScreenRenderEnv to the rvc Env protocol."""

    name = "libero"
    degraded = False
    degraded_reason = ""
    inject = "none"  # LIBERO faults come from the physics, not from us

    #: Deliberately None. LIBERO's reachable workspace depends on the Panda's
    #: joint limits and the scene, and I have not measured it - the OSC
    #: controller and the robot's own limits are the real boundary there.
    #: Inventing a box here previously produced dz = -16 commands every step.
    #: Replace with a measured (min_xyz, max_xyz) if you want this guard back.
    workspace_bounds = None

    SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        task_index: int = 0,
        init_state_index: int = 0,
        resolution: int = 256,
        max_steps: int = 220,
        seed: int = 0,
        instruction_override: str = "",
    ) -> None:
        ok, reason = probe()
        if not ok:
            raise LiberoUnavailable(reason)

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        if task_suite not in suites:
            raise LiberoUnavailable(
                f"unknown task suite {task_suite!r}; available: {sorted(suites)}"
            )
        suite = suites[task_suite]()
        if not 0 <= task_index < suite.n_tasks:
            raise LiberoUnavailable(
                f"task_index {task_index} out of range for {task_suite} "
                f"(0..{suite.n_tasks - 1})"
            )

        task = suite.get_task(task_index)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # instruction_override exists for the wrong-instruction ablation: the
        # SCENE stays this task's, but the policy hears a different sentence
        self._instruction = instruction_override or task.language
        self.task_id = f"{task_suite}/{task_index}"
        self.max_steps = max_steps
        self.n_tasks = suite.n_tasks

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution
        )
        self._env.seed(seed)
        with torch_load_compat():  # LIBERO's torch.load predates weights_only=True
            self._init_states = suite.get_task_init_states(task_index)
        self._init_state_index = init_state_index % len(self._init_states)
        self.t = 0
        self._done = False

    # -- Env protocol --------------------------------------------------------

    @property
    def instruction(self) -> str:
        return self._instruction

    def reset(self) -> Observation:
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[self._init_state_index])
        self.t = 0
        self._done = False
        return self._wrap(raw, 0.0)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        raw, reward, done, info = self._env.step(self._to_libero(action).tolist())
        self.t += 1
        timeout = self.t >= self.max_steps
        self._done = bool(done) or timeout
        info = dict(info)
        info["reward"] = float(reward)
        info["success"] = bool(done)
        if timeout and not done:
            info["event"] = "timeout"
        return self._wrap(raw, float(reward)), float(reward), self._done, info

    def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover
            self._env.close()

    # -- conversions ---------------------------------------------------------

    @staticmethod
    def _to_libero(action: Action) -> np.ndarray:
        """OpenVLA action convention -> LIBERO OSC controller convention.

        Mirrors `normalize_gripper_action` + `invert_gripper_action` from
        openvla's `experiments/robot/robot_utils.py`: OpenVLA emits the gripper
        dim in [0, 1], LIBERO's controller wants [-1, +1] with the opposite
        sign. Getting this wrong is the classic "the arm never grasps" bug, so
        it lives in exactly one place.
        """
        a = np.array(action.vector, dtype=np.float64, copy=True)
        a[-1] = 2.0 * a[-1] - 1.0  # [0,1] -> [-1,+1]
        a[-1] = float(np.sign(a[-1])) if a[-1] != 0 else -1.0  # binarize
        a[-1] *= -1.0  # invert
        return a

    def _wrap(self, raw: dict, reward: float) -> Observation:
        # LIBERO returns images flipped; the official eval un-flips with [::-1, ::-1].
        agent = np.asarray(raw["agentview_image"])[::-1, ::-1].copy()
        wrist = raw.get("robot0_eye_in_hand_image")
        wrist = np.asarray(wrist)[::-1, ::-1].copy() if wrist is not None else None
        eef = np.asarray(raw.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
        eef_quat = np.asarray(
            raw.get("robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        gripper_q = np.asarray(raw.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
        return Observation(
            image=agent.astype(np.uint8),
            wrist_image=None if wrist is None else wrist.astype(np.uint8),
            instruction=self._instruction,
            step=self.t,
            proprio=eef,
            # Only what a verifier legitimately needs. NOT full object state -
            # LIBERO's success signal is the reward, not privileged geometry.
            privileged={
                "ee": eef.tolist(),
                "ee_quat": eef_quat.tolist(),
                "gripper_qpos": gripper_q.tolist(),
                "success": bool(reward > 0),
            },
        )


def list_tasks(task_suite: str = "libero_spatial") -> list[tuple[int, str]]:
    """(index, language) for every task in a suite. Used by `make libero-tasks`."""
    ok, reason = probe()
    if not ok:
        raise LiberoUnavailable(reason)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[task_suite]()
    return [(i, suite.get_task(i).language) for i in range(suite.n_tasks)]
