"""Compatibility glue that makes LIBERO importable and non-interactive.

LIBERO is a research repo, not a packaged library. Getting it to import on a
2026 macOS arm64 box took six separate fixes; each one is applied here with the
symptom it produces, so the next person does not have to rediscover them.

    1. `pip install git+…LIBERO` installs NOTHING usable.
       `libero/` has no top-level `__init__.py` (implicit namespace package),
       and their setup.py uses `find_packages()`, which skips such directories.
       -> vendor a clone in external/LIBERO and put it on sys.path.
       Symptom: `ModuleNotFoundError: No module named 'libero'` right after a
       "successfully installed libero-0.1.0" message.

    2. First import PROMPTS on stdin for a dataset path and hangs / EOFErrors
       in any non-interactive context.
       -> pre-write config.yaml and point LIBERO_CONFIG_PATH at a project-local
       directory, so we also never touch the user's ~/.libero.
       Symptom: `EOFError: EOF when reading a line` inside libero/__init__.py.

    3. `libero.libero.benchmark` imports torch at module scope, even though the
       simulator itself does not need it.
       -> torch is a hard dependency of the `libero` extra.

    4. `get_task_init_states` calls `torch.load()` without `weights_only`.
       torch >= 2.6 flipped that default to True, and the .pruned_init files are
       pickled numpy arrays.
       -> `torch_load_compat()` below, scoped to just that call.
       Symptom: `UnpicklingError: Weights only load failed … numpy.core.
       multiarray._reconstruct was not an allowed global`.

    5. robosuite 1.4.1 calls `mujoco.mj_fullM(…, self.sim.data.qM)`. MuJoCo
       renamed `MjData.qM` to `MjData.M` in 3.3.0.
       -> pin `mujoco>=3.2,<3.3` in the `libero` extra.
       Symptom: `AttributeError: 'MjData' object has no attribute 'qM'`.

    6. Their requirements.txt pins numpy==1.22.4, robomimic, wandb,
       transformers==4.21.1 — all for the *lifelong learning* training code,
       none of it needed to run the simulator, and installing it would drag
       numpy back to 1.22.
       -> we install only what `libero/libero/**` actually imports:
       cloudpickle, gym, h5py, huggingface_hub, matplotlib, easydict, bddl.

Everything here is idempotent and safe to call repeatedly.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "external" / "LIBERO"
CONFIG_DIR = REPO_ROOT / "external" / ".libero-config"
DATASETS_DIR = REPO_ROOT / "external" / "libero-datasets"

_done = False


def _write_config() -> None:
    """Create LIBERO's config.yaml so its first-import prompt never fires."""
    import yaml

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_DIR / "config.yaml"
    if cfg.is_file():
        return
    root = VENDORED / "libero" / "libero"
    yaml.safe_dump(
        {
            "benchmark_root": str(root),
            "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"),
            # Demonstration HDF5s are a separate multi-GB download we do NOT
            # need for evaluation. Pointing at an existing empty dir keeps
            # LIBERO from printing a warning on every path lookup.
            "datasets": str(DATASETS_DIR),
            "assets": str(root / "assets"),
        },
        cfg.open("w"),
    )


def bootstrap() -> tuple[bool, str]:
    """Make `import libero` work. Returns (ok, reason). Never raises."""
    global _done
    if _done:
        return True, ""

    # LIBERO must not prompt, and must not write to the user's home dir.
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(CONFIG_DIR))

    # Prefer an already-installed libero; fall back to the vendored clone.
    try:
        import libero  # noqa: F401
    except ImportError:
        if not (VENDORED / "libero" / "libero" / "__init__.py").is_file():
            return False, (
                f"LIBERO not importable and no vendored clone at {VENDORED}. "
                "Run: make setup-libero"
            )
        sys.path.insert(0, str(VENDORED))

    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        return False, f"pyyaml missing ({exc}); run: make setup-libero"

    _write_config()
    _done = True
    return True, ""


@contextlib.contextmanager
def torch_load_compat():
    """Restore torch.load's pre-2.6 `weights_only=False` default, scoped.

    LIBERO's `.pruned_init` files are pickled numpy arrays shipped in the
    official repo, so this is the same trust decision as cloning it. Scoped to
    a single call rather than patched globally on purpose - unpickling
    arbitrary files is exactly the thing you do not want silently enabled
    process-wide.
    """
    import torch

    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        yield
    finally:
        torch.load = original
