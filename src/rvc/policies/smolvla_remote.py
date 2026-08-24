"""SmolVLA client - the first NON-degraded policy backend in this repo.

SmolVLA-450M (`lerobot/smolvla_libero`) is a genuine vision-language-action
model: SmolVLM2 backbone + flow-matching action expert, finetuned on the
LIBERO suites. Small enough for this machine's MPS, so the "real VLA" slot is
finally filled locally - OpenVLA-7B stays the roadmap target for a GPU box.

Wire format matches `rvc.service.smolvla_server`. The model emits actions in
the LIBERO env space (gripper -1 open / +1 close); this client converts the
gripper to the repo contract (OpenVLA convention, [0,1], 1 = open) so
`LiberoEnv._to_libero` can convert it back - one convention, one place,
pinned by tests/test_contract.py.
"""

from __future__ import annotations

import base64
import io
import math
import time

import numpy as np
from PIL import Image

from rvc.policies.base import PolicyUnavailable
from rvc.types import Action, Observation

DEFAULT_URL = "http://127.0.0.1:8100"


def quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis*angle, robosuite's convention - the same function
    the lerobot/libero dataset used to build its 8-dim state, so the model
    sees proprio in the distribution it was trained on."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (np.asarray(q[:3], dtype=np.float32) * 2.0 * math.acos(w)) / den


def libero_state8(obs: Observation) -> np.ndarray:
    """eef_pos(3) + axisangle(eef_quat)(3) + gripper_qpos(2)."""
    ee = np.asarray(obs.proprio if obs.proprio is not None else np.zeros(3), dtype=np.float32)
    quat = np.asarray(obs.privileged.get("ee_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    gq = np.asarray(obs.privileged.get("gripper_qpos", [0.0, 0.0]), dtype=np.float32)
    return np.concatenate([ee[:3], quat_to_axisangle(quat), gq[:2]]).astype(np.float32)


def _b64(img: np.ndarray | None) -> str:
    buf = io.BytesIO()
    arr = np.zeros((256, 256, 3), np.uint8) if img is None else np.asarray(img, np.uint8)
    Image.fromarray(arr).convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class SmolVLARemotePolicy:
    """Posts (both cameras, 8-dim state, instruction); gets 7 floats back."""

    degraded = False  # a real VLA - the first backend that can honestly say so
    degraded_reason = ""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        try:
            import httpx
        except Exception as exc:
            raise PolicyUnavailable(f"httpx not installed ({type(exc).__name__})") from exc
        self.url = url.rstrip("/")
        try:
            r = httpx.get(f"{self.url}/health", timeout=5.0)
            r.raise_for_status()
            self.health = r.json()
        except Exception as exc:
            raise PolicyUnavailable(
                f"smolvla server at {url} unreachable ({type(exc).__name__}) - "
                "start it: make smolvla-serve"
            ) from exc
        if not self.health.get("model_loaded"):
            raise PolicyUnavailable(f"smolvla server at {url} reports model_loaded=false")
        self.name = "smolvla-remote"
        self._client = httpx.Client(timeout=timeout)
        self.last_latency_ms = 0.0
        self.chunk_latencies_ms: list[float] = []  # heavy forwards only

    def describe(self) -> str:
        h = self.health
        return (f"{self.name} -> {self.url}  REAL VLA: {h.get('model_id')} "
                f"({h.get('params_m')}M, {h.get('device')}, "
                f"chunk={h.get('chunk_size')})")

    def reset(self) -> None:
        """Clear the server-side action-chunk queue. Call at episode start."""
        self._client.post(f"{self.url}/reset", json={})

    def predict(self, obs: Observation) -> Action:
        payload = {
            "image_b64": _b64(obs.image),
            "wrist_b64": _b64(obs.wrist_image),
            "state": [float(x) for x in libero_state8(obs)],
            "instruction": obs.instruction,
        }
        t0 = time.perf_counter()
        r = self._client.post(f"{self.url}/predict_action", json=payload)
        r.raise_for_status()
        body = r.json()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        if body.get("queue", 0) >= int(self.health.get("n_action_steps", 50)) - 1:
            self.chunk_latencies_ms.append(self.last_latency_ms)
        a = np.asarray(body["action"], dtype=np.float32)
        vec = np.zeros(7, dtype=np.float32)
        vec[:6] = np.clip(a[:6], -1.0, 1.0)
        vec[6] = float(np.clip((1.0 - a[6]) / 2.0, 0.0, 1.0))  # LIBERO -> contract
        return Action(vec)

    def close(self) -> None:
        self._client.close()
