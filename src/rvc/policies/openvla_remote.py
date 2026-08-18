"""Remote OpenVLA client - the practical way to get REAL VLA actions from a
laptop that cannot host a 7B model.

Topology:

    MacBook (this repo)                    Cloud GPU (A100/L40S/4090)
    ------------------                     --------------------------
    TabletopSim / LIBERO                   rvc.service.vla_server
    agent state machine     --HTTP-->      OpenVLALocalPolicy
    validators, recovery    <--action--    openvla/openvla-7b
    logging, FastAPI

The laptop keeps the entire robotics stack; only the 7B forward pass moves. One
image round-trip per control step, so latency dominates - that is why the
client reports `latency_ms` on every call and the runner prints it.

Server side (on the GPU box):
    uv pip install -e ".[vla,api]"
    uvicorn rvc.service.vla_server:app --host 0.0.0.0 --port 8000

Client side (here):
    make demo-libero BACKEND=openvla-remote RVC_VLA_URL=http://<gpu-host>:8000
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
from PIL import Image

from rvc.policies.base import PolicyUnavailable
from rvc.types import Action, Observation


def probe(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Ping the remote /health. Never raises."""
    try:
        import httpx
    except Exception as exc:
        return False, f"httpx not installed ({type(exc).__name__}). Run: uv pip install -e '.[api]'"
    try:
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:
        return False, f"remote VLA at {url} unreachable: {type(exc).__name__}: {exc}"
    if not body.get("model_loaded"):
        return False, f"remote VLA at {url} is up but reports model_loaded=false"
    return True, ""


class OpenVLARemotePolicy:
    """Posts (image, instruction) to a remote OpenVLA server, gets 7 floats back."""

    degraded = False
    degraded_reason = ""

    def __init__(self, url: str, unnorm_key: str = "bridge_orig", timeout: float = 60.0) -> None:
        ok, reason = probe(url)
        if not ok:
            raise PolicyUnavailable(reason)
        import httpx

        self.url = url.rstrip("/")
        self.unnorm_key = unnorm_key
        self.name = "openvla-remote"
        self._client = httpx.Client(timeout=timeout)
        self.last_latency_ms = 0.0

    def describe(self) -> str:
        return f"{self.name} -> {self.url} (unnorm_key={self.unnorm_key})"

    def predict(self, obs: Observation) -> Action:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(obs.image, dtype=np.uint8)).convert("RGB").save(buf, "PNG")
        payload = {
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "instruction": obs.instruction,
            "unnorm_key": self.unnorm_key,
        }
        t0 = time.perf_counter()
        r = self._client.post(f"{self.url}/predict_action", json=payload)
        r.raise_for_status()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return Action(np.asarray(r.json()["action"], dtype=np.float32))

    def close(self) -> None:
        self._client.close()
