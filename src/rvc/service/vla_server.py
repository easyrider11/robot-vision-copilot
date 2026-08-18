"""Remote OpenVLA inference server - run this ON THE GPU BOX, not on the laptop.

This is the other half of `rvc.policies.openvla_remote`. It does one thing:
hold `openvla/openvla-7b` in VRAM and answer `POST /predict_action`.

    # on a cloud GPU (A100 40/80GB, L40S, or a 24GB 4090 with --4bit)
    git clone <this repo> && cd robot-vision-copilot
    uv venv --python 3.10 && uv pip install -e ".[vla,api]"
    RVC_MODEL_ID=openvla/openvla-7b RVC_UNNORM_KEY=bridge_orig \
      uvicorn rvc.service.vla_server:app --host 0.0.0.0 --port 8000

    # back on the laptop
    make demo-libero BACKEND=openvla-remote RVC_VLA_URL=http://<gpu-host>:8000

The model is loaded lazily on the first request so `/health` answers instantly
and you can confirm the box is reachable before paying the ~2 minute load.

SECURITY: bind to 127.0.0.1 and use an SSH tunnel unless the box is on a
trusted network - there is no auth here on purpose, keeping it a teaching
artifact rather than something you would expose to the internet.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from rvc.service.schemas import PredictActionIn, PredictActionOut
from rvc.types import Observation

MODEL_ID = os.environ.get("RVC_MODEL_ID", "openvla/openvla-7b")
UNNORM_KEY = os.environ.get("RVC_UNNORM_KEY", "bridge_orig")
LOAD_IN_4BIT = os.environ.get("RVC_4BIT", "0") == "1"

app = FastAPI(title="OpenVLA inference server", version="0.1.0")

_policy = None
_load_error = ""
_lock = threading.Lock()


def _get_policy():
    global _policy, _load_error
    if _policy is not None:
        return _policy
    with _lock:
        if _policy is None and not _load_error:
            from rvc.policies.openvla_local import OpenVLALocalPolicy

            try:
                _policy = OpenVLALocalPolicy(
                    model_id=MODEL_ID, unnorm_key=UNNORM_KEY, load_in_4bit=LOAD_IN_4BIT
                )
            except Exception as exc:
                _load_error = f"{type(exc).__name__}: {exc}"
    if _policy is None:
        raise HTTPException(503, f"model not loaded: {_load_error}")
    return _policy


@app.get("/health")
def health() -> dict:
    from rvc.policies.openvla_local import probe

    ok, why = probe()
    return {
        "status": "ok" if (ok or _policy is not None) else "error",
        "model_loaded": _policy is not None,
        "model_id": MODEL_ID,
        "unnorm_key": UNNORM_KEY,
        "load_in_4bit": LOAD_IN_4BIT,
        "can_load": ok,
        "reason": why or "OK",
        "load_error": _load_error,
    }


@app.post("/load")
def load() -> dict:
    """Eagerly load the weights so the first control step is not 2 minutes long."""
    t0 = time.perf_counter()
    _get_policy()
    return {"loaded": True, "seconds": round(time.perf_counter() - t0, 1)}


@app.post("/predict_action", response_model=PredictActionOut)
def predict_action(body: PredictActionIn) -> PredictActionOut:
    policy = _get_policy()
    try:
        img = Image.open(io.BytesIO(base64.b64decode(body.image_b64))).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, f"bad image_b64: {exc}") from exc

    policy.unnorm_key = body.unnorm_key or UNNORM_KEY
    t0 = time.perf_counter()
    action = policy.predict(
        Observation(image=np.asarray(img, dtype=np.uint8), instruction=body.instruction)
    )
    return PredictActionOut(
        action=action.to_list(),
        model_id=MODEL_ID,
        unnorm_key=policy.unnorm_key,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
