"""Local SmolVLA inference server - a REAL VLA on Apple Silicon.

Serves `lerobot/smolvla_libero` (SmolVLA-450M finetuned on the LIBERO suites)
over the same HTTP shape as `vla_server` so the robotics stack stays in the
main venv while inference lives in `.venv-lerobot` (lerobot pins its own
torch; mixing it into the LIBERO venv would risk the working BC setup).

    .venv-lerobot/bin/python -m rvc.service.smolvla_server --port 8100

Endpoints:
    GET  /health          {"model_loaded": true, "model_id": ..., "device": ...}
    POST /reset           clears the action-chunk queue (call at episode start)
    POST /predict_action  {"image_b64", "wrist_b64", "state": [8], "instruction"}
                          -> {"action": [7] in LIBERO env space, "latency_ms",
                              "queue": remaining chunk length}

The checkpoint predicts 50-step action chunks (n_action_steps=50): one heavy
forward (~1-5 s on M3 MPS) then ~2 ms queue pops - report BOTH numbers, never
just the flattering one. stdlib HTTP only: no fastapi in this venv.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "lerobot/smolvla_libero"


def load(model_id: str, device: str):
    import torch  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def _decode(b64: str):
    import numpy as np
    import torch
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def make_handler(policy, pre, post, model_id: str, device: str):
    import torch

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {
                    "model_loaded": True, "model_id": model_id, "device": device,
                    "params_m": round(sum(p.numel() for p in policy.parameters()) / 1e6),
                    "chunk_size": policy.config.chunk_size,
                    "n_action_steps": policy.config.n_action_steps,
                })
            else:
                self._json(404, {"error": self.path})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/reset":
                policy.reset()
                self._json(200, {"ok": True})
                return
            if self.path != "/predict_action":
                self._json(404, {"error": self.path})
                return
            try:
                t0 = time.perf_counter()
                obs = {
                    "observation.images.image": _decode(req["image_b64"]),
                    "observation.images.image2": _decode(req["wrist_b64"]),
                    "observation.state": torch.tensor(req["state"], dtype=torch.float32),
                    "task": req["instruction"],
                }
                batch = pre(obs)
                with torch.no_grad():
                    act = policy.select_action(batch)
                act = post(act)
                ms = (time.perf_counter() - t0) * 1000
                self._json(200, {
                    "action": [float(x) for x in act.squeeze().tolist()],
                    "latency_ms": round(ms, 1),
                    "queue": len(policy._queues["action"]),
                })
            except Exception as exc:  # surface, don't hide
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)

    print(f"[smolvla-server] loading {args.model} on {args.device} ...", flush=True)
    t0 = time.time()
    policy, pre, post = load(args.model, args.device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[smolvla-server] loaded in {time.time() - t0:.1f}s - "
          f"REAL VLA, {n_params:.0f}M params", flush=True)
    handler = make_handler(policy, pre, post, args.model, args.device)
    srv = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"[smolvla-server] http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
