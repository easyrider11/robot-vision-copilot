"""Stage 2 - the observable service.

    uvicorn rvc.service.app:app --reload --port 8080

Endpoints
    GET  /health              host + backend capability report
    POST /infer               image + instruction -> validated 7-DoF action
    POST /episode             run a full agent episode, return the summary
    GET  /runs                list past runs
    GET  /runs/{id}           one run's summary + step log
    GET  /runs/{id}/frames/*  the saved observation frames / rollout.gif
    GET  /                    a self-contained dashboard (no CDN, no build step)

The service is a thin shell over the same objects Stage 1 uses. It adds no
robot logic of its own - that is deliberate, so the HTTP layer can never drift
from what `make demo-libero` does.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from rvc import __version__
from rvc.agent.validators import ActionValidator
from rvc.envs.tabletop import TabletopSim
from rvc.perception.detector import ColorDetector
from rvc.policies.registry import Resolution, resolve_policy
from rvc.service.schemas import (
    ActionOut,
    BackendInfo,
    DetectionOut,
    EpisodeIn,
    EpisodeOut,
    HealthOut,
    InferIn,
    InferOut,
    PerceptionOut,
    RunSummary,
    ValidationOut,
)
from rvc.types import Observation

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"

app = FastAPI(
    title="Robot Vision Copilot",
    version=__version__,
    description="OpenVLA action model + agent layer. Every response states whether the "
    "action came from a real VLA or from the degraded scripted fallback.",
)

_detector = ColorDetector()
_resolution: Resolution | None = None


# ---------------------------------------------------------------------------
# backend cache
# ---------------------------------------------------------------------------


def get_resolution(backend: str = "auto") -> Resolution:
    global _resolution
    if _resolution is None or _resolution.chosen != backend != "auto":
        _resolution = resolve_policy(backend)
    return _resolution


def _backend_info(res: Resolution) -> BackendInfo:
    return BackendInfo(
        name=getattr(res.policy, "name", res.chosen),
        kind=res.chosen,  # type: ignore[arg-type]
        degraded=res.degraded,
        degraded_reason=res.degraded_reason,
        attempts=res.attempts,
    )


def _warnings(res: Resolution) -> list[str]:
    if not res.degraded:
        return []
    return [
        "DEGRADED: 该动作不是 OpenVLA-7B 推理结果，不能作为模型性能证据。",
        res.degraded_reason,
    ]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    from rvc.runners.audit import collect

    audit = collect()
    res = get_resolution()
    model_loaded = not res.degraded

    from rvc.envs.libero_env import probe as libero_probe
    from rvc.policies.openvla_local import probe as vla_probe

    vla_ok, vla_why = vla_probe()
    lib_ok, lib_why = libero_probe()

    return HealthOut(
        status="ok" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
        backend=_backend_info(res),
        host={
            "os": audit["os"].get("macos") or audit["os"]["release"],
            "cpu": audit["cpu"]["brand"],
            "ram_gb": audit["memory_gb"],
            "disk_free_gb": audit["disk"]["free_gb"],
            "cuda": audit["gpu"].get("cuda"),
            "mps": audit["gpu"].get("mps"),
            "vram_gb": audit["gpu"].get("vram_gb"),
        },
        capabilities={
            "openvla_local": {"available": vla_ok, "reason": vla_why or "OK"},
            "libero": {"available": lib_ok, "reason": lib_why or "OK"},
            "tabletop_sim": {"available": True, "reason": "built-in, always degraded"},
            "ros2_gazebo": {
                "available": bool(audit["tools"]["ros2"] or audit["tools"]["docker"]),
                "reason": "Stage 3; needs ROS 2 natively or Docker",
            },
        },
        warnings=_warnings(res),
    )


# ---------------------------------------------------------------------------
# POST /infer
# ---------------------------------------------------------------------------


def _decode_image(image_b64: str | None) -> tuple[np.ndarray, dict]:
    """Decode a caller-supplied image, or fall back to a fresh tabletop frame."""
    if not image_b64:
        sim = TabletopSim()
        obs = sim.reset()
        return obs.image, obs.privileged

    raw = image_b64.split(",", 1)[-1]  # tolerate data: URLs
    try:
        blob = base64.b64decode(raw, validate=True)
        img = Image.open(io.BytesIO(blob)).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(422, f"image_b64 不是合法的 base64 图像: {exc}") from exc
    return np.asarray(img, dtype=np.uint8), {}


@app.post("/infer", response_model=InferOut)
def infer(body: InferIn) -> InferOut:
    t_total = time.perf_counter()
    image, privileged = _decode_image(body.image_b64)

    res = get_resolution()
    obs = Observation(image=image, instruction=body.instruction, privileged=privileged)

    t0 = time.perf_counter()
    raw_action = res.policy.predict(obs)
    policy_ms = (time.perf_counter() - t0) * 1000

    if body.validate_action:
        ee = np.asarray(privileged["ee"], dtype=np.float32) if privileged.get("ee") else None
        action, ok, note = ActionValidator().validate(raw_action, ee)
    else:
        action, ok, note = raw_action, True, "validation skipped"

    dets = _detector.detect(image, ("red_block", "blue_box"))

    return InferOut(
        request_id=uuid.uuid4().hex[:12],
        instruction=body.instruction,
        action=ActionOut(
            vector=action.to_list(),
            delta_xyz=[round(float(v), 5) for v in action.delta_xyz],
            delta_rpy=[round(float(v), 5) for v in action.delta_rpy],
            gripper=action.gripper,
            gripper_label="CLOSE" if action.gripper > 0.5 else "OPEN",
        ),
        validation=ValidationOut(
            ok=ok, note=note, clamped=bool(note) and ok, raw_vector=raw_action.to_list()
        ),
        perception=PerceptionOut(
            detector=_detector.name,
            detections=[
                DetectionOut(
                    label=d.label,
                    confidence=d.confidence,
                    bbox_px=list(d.bbox_px),
                    center_world=[round(v, 4) for v in d.center_world],
                )
                for d in dets
            ],
            target_found=any(d.label == "red_block" for d in dets),
        ),
        backend=_backend_info(res),
        latency_ms={
            "policy": round(policy_ms, 2),
            "total": round((time.perf_counter() - t_total) * 1000, 2),
        },
        warnings=_warnings(res),
        image_size=[int(image.shape[1]), int(image.shape[0])],
    )


# ---------------------------------------------------------------------------
# POST /episode
# ---------------------------------------------------------------------------


@app.post("/episode", response_model=EpisodeOut)
def episode(body: EpisodeIn) -> EpisodeOut:
    from rvc.agent.state_machine import AgentConfig, RobotAgent
    from rvc.agent.verifier import RewardVerifier, TabletopVerifier
    from rvc.envs.registry import resolve_env
    from rvc.runners.demo_libero import write_artifacts

    pol = resolve_policy(body.backend)
    env = resolve_env(
        body.env,
        task_id=body.task,
        max_steps=body.max_steps,
        inject=body.inject,
        seed=body.seed,
       policy_kind=pol.chosen,
    )

    agent = RobotAgent(
        env=env.env,
        policy=pol.policy,
        verifier=TabletopVerifier() if env.chosen == "tabletop" else RewardVerifier(),
        detector=ColorDetector(),
        config=AgentConfig(
            max_recoveries=body.max_recoveries,
            max_total_steps=body.max_steps,
            mode=body.mode,
            perceive_required=env.chosen == "tabletop",
        ),
        collect_frames=body.save_frames,
    )

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{env.chosen}_{pol.chosen}_{body.inject}"
    run_dir = RUNS_DIR / run_id
    result = agent.run(run_dir=str(run_dir))
    write_artifacts(run_dir, agent, result, extra={
        "policy_attempts": pol.attempts,
        "env_attempts": env.attempts,
        "args": body.model_dump(),
        "source": "http",
    })
    with contextlib.suppress(Exception):
        env.env.close()

    return EpisodeOut(
        run_id=run_id,
        success=result.success,
        steps=result.steps,
        recoveries=result.recoveries,
        final_state=result.final_state.value,
        failure=result.failure.value,
        degraded=result.degraded,
        degraded_reason=result.degraded_reason,
        injected_fault=result.injected_fault,
        wall_time_s=result.wall_time_s,
        instruction=result.instruction,
        backend=result.backend,
        state_timeline=[
            {"step": t.step, "from": t.frm.value, "to": t.to.value, "reason": t.reason}
            for t in agent.trace.transitions
        ],
        artifacts={
            "summary": f"/runs/{run_id}",
            "gif": f"/runs/{run_id}/frames/rollout.gif",
            "actions_jsonl": f"/runs/{run_id}/actions.jsonl",
        },
    )


# ---------------------------------------------------------------------------
# run browsing
# ---------------------------------------------------------------------------


def _safe_run_dir(run_id: str) -> Path:
    d = (RUNS_DIR / run_id).resolve()
    if not str(d).startswith(str(RUNS_DIR.resolve())) or not d.is_dir():
        raise HTTPException(404, f"unknown run {run_id!r}")
    return d


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 50) -> list[RunSummary]:
    out: list[RunSummary] = []
    if not RUNS_DIR.is_dir():
        return out
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        f = d / "summary.json"
        if not f.is_file():
            continue
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(RunSummary(
            run_id=d.name,
            success=bool(s.get("success")),
            steps=int(s.get("steps", 0)),
            recoveries=int(s.get("recoveries", 0)),
            degraded=bool(s.get("degraded")),
            injected_fault=str(s.get("injected_fault", "none")),
            started_at=float(s.get("started_at", 0.0)),
        ))
        if len(out) >= limit:
            break
    return out


@app.get("/runs/{run_id}")
def get_run(run_id: str, max_steps: int = 400) -> dict[str, Any]:
    d = _safe_run_dir(run_id)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    steps = []
    log = d / "actions.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines()[:max_steps]:
            if line.strip():
                steps.append(json.loads(line))
    frames = sorted(p.name for p in (d / "frames").glob("*.png")) if (d / "frames").is_dir() else []
    return {"run_id": run_id, "summary": summary, "steps": steps, "frames": frames,
            "has_gif": (d / "rollout.gif").is_file()}


@app.get("/runs/{run_id}/actions.jsonl")
def get_run_log(run_id: str) -> FileResponse:
    f = _safe_run_dir(run_id) / "actions.jsonl"
    if not f.is_file():
        raise HTTPException(404, "no actions.jsonl for this run")
    return FileResponse(f, media_type="application/x-ndjson")


@app.get("/runs/{run_id}/frames/{name}")
def get_frame(run_id: str, name: str) -> FileResponse:
    d = _safe_run_dir(run_id)
    if name == "rollout.gif":
        f = d / "rollout.gif"
    else:
        if "/" in name or ".." in name:
            raise HTTPException(400, "bad frame name")
        f = d / "frames" / name
    if not f.is_file():
        raise HTTPException(404, f"no such frame {name!r}")
    return FileResponse(f, media_type="image/gif" if f.suffix == ".gif" else "image/png")


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

DASHBOARD = (Path(__file__).parent / "dashboard.html")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))
