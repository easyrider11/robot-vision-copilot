"""Request/response contracts for the Stage 2 service.

Design rule: every response that involved a model carries its provenance. A
caller must be able to tell, from the payload alone and without reading logs,
whether the numbers came from OpenVLA-7B or from the scripted fallback.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rvc.types import ACTION_LABELS

# --- shared -----------------------------------------------------------------


class BackendInfo(BaseModel):
    name: str = Field(description="解析到的动作模型后端，如 openvla:openvla-7b / scripted-mock")
    kind: Literal["openvla-local", "openvla-remote", "mock"]
    degraded: bool = Field(description="true = 这不是真实 VLA 推理结果")
    degraded_reason: str = ""
    attempts: list[tuple[str, str]] = Field(
        default_factory=list, description="每个后端尝试的结果：(backend, 'OK' 或失败原因)"
    )


class ActionOut(BaseModel):
    vector: list[float] = Field(description="7 维动作 [dx,dy,dz,droll,dpitch,dyaw,gripper]")
    labels: list[str] = list(ACTION_LABELS)
    delta_xyz: list[float]
    delta_rpy: list[float]
    gripper: float
    gripper_label: Literal["OPEN", "CLOSE"]


class ValidationOut(BaseModel):
    ok: bool = Field(description="false = 动作被安全校验拒绝，不应下发给机器人")
    note: str = ""
    clamped: bool = False
    raw_vector: list[float] = Field(description="校验前模型原始输出，便于对比")


class DetectionOut(BaseModel):
    label: str
    confidence: float
    bbox_px: list[int]
    center_world: list[float]


class PerceptionOut(BaseModel):
    detector: str
    detections: list[DetectionOut]
    target_found: bool


# --- /health ----------------------------------------------------------------


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    model_loaded: bool = Field(description="真实 VLA 是否已加载；mock 后端为 false")
    backend: BackendInfo
    host: dict[str, Any] = Field(description="CPU/RAM/磁盘/GPU 摘要，来自 Stage 0 审计")
    capabilities: dict[str, Any]
    warnings: list[str] = []


# --- /infer -----------------------------------------------------------------


class InferIn(BaseModel):
    instruction: str = Field(description="自然语言任务，例如 'pick up the red block'")
    image_b64: str | None = Field(
        default=None, description="base64 编码的 PNG/JPEG；留空则用内置 tabletop 初始观测"
    )
    unnorm_key: str | None = Field(default=None, description="覆盖 OpenVLA 的反归一化 key")
    validate_action: bool = True


class InferOut(BaseModel):
    request_id: str
    instruction: str
    action: ActionOut
    validation: ValidationOut
    perception: PerceptionOut
    backend: BackendInfo
    latency_ms: dict[str, float]
    warnings: list[str] = Field(
        default_factory=list, description="降级提示等；degraded 时这里一定非空"
    )
    image_size: list[int]


# --- /episode ---------------------------------------------------------------


class EpisodeIn(BaseModel):
    task: str = "pick_place_block"
    backend: Literal["auto", "openvla-local", "openvla-remote", "mock"] = "auto"
    env: Literal["auto", "libero", "tabletop"] = "auto"
    mode: Literal["subgoal", "e2e"] = "subgoal"
    inject: Literal["none", "target_lost", "grasp_fail", "grasp_slip"] = "none"
    max_steps: int = 160
    max_recoveries: int = 3
    seed: int = 0
    save_frames: bool = True


class EpisodeOut(BaseModel):
    run_id: str
    success: bool
    steps: int
    recoveries: int
    final_state: str
    failure: str
    degraded: bool
    degraded_reason: str
    injected_fault: str
    wall_time_s: float
    instruction: str
    backend: str
    state_timeline: list[dict[str, Any]]
    artifacts: dict[str, str]


class RunSummary(BaseModel):
    run_id: str
    success: bool
    steps: int
    recoveries: int
    degraded: bool
    injected_fault: str
    started_at: float


# --- remote VLA server ------------------------------------------------------


class PredictActionIn(BaseModel):
    image_b64: str
    instruction: str
    unnorm_key: str = "bridge_orig"


class PredictActionOut(BaseModel):
    action: list[float]
    model_id: str
    unnorm_key: str
    latency_ms: float
