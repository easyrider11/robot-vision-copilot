"""Policy x environment compatibility checks.

Some pairings are technically runnable but scientifically meaningless. The
worst one is `--env libero --backend mock`: the scripted controller servos on
TabletopSim's privileged state, which LIBERO does not provide, so it emits
zeros forever and the task fails at the step limit.

Silently producing "LIBERO success rate: 0%" from that setup would be actively
misleading. So the pairing is allowed - it is a genuinely useful plumbing test
- but it is announced in the loudest terms available, everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompatNote:
    level: str  # "error" | "warn" | "info"
    title: str
    lines: list[str]


def check(policy_kind: str, env_kind: str) -> list[CompatNote]:
    notes: list[CompatNote] = []

    if env_kind == "libero" and policy_kind == "mock":
        notes.append(CompatNote(
            level="error",
            title="MOCK 策略无法完成 LIBERO 任务",
            lines=[
                "脚本 mock 策略伺服的是 TabletopSim 的特权状态（物体/盒子坐标），",
                "LIBERO 不提供这些，所以它会持续输出零动作，任务必然失败。",
                "",
                "这次运行只能验证「管线是否打通」——环境创建、渲染、动作转换、",
                "日志、状态机是否正常。它 **不是** 一次 LIBERO 评测，",
                "得到的成功率不代表任何模型的能力。",
                "",
                "要拿到真实结果：--backend openvla-remote 配合云 GPU 上的",
                "openvla/openvla-7b-finetuned-libero-spatial（见 docs/04-real-openvla.md）。",
            ],
        ))

    if env_kind == "libero" and policy_kind in ("openvla-local", "openvla-remote"):
        notes.append(CompatNote(
            level="info",
            title="LIBERO 评测提示",
            lines=[
                "确认 unnorm_key 与检查点匹配，否则动作缩放会错：",
                "  openvla-7b-finetuned-libero-spatial -> unnorm_key=libero_spatial",
                "用 base 检查点 + bridge_orig 跑 LIBERO 是常见的错误对照。",
            ],
        ))

    if env_kind == "tabletop" and policy_kind in ("openvla-local", "openvla-remote"):
        notes.append(CompatNote(
            level="warn",
            title="真实 OpenVLA 驱动内置 tabletop 仿真",
            lines=[
                "内置 tabletop 是俯视正交渲染的教学仿真，与 OpenVLA 的训练分布",
                "（Bridge / LIBERO 的真实相机视角）相差很远，模型多半表现很差。",
                "这不能说明模型有问题。真实评测请用 --env libero。",
            ],
        ))

    return notes
