"""Stage 0 - re-runnable environment audit.

`make audit`. Reads only; installs nothing, downloads nothing, needs no sudo.
Writes docs/00-environment-audit.json so the report can be diffed over time
(e.g. after you free disk space or move to a GPU box).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENVLA_WEIGHTS_GB = 15.08


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _sysctl(key: str) -> str:
    return _sh(["sysctl", "-n", key])


def collect() -> dict:
    d: dict = {}
    d["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "version": platform.platform(),
    }
    if sys.platform == "darwin":
        d["os"]["macos"] = _sh(["sw_vers", "-productVersion"])
        d["cpu"] = {
            "brand": _sysctl("machdep.cpu.brand_string"),
            "model": _sysctl("hw.model"),
            "cores": _sysctl("hw.ncpu"),
        }
        mem = _sysctl("hw.memsize")
        d["memory_gb"] = round(int(mem) / 1024**3, 1) if mem.isdigit() else None
    else:
        d["cpu"] = {"brand": platform.processor(), "cores": str(os.cpu_count())}
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            d["memory_gb"] = round(kb / 1024**2, 1)
        except Exception:
            d["memory_gb"] = None

    usage = shutil.disk_usage(str(REPO_ROOT))
    d["disk"] = {
        "total_gb": round(usage.total / 1024**3, 1),
        "free_gb": round(usage.free / 1024**3, 1),
    }

    # GPU
    gpu: dict = {"cuda": False, "mps": False, "nvidia_smi": bool(shutil.which("nvidia-smi"))}
    if gpu["nvidia_smi"]:
        gpu["nvidia_smi_out"] = _sh(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"]
        )
    try:
        import torch

        gpu["torch"] = torch.__version__
        gpu["cuda"] = torch.cuda.is_available()
        if gpu["cuda"]:
            gpu["cuda_device"] = torch.cuda.get_device_name(0)
            gpu["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        m = getattr(torch.backends, "mps", None)
        gpu["mps"] = bool(m and m.is_available())
    except Exception as exc:
        gpu["torch"] = f"not installed ({type(exc).__name__})"
    d["gpu"] = gpu

    # tooling
    tools = {}
    for t in ("python3", "git", "node", "npm", "docker", "colima", "podman",
              "make", "cmake", "ros2", "gz", "gazebo", "uv"):
        p = shutil.which(t)
        tools[t] = p or None
    d["tools"] = tools
    d["python"] = {"executable": sys.executable, "version": sys.version.split()[0]}

    # python packages that matter
    pkgs = {}
    for mod in ("numpy", "PIL", "torch", "transformers", "fastapi", "uvicorn",
                "robosuite", "mujoco", "libero", "ultralytics", "cv2"):
        try:
            m = __import__(mod)
            pkgs[mod] = getattr(m, "__version__", "installed")
        except Exception:
            pkgs[mod] = None
    d["packages"] = pkgs

    # verdicts
    free = d["disk"]["free_gb"]
    ram = d["memory_gb"] or 0
    can_local_vla = (
        bool(gpu.get("cuda"))
        and gpu.get("vram_gb", 0) >= 15
        and free > OPENVLA_WEIGHTS_GB + 8
    )
    d["verdict"] = {
        "openvla_local": can_local_vla,
        "openvla_local_reason": (
            "OK"
            if can_local_vla
            else (
                "no CUDA GPU" if not gpu.get("cuda")
                else f"VRAM {gpu.get('vram_gb')}GB or free disk {free}GB insufficient "
                     f"(weights alone = {OPENVLA_WEIGHTS_GB} GB)"
            )
        ),
        "openvla_remote": "always possible - run rvc.service.vla_server on a cloud GPU",
        "libero": pkgs["robosuite"] is not None and pkgs["libero"] is not None,
        "gazebo_ros2": bool(tools["ros2"]) or bool(tools["docker"]) or bool(tools["colima"]),
        "lora_finetune": can_local_vla and gpu.get("vram_gb", 0) >= 40,
        "notes": [
            f"free disk {free} GB vs {OPENVLA_WEIGHTS_GB} GB of OpenVLA weights",
            f"system RAM {ram} GB",
        ],
    }
    return d


def render(d: dict) -> str:
    g = d["gpu"]
    v = d["verdict"]
    L = [
        "=" * 78,
        " Stage 0 · 环境审计 Environment Audit",
        "=" * 78,
        f"  OS            : {d['os'].get('macos') or d['os']['release']} "
        f"({d['os']['system']} {d['os']['machine']})",
        f"  CPU           : {d['cpu']['brand']}  ×{d['cpu']['cores']}",
        f"  RAM           : {d['memory_gb']} GB",
        f"  Disk          : {d['disk']['free_gb']} GB free / {d['disk']['total_gb']} GB",
        f"  GPU CUDA      : {g.get('cuda')}  {g.get('cuda_device', '')} "
        f"{('VRAM ' + str(g.get('vram_gb')) + 'GB') if g.get('vram_gb') else ''}",
        f"  GPU MPS       : {g.get('mps')}   (Apple Metal)",
        f"  torch         : {g.get('torch')}",
        f"  Python        : {d['python']['version']}  @ {d['python']['executable']}",
        "-" * 78,
        "  工具 tools:",
    ]
    for t, p in d["tools"].items():
        L.append(f"    {'✓' if p else '✗'} {t:<10} {p or '(not found)'}")
    L.append("-" * 78)
    L.append("  关键 Python 包:")
    for k, val in d["packages"].items():
        L.append(f"    {'✓' if val else '✗'} {k:<14} {val or '(not installed)'}")
    L += [
        "-" * 78,
        "  判定 verdict:",
        f"    OpenVLA-7B 本地真实推理 : {'YES' if v['openvla_local'] else 'NO'}  "
        f"— {v['openvla_local_reason']}",
        f"    OpenVLA 远程推理        : {v['openvla_remote']}",
        f"    LIBERO 已安装           : {'YES' if v['libero'] else 'NO'}",
        f"    ROS 2 / Gazebo 可运行   : "
        f"{'YES' if v['gazebo_ros2'] else 'NO (需要 Docker 或 Linux)'}",
        f"    LoRA 微调 7B            : {'YES' if v['lora_finetune'] else 'NO (需要 ≥40GB VRAM)'}",
        "=" * 78,
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    d = collect()
    print(render(d))
    out = REPO_ROOT / "docs" / "00-environment-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  JSON 已写入 {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
