#!/usr/bin/env bash
# Reproducible LIBERO setup.  `make setup-libero`
#
# Verified on macOS 15.5 / Apple M3 / arm64 on 2026-08-13.
# Costs ~2.6 GB: ~650 MB repo clone + ~1.9 GB of wheels (torch, mujoco,
# robosuite, scipy, opencv, matplotlib, numba).
#
# Everything lands inside the project: the venv, the vendored clone in
# external/LIBERO, and LIBERO's config in external/.libero-config.
# Your ~/.libero is never touched. No sudo, no global installs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
VENDOR="$ROOT/external/LIBERO"
LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"

cd "$ROOT"
[[ -x "$PY" ]] || { echo "✗ 没有 .venv，先跑 make setup"; exit 1; }

free_gb=$(df -k . | awk 'NR==2{printf "%.1f", $4/1024/1024}')
echo "── LIBERO 安装 ────────────────────────────────────────────────"
echo "  可用磁盘 : ${free_gb} GB   (需要约 2.6 GB)"
echo "  venv     : $PY"
echo "  clone 到 : $VENDOR"
echo

# 1. Python wheels ------------------------------------------------------------
echo "[1/4] 安装 Python 依赖（robosuite 1.4.1 + mujoco<3.3 + torch + …）"
uv pip install --python "$PY" -e "$ROOT[libero]"

# 2. Vendored clone -----------------------------------------------------------
# `pip install git+…LIBERO` silently installs an EMPTY package: libero/ has no
# top-level __init__.py and their setup.py uses find_packages(), which skips
# implicit namespace packages. A clone on sys.path is the only thing that
# works. See src/rvc/envs/libero_bootstrap.py.
if [[ -d "$VENDOR/.git" ]]; then
  echo "[2/4] 已存在 clone，跳过（更新用: git -C external/LIBERO pull）"
else
  echo "[2/4] 浅克隆 LIBERO 仓库（约 650 MB，含 bddl 与 init_files 资产）"
  git clone --depth 1 "$LIBERO_REPO" "$VENDOR"
fi

# 3. Sanity: the pin that bites ------------------------------------------------
echo "[3/4] 校验 mujoco / robosuite 版本组合"
"$PY" - <<'EOF'
import sys, mujoco, robosuite
mj = tuple(int(p) for p in mujoco.__version__.split(".")[:2])
print(f"      mujoco    {mujoco.__version__}")
print(f"      robosuite {robosuite.__version__}")
if mj >= (3, 3):
    sys.exit(
        f"✗ mujoco {mujoco.__version__} 移除了 MjData.qM，robosuite 1.4.x 仍在使用。\n"
        '  修复: uv pip install "mujoco>=3.2,<3.3"'
    )
EOF

# 4. End-to-end check ---------------------------------------------------------
echo "[4/4] 端到端验证：建环境 + 离屏渲染 + 步进"
"$PY" - <<'EOF'
import time
import numpy as np
from rvc.envs.libero_env import LiberoEnv, probe
from rvc.types import Action

ok, why = probe()
if not ok:
    raise SystemExit(f"✗ probe 失败: {why}")

t0 = time.time()
env = LiberoEnv(task_suite="libero_spatial", task_index=0, max_steps=20)
print(f"      env 创建   {time.time()-t0:.1f}s")
print(f"      任务       {env.instruction!r}")
obs = env.reset()
img = np.asarray(obs.image)
assert img.shape == (256, 256, 3) and img.dtype == np.uint8, img.shape
assert img.std() > 5, "渲染出的是空白图"
print(f"      agentview  {img.shape} std={img.std():.1f}")
print(f"      wrist cam  {None if obs.wrist_image is None else obs.wrist_image.shape}")
t0 = time.time()
for _ in range(5):
    obs, r, done, info = env.step(Action.zeros())
print(f"      步进       {(time.time()-t0)/5*1000:.0f} ms/step")
env.close()
EOF

echo
echo "✓ LIBERO 就绪。试试："
echo "    make libero-tasks                     # 列出所有任务"
echo "    make demo-libero ENV=libero STEPS=60  # 管线验证（mock 策略不会成功，见提示）"
