#!/usr/bin/env bash
# Stage 3 bootstrap: colima + docker CLI + ROS 2/Gazebo image, then a headless
# smoke test.  `make ros-up`
#
# Written 2026-08-14 while the host had too little disk to actually run it -
# the flow below is untested as a whole. Each step is guarded and idempotent;
# rerun the same command after fixing whatever a step complains about.
#
# No sudo anywhere: colima is a userland VM (macOS Virtualization.framework),
# and `docker`/`docker-compose` here are CLI formulae, not Docker Desktop.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NEED_GB=6            # image build + VM headroom; refuse below this
CPU="${RVC_COLIMA_CPU:-4}"
MEM="${RVC_COLIMA_MEM:-4}"
DISK="${RVC_COLIMA_DISK:-15}"   # GB, sparse - caps VM growth on a tight host

free_gb=$(df -k . | awk 'NR==2{printf "%.1f", $4/1024/1024}')
echo "── Stage 3 bootstrap ──────────────────────────────────────────"
echo "  可用磁盘 : ${free_gb} GB   (安全下限 ${NEED_GB} GB)"
if awk "BEGIN{exit !(${free_gb} < ${NEED_GB})}"; then
  echo "✗ 磁盘不足。装完 macOS 更新或清理后重试（见 docs/03-ros2-gazebo.md）。"
  exit 1
fi

# 1. tooling ------------------------------------------------------------------
echo "[1/5] colima + docker CLI（brew 用户态安装，无 sudo）"
for f in colima docker docker-compose; do
  if ! command -v "$f" >/dev/null 2>&1; then
    brew install "$f"
  else
    echo "      ✓ $f 已存在"
  fi
done

# 2. VM -----------------------------------------------------------------------
echo "[2/5] 启动 colima 虚拟机 (cpu=${CPU} mem=${MEM}G disk=${DISK}G, vz 原生虚拟化)"
if colima status >/dev/null 2>&1; then
  echo "      ✓ colima 已在运行"
else
  # --vm-type vz: macOS Virtualization.framework - no qemu, faster, smaller
  colima start --cpu "$CPU" --memory "$MEM" --disk "$DISK" --vm-type vz
fi
docker info --format '      ✓ docker {{.ServerVersion}} ({{.OSType}}/{{.Architecture}})'

# 3. image --------------------------------------------------------------------
echo "[3/5] 构建 ROS 2 Jazzy + Gazebo 镜像（首次约 10-20 分钟，视网速）"
docker compose -f ros2_ws/docker-compose.yml build

# 4. world sanity -------------------------------------------------------------
echo "[4/5] 容器内校验 SDF 世界"
docker compose -f ros2_ws/docker-compose.yml run --rm ros2 \
  bash -lc "gz sdf --check /ws/src/rvc_agent/worlds/tabletop.sdf && echo '      ✓ tabletop.sdf 通过 gz 校验'"

# 5. headless smoke -----------------------------------------------------------
echo "[5/5] 无头冒烟：起 gz + bridge + agent_node 20 秒，抓状态与话题"
docker compose -f ros2_ws/docker-compose.yml run --rm ros2 bash -lc '
  set -e
  ros2 launch rvc_agent tabletop.launch.py &
  LP=$!
  sleep 12
  echo "--- topics ---";  ros2 topic list
  echo "--- /rvc/state (5s) ---"
  timeout 5 ros2 topic echo /rvc/state --once || true
  echo "--- camera hz ---"
  timeout 5 ros2 topic hz /camera/image_raw || true
  kill $LP 2>/dev/null || true
'

echo
echo "✓ Stage 3 就绪。常用命令："
echo "    make ros-shell                                   # 进容器"
echo "    ros2 launch rvc_agent tabletop.launch.py         # 容器内起全套"
echo "    ros2 topic echo /rvc/state                       # 看状态机"
echo "  首跑校准清单（相机取向 / axis_map 符号 / 颜色阈值）见 docs/03-ros2-gazebo.md"
