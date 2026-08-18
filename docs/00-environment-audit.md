# Stage 0 · 环境审计

实测时间 2026-08-12。用 `make audit` 可随时重跑（只读，不安装、不下载、不用 `sudo`），
结果同时写入 `docs/00-environment-audit.json` 便于日后 diff。

## 实测结果

| 项目 | 值 |
|---|---|
| OS | macOS 15.5 (24F74) · Darwin 24.5.0 · arm64 |
| 机型 | MacBook Air `Mac15,13` |
| CPU | Apple M3 · 8 核 |
| GPU | Apple M3 10 核 · Metal 3 · **无 NVIDIA / 无 CUDA** |
| 内存 | **16 GB 统一内存**（CPU 与 GPU 共享） |
| 磁盘 | 460 GB 总量 / **28.1 GB 可用**（94% 已满） |
| Python | 3.14.5 (brew) · 3.11.2 · 3.10.6 (framework) |
| 包管理 | **uv 0.11.19** ✅ · 无 conda / pyenv / poetry |
| 容器 | **无 Docker / Colima / Podman** |
| ROS 2 / Gazebo | **未安装**（macOS 亦无官方二进制） |
| 其他 | git 2.50.1 ✅ · node 22.9 ✅ · make ✅ · clang ✅ · brew ✅ · cmake ✗ |

## OpenVLA-7B 权重实测

`openvla/openvla-7b` 的 Hub 元数据（只查元数据，未下载）：

| 字段 | 值 |
|---|---|
| gated | **False**（不需要申请授权） |
| private | False |
| license | **MIT** |
| 权重分片 | 3 × safetensors |
| **合计大小** | **15.08 GB** |

> 注意：模型本身 MIT 且不 gated，但加载需要 `trust_remote_code=True`，即执行从 Hub 下载的
> Python 代码。生产环境应固定 `revision`。`OpenVLALocalPolicy` 已暴露该参数。

## 判定

| 能力 | 本机 | 依据 |
|---|---|---|
| **OpenVLA-7B 真实推理** | ❌ **不可行** | 见下 |
| LIBERO 仿真 | ✅ **已验证**（2026-08-13 更新） | 审计时判为「有条件」，实际安装后**确认可用**：robosuite 1.4.1 + MuJoCo 3.2.7，CGL 离屏渲染，24 ms/step。代价 2.6 GB，需 6 处版本兼容修复。见 [05](05-libero.md)。 |
| ROS 2 / Gazebo | ❌ 本机不可 | 无 Docker 引擎；macOS 无官方 ROS 2 二进制。见 [03](03-ros2-gazebo.md)。 |
| YOLO / OpenCV | ✅ 可行 | ultralytics 可走 MPS，`yolo11n` 很小。 |
| FastAPI 服务 | ✅ 可行 | 无重依赖。 |
| LoRA 微调 7B | ❌ 本机不可 | 需 ≥1×A100 80G，或 2×4090 QLoRA。 |

### 为什么 OpenVLA-7B 在这台机器上跑不了

四条独立的硬约束，任何一条都足以否决：

1. **内存**：bf16 权重 15.08 GB 必须常驻。统一内存 16 GB，其中系统常驻已占 4–6 GB。必然 OOM。
2. **磁盘**：只剩 28.1 GB。下完权重剩 13 GB，macOS 在低于 ~15 GB 时会开始异常（Spotlight、
   交换文件、更新）。这是**比内存更早触顶**的约束。
3. **量化不可用**：4-bit 走 `bitsandbytes`，该库**只有 CUDA 后端**，MPS 上不可用。
4. **CPU 推理不现实**：7B VLA 每步需数十秒到数分钟。一条 LIBERO rollout 通常 200+ 步。

**不要相信任何声称在这类机器上「流畅运行 7B VLA 推理」的说法。**

### 需要你决定的事（我没有自动执行）

- **未下载** 那 15.08 GB 权重。
- ~~**未安装** LIBERO~~ → **已于 2026-08-13 安装并验证**：`make setup-libero`，实际代价 2.6 GB（wheels 1.9 GB + 仓库克隆 651 MB）。演示数据集（数十 GB）仍未下载，评测不需要。
- **未安装** Docker（需要密码，我不用 `sudo`）。Stage 3 需要它。

## 可行路径

| 目标 | 方案 |
|---|---|
| 现在就看到完整链路 | ✅ 已交付：Stage 1 + Stage 2，降级为 mock policy，全程明确标注 |
| 真实 OpenVLA 动作 | 云 GPU 跑 `rvc.service.vla_server`，笔记本用 `--backend openvla-remote` 驱动。见 [04](04-real-openvla.md) |
| 真实 LIBERO 评测 | ✅ LIBERO 本机已可跑（24 ms/step）；再接一台云 GPU 上的 OpenVLA 即可得到真实评测数字 |
| ROS 2 / Gazebo | 装 Docker Desktop / Colima，或用 Ubuntu 机器。见 [03](03-ros2-gazebo.md) |
| LoRA 微调 | 租 A100 80G；`openvla/openvla-7b-finetuned-libero-*` 是现成的对照检查点 |


## 判定修正记录

| 日期 | 项目 | 原判定 | 实测 |
|---|---|---|---|
| 2026-08-13 | LIBERO 仿真 | ⚠️ 有条件 | ✅ **可用**。robosuite 1.4.1 + MuJoCo 3.2.7 + CGL，建环境 2.3 s，稳态 24 ms/step。需 6 处版本兼容修复（见 [05](05-libero.md)）。 |
| 2026-08-13 | 磁盘 | 28.1 GB 可用 | 装完 LIBERO + torch 后剩 **23.3 GB**。OpenVLA-7B 的 15.08 GB 权重依然放不下（剩 8 GB，低于 macOS 安全线）。 |
| 2026-08-14 | ROS 2 / Gazebo | ❌ 本机不可（无 Docker） | ✅ **已实跑**。colima（用户态，无 sudo）+ vz 虚拟化；镜像 3.86 GB；Gazebo 无头渲染 + ros_gz_bridge + agent 节点端到端 SUCCEEDED。 |
| 2026-08-14 | 磁盘 | — | macOS 更新暂存一夜吃掉 ~18 GB；清 68 GB Xcode DerivedData（可再生）后回到 ~29 GB，警报解除。 |

审计的价值不在于一次判对，而在于把判断依据写下来，之后能被实测推翻或确认。
LIBERO 这一项被推翻了；OpenVLA-7B 本地推理那一项没有 —— 装完 LIBERO 后磁盘反而更紧张了。
