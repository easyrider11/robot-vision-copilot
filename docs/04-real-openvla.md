# 04 · 跑真实 OpenVLA

本机跑不了（原因见 [00](00-environment-audit.md)）。下面三条路都能拿到**真实的** VLA 动作。

## 路径 A：远程推理（推荐，改动最小）

只把 7B 前向搬到云 GPU，整个机器人栈留在笔记本。

```
MacBook（本仓库）                       云 GPU
─────────────────                       ──────
TabletopSim / LIBERO                    rvc.service.vla_server
Agent 状态机           ──HTTP──▶         OpenVLALocalPolicy
校验 / 恢复 / 日志     ◀─action──        openvla/openvla-7b
```

**GPU 机器上：**

```bash
git clone <this repo> && cd robot-vision-copilot
uv venv --python 3.10 && uv pip install -e ".[vla,api]"

# 15.08 GB 权重会在第一次 /load 时下载
RVC_MODEL_ID=openvla/openvla-7b RVC_UNNORM_KEY=bridge_orig \
  uvicorn rvc.service.vla_server:app --host 0.0.0.0 --port 8000
```

先确认能加载再跑：

```bash
curl localhost:8000/health          # can_load / reason
curl -X POST localhost:8000/load    # 预加载，避免第一步等两分钟
```

**笔记本上（走 SSH 隧道，不要裸暴露）：**

```bash
ssh -N -L 8000:localhost:8000 user@gpu-host &
make demo-libero BACKEND=openvla-remote RVC_VLA_URL=http://127.0.0.1:8000
```

这时终端横幅会变成 `✓ 真实 VLA 推理 (not degraded)`，`summary.json` 里 `degraded: false`。

**延迟预期**：每个控制步一次图像往返。同区域 A100 上 OpenVLA 单步 ~150–400 ms，
加网络往返，一条 200 步 rollout 大约 1–3 分钟。客户端会记录每次的 `latency_ms`。

### 硬件与费用（按 2026 年常见价位，仅供估算，请自行核实）

| GPU | VRAM | 能否 bf16 | 备注 |
|---|---|---|---|
| A100 80G | 80 GB | ✅ 宽裕 | 也能做 LoRA |
| A100 40G | 40 GB | ✅ | 推理够 |
| L40S | 48 GB | ✅ | 性价比好 |
| RTX 4090 | 24 GB | ✅ 紧张 | 建议 `RVC_4BIT=1` |
| RTX 3090 | 24 GB | ⚠️ | 需 4-bit |
| T4 / V100 16G | 16 GB | ❌ | 不够 |

## 路径 B：整体迁到 Linux + NVIDIA

```bash
uv venv --python 3.10 && uv pip install -e ".[vla]"
make demo-libero BACKEND=openvla-local
```

`OpenVLALocalPolicy` 的 `probe()` 会在加载前检查 CUDA、VRAM、磁盘，不满足就给出具体原因，
不会跑到一半 OOM。

## 检查点与 `unnorm_key` 的配对

**这是最常见的「机械臂在抖但不动」的原因。** 反归一化统计量是烧在检查点里的，
要一一对应：

| 检查点 | `unnorm_key` |
|---|---|
| `openvla/openvla-7b` | `bridge_orig` |
| `openvla/openvla-7b-finetuned-libero-spatial` | `libero_spatial` |
| `openvla/openvla-7b-finetuned-libero-object` | `libero_object` |
| `openvla/openvla-7b-finetuned-libero-goal` | `libero_goal` |
| `openvla/openvla-7b-finetuned-libero-10` | `libero_10` |

跑 LIBERO 评测就该用对应的 finetuned 检查点，别用 base + `bridge_orig`。

```bash
make demo-libero BACKEND=openvla-remote ENV=libero \
  RVC_VLA_URL=http://127.0.0.1:8000
# 服务端: RVC_MODEL_ID=openvla/openvla-7b-finetuned-libero-spatial RVC_UNNORM_KEY=libero_spatial
```

## Prompt 模板不能改

```python
"In: What action should the robot take to {instruction}?\nOut:"
```

一字不差地复现在 [`openvla_local.py`](../src/rvc/policies/openvla_local.py)。改措辞会让模型偏离训练分布。
指令统一小写、去掉句末句号。

## `trust_remote_code` 是一个真实的信任决定

OpenVLA 带自定义建模代码，加载必须 `trust_remote_code=True`，也就是执行从 Hub 下载的 Python。
这是上游文档的路径，但它确实是在你的机器上跑别人的代码。生产环境请固定 `revision`：

```python
OpenVLALocalPolicy(model_id="openvla/openvla-7b", revision="<commit-sha>")
```

`revision` 参数已经暴露出来了，就是为了这件事。

## flash-attn

上游把 `flash-attn` 当硬依赖，但它编译很慢且经常失败。
`OpenVLALocalPolicy` 会尝试 import，失败就回落到 `sdpa`：慢一些，数值上没问题。

## 路径 C：LoRA 微调（更后面的阶段）

**先把评测跑通再谈微调。** 有了 `openvla/openvla-7b-finetuned-libero-*` 作为对照，
你才知道自己微调出来的东西是好是坏。

硬件底线：

| 方式 | VRAM | 说明 |
|---|---|---|
| 全量微调 | 8×A100 80G | 上游论文的配置 |
| LoRA (r=32) | 1×A100 80G | 上游推荐 |
| QLoRA 4-bit | 1×A100 40G 或 2×4090 | 更慢 |

本机（16 GB 统一内存、无 CUDA）**完全不可能**。

大致步骤：

1. 用 `rvc.service.vla_server` 跑通 LIBERO 评测，拿到 baseline 成功率
2. 采集或选取数据集，转成 RLDS 格式
3. 按上游 `vla-scripts/finetune.py` 跑 LoRA
4. 合并 LoRA 权重，写入 `dataset_statistics.json`（否则 `unnorm_key` 找不到）
5. 用同一套 `make demo-libero BACKEND=openvla-remote ENV=libero` 对比 baseline

第 4 步最容易漏 —— 微调后没写统计量，加载时 `unnorm_key` 会报错或静默用错缩放。

## 怎么确认自己真的没在跑降级演示

三个地方同时说了算，任何一个显示降级就是降级：

```bash
# 1. 终端横幅
make demo-libero BACKEND=openvla-remote     # 应显示 "✓ 真实 VLA 推理 (not degraded)"

# 2. summary.json
jq '.degraded, .backend' runs/*/summary.json | tail -2   # false, "openvla-remote"

# 3. HTTP
curl -s localhost:8080/health | jq '.model_loaded'       # true
```

或者干脆让它不许降级：

```bash
make demo-real BACKEND=openvla-remote       # 拿不到真实 VLA 直接非零退出
```
