# 10 · SmolVLA：真 VLA 终于在本机跑起来了

> ✅ 2026-08-24 实测：`lerobot/smolvla_libero`（SmolVLA-450M，LIBERO 微调）在
> 16GB M3 的 MPS 上本地推理，通过**完整 agent 运行时**在 LIBERO 官方初始状态上
> 评测：**66% (33/50)**。这是本仓库第一个 `degraded=False` 的模型后端。
> OpenVLA-7B 仍是路线图上的目标动作模型（等 GPU）——SmolVLA 是新增的真 VLA
> 后端，不是替代。

## 为什么是 SmolVLA

审计（docs/00）钉死的硬约束：OpenVLA-7B bf16 权重 15 GB，16GB 统一内存放不下。
SmolVLA-450M 是 Hugging Face 2025 年发布的紧凑 VLA —— SmolVLM2 视觉语言骨干
（裁到 16 层）+ flow-matching 动作专家，论文在 LIBERO 上自报平均 87.3%，
明确面向消费级硬件。450M 参数 ≈ 1 GB 权重，MPS 上单次 50 步 chunk 前向 1–5 s。

关键：它是**货真价实的 VLA**（视觉+语言→动作，语言指令真的进 tokenizer），
所以 `degraded=False` 第一次能诚实地写出来。

## 拓扑：两个 venv，一条 HTTP

lerobot 钉自己的 torch 版本；混进主 venv 会赌上能跑 LIBERO/BC 的工作环境。
所以复用为 OpenVLA 远程推理设计的 serving 拆分 —— 只是"远端"就在本机：

```
.venv (主)                              .venv-lerobot
LIBERO + robosuite + mujoco             lerobot 0.4.4 + torch 2.10
RobotAgent + ActionValidator            SmolVLAPolicy (MPS)
SmolVLARemotePolicy   --HTTP:8100-->    rvc.service.smolvla_server (纯 stdlib)
                      <--7 floats--
```

    make smolvla-serve   # 终端 1：加载模型，~40 s
    make smolvla-eval    # 终端 2：50 回合官方初始状态

当初给云 GPU 设计的架构，第一次真正被用上，服务的却是本机模型 ——
分层的又一次兑现。

## 输入契约：三个必须钉死的约定

给 VLA 喂错分布不会报错，只会默默变蠢。三个约定各有一个测试钉住：

1. **8 维状态布局**：`eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)`，
   与训练集 `lerobot/libero` 的构造一致（robosuite 的 quat2axisangle 约定）。
   为此 `LiberoEnv._wrap` 补暴露了 `ee_quat`。
2. **夹爪符号链**：模型输出 LIBERO 原生（+1 闭 −1 开）→ 客户端转仓库契约
   `(1−g)/2`（OpenVLA 约定 1=开）→ `_to_libero` 转回去。全链恒等，
   在饱和值上无损（test_contract.py）。
3. **相机命名**：`image`(agentview 256²) + `image2`(腕部) → 预处理管道
   rename 成 camera1/2。config.json 里的 `state: [6]` 和 `camera3` 是**陈旧
   元数据** —— 归一化统计 safetensors 里真实是 state(8)、只有两路相机。
   教训重演：约定要向权重文件核对，不要信配置快照。

## 动作分块（chunking）的诚实账本

检查点配置 `chunk_size=50, n_action_steps=50`：一次重前向算 50 步，之后 49 步
从队列弹出。10 Hz 下等于 **5 秒开环** —— 这是该检查点自带的设计，评测时忠实
保留，不做修饰：

| 指标 | 数值 |
|---|---|
| 重前向（整 chunk，MPS） | 首次 ~5.4 s（含 warmup），之后 563 ms p50（168 次实测） |
| 队列弹出步 | 2–3 ms |
| 客户端观测 p50（HTTP 往返摊销） | ~15 ms/步 |

ActionValidator 照常包在外面 —— chunk 里的每一步动作仍逐条过 6 道安全检查，
这正是"运行时与模型无关"的意义。50 回合里校验器实际限幅了 **12 次**真 VLA
的动作 —— 安全层第一次拦到来自真模型（而不是注入故障）的越界输出。

## 失败模式（诚实记录）

17 个失败全部是 280 步超时，无崩溃、无非法动作：模型在难初始状态上抓空后
反复重试直到超时。成功回合平均 ~80 步干净利落（最快 72）。66% 与论文自报
87.3% 的差距来源（不猜测，列可查证的差异）：论文是四套件全均值 + 自有 eval
harness；本检查点是 25k 步的 hub 版本；本协议是单任务 50 初始状态 +
max_steps 280 + 我们的运行时。哪个因素占多少，要换协议对照才能说。

## 结果：三代策略同一协议对比

同一任务（libero_spatial task 0）、同一批官方初始状态、同一个 RobotAgent
e2e 运行时 + 校验器、同一台 M3：

| 策略 | 参数量 | 成功率 | 备注 |
|---|---|---|---|
| BC 基线（ResNet18×2+MLP） | ~23M | 50% (25/50) | 本机训练，无语言 |
| **SmolVLA-450M**（LIBERO 微调） | 450M | **66% (33/50)** | 真 VLA，本机 MPS 推理 |
| OpenVLA-7B（官方微调，文献值） | 7B | 84.7% | libero_spatial 全套件均值，协议不同，仅作量级对照 |

（SmolVLA 论文自报 LIBERO 四套件平均 87.3%，为其自有训练配置下的全套件数字；
本表是单任务 50 初始状态、经 agent 运行时的实测。）

## 复现

```bash
uv venv .venv-lerobot --python 3.11
uv pip install --python .venv-lerobot/bin/python "lerobot[smolvla]"
uv pip install --python .venv-lerobot/bin/python -e .
make smolvla-serve   # 等 "http://127.0.0.1:8100"
make smolvla-eval    # -> runs/smolvla-eval-*/eval.json
```
