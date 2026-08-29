# 11 · 语言条件多任务 BC：一个模型、三个任务、一次诚实的消融

> ✅ 2026-08-29 本机实测：同一个检查点在 LIBERO spatial 任务 0/1/2 上
> **50% / 58% / 42%**（各 50 个官方初始状态），总体 **75/150 = 50%** ——
> 与单任务专家（task-0：25/50 = 50%）持平，但现在是**一个听指令的模型**。
> `make bc-lang-train / bc-lang-eval / bc-lang-ablation` 复现。

## 为什么做这一步

单任务 BC（docs/08）根本不看指令 —— 任务焊死在权重里。VLA 的灵魂机制是
**语言条件**：同一套权重，指令变、行为变。这一步在 450M 的 SmolVLA 和
7B 的 OpenVLA 之下，用最小的模型亲手实现同一机制：

    ResNet18×2 (agentview + wrist) + proprio MLP + 冻结 MiniLM 句向量(384d) → MLP → 7-DoF

三个任务的指令只差空间指代（"between the plate and the ramekin" /
"next to the ramekin" / "from table center"）—— 条件网络要区分的正是
指代表达。

## 机制细节（面试可讲）

- **冻结 MiniLM**（sentence-transformers/all-MiniLM-L6-v2）：mean-pool +
  L2 归一化，训练时逐步查表，推理时未见指令在线编码 —— 没有
  transformers 时**诚实报错**，绝不静默回退成任务 id。
- **向后兼容**：`lang_dim=0` 时网络与旧单任务检查点逐参数同构
  （tests/test_bc_lang.py 钉死），旧检查点照常加载。
- **16GB 统一内存的现实**：三任务图像 1.7GB，超过 1.6GB 阈值自动改
  CPU 驻留、逐批搬运（6MB/批），MPS 只放模型和激活。
- **逐 epoch 落盘**：这台笔记本合盖即睡，一次训练曾以孤儿进程形态
  静默存活三天。现在每个 epoch 覆写检查点并带 `partial` 标记。

## 结果

| 任务（指令差异部分） | 语言条件多任务 | 单任务专家 |
|---|---|---|
| 0 · between the plate and the ramekin | **25/50 = 50%** | 25/50 = 50%（docs/08） |
| 1 · next to the ramekin | **29/50 = 58%** | 未训练 |
| 2 · from table center | **21/50 = 42%** | 未训练 |
| **总体** | **75/150 = 50%** | — |

训练 30 epochs（MPS 约 4.7 分钟/epoch 实测，独占时），推理 12-13 ms/步，
150 回合零安全限幅。在 task-0 上与专家**一分不差**（25/50 vs 25/50）——
多任务没有偷走单任务性能，还白拿了另外两个任务。

## 错误指令消融（这是本步的灵魂）

场景不变，把 task-0 的指令喂给 task-1/2 的评测：

| 任务 | 正确指令 | 错误指令 | Δ |
|---|---|---|---|
| 1 | 58% (29/50) | **40% (8/20)** | −18pp |
| 2 | 42% (21/50) | **35% (7/20)** | −7pp |
| 合计 | 50% | **38% (15/40)** | −12pp |

两个诚实的结论：

1. **语言条件在承重**：方向一致的显著下降 —— 网络没有把 384 维向量
   当噪声忽略（单测也从梯度上钉死了这一点）。
2. **但视觉仍是主导**：错指令下还有 38% 成功率，因为三个任务的碗摆位
   本身就不同 —— agentview 一眼就能分辨任务。这是语言条件 BC 的
   经典现象：**场景可区分时，语言的边际贡献是有限的**。要让语言
   成为必需，需要同场景多目标的任务（roadmap 上的下一步候选）。
   样本量注记：每格 20 回合，±11pp 标准误 —— 方向可信，精确幅度粗糙。

## 谱系表（这个 repo 的策略进化史）

| 策略 | 参数量 | 语言 | LIBERO spatial | 备注 |
|---|---|---|---|---|
| 单任务 BC（docs/08） | 23M×2+MLP | ✗ | task-0: 50% | 任务焊死 |
| **语言条件 BC（本篇）** | +25K 条件支路 | **✓ 冻结 MiniLM** | 3 任务均值 50% | 一模型三任务 |
| SmolVLA-450M（docs/10） | 450M | ✓ VLM | task-0: 66% | 真 VLA，本机 MPS |
| OpenVLA-7B（文献） | 7B | ✓ VLM | 套件均值 84.7% | 待 GPU 实测 |

同一条评测管线、同一个 Agent 运行时、同样的官方初始状态。

## 复现

```bash
make bc-data TASKS=0,1,2      # 或: python -m rvc.runners.bc data --task-index 0,1,2
make bc-lang-train            # ~30 epochs, MPS
make bc-lang-eval             # 3 任务 × 50 回合
make bc-lang-ablation         # 错误指令消融
```

产物：runs/bc-eval-*/eval.json（含逐回合记录与 provenance）、
docs/assets/bc-lang-t{1,2}-success.gif（同一检查点、两个任务的成功回合）。
