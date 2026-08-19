# 08 · LIBERO 行为克隆基线：VLA 之外的路径

> ✅ 2026-08-19 本机（M3, Apple MPS）实跑。`make bc-data && make bc-train && make bc-eval` 三条命令复现。

## 为什么做这个

这台机器跑不动 OpenVLA-7B，LIBERO 因此一直是"跑得很快的空考场"。行为克隆（BC）是
机器人学习里最经典的基线：**不需要 VLA、不需要 GPU 集群**，50 条人类示教 + 一个小 CNN，
本机十几分钟训完。它回答的问题和 OpenVLA 不同 ——"50 条示教加一个小模型在单个任务上能
走多远？"—— 并且照旧标 `degraded=True`，原因写明"BC 基线，不是 VLA，不看语言"。

它同时把 LIBERO 那半边的管线**真正跑通了**：数据 → 训练 → 通过 agent 运行时（校验器、
日志、产物）评测 → 成功率，一条在真实基准上的端到端学习流水线。

## 数据与约定（全部实测，不靠假设）

| 项 | 实测 |
|---|---|
| 数据 | HF `yifengzhu-hf/LIBERO-datasets` · `libero_spatial` 任务 0 · 50 条示教 · 5,068 步 · 509 MB |
| 图像方向 | hdf5 帧相对 `LiberoEnv` 观测是上下左右翻转的：同一仿真状态下 corr **0.968**（翻转后）vs −0.236（原样）→ 训练时 `[::-1, ::-1]` |
| 动作 | 6 维 OSC 增量 ∈ [−1, 1] 直接用；夹爪 hdf5 `{−1 开, +1 闭}` → 契约 `(1−g)/2` → `{1 开, 0 闭}`（OpenVLA 约定） |
| 本体感知 | hdf5 `ee_pos`(3) + `gripper_states`(2) ≡ env 的 `robot0_eef_pos` + `robot0_gripper_qpos`（数值核对过；ee_pos 有 ~1 cm 重建偏差，归一化后可忽略） |

**这一步逼出了一个契约 bug**：写 `(1−g)/2` 时不得不把约定写死，发现 `types.py` 原来声明的
"OpenVLA 0=开 1=闭"是反的（OpenVLA 源码注释：*0 = close, 1 = open*）。tabletop/mock/视觉伺服
与错的那个符号自洽、`_to_libero` 与对的自洽，互为反号 —— 真 OpenVLA 接上 tabletop 会"说开就闭"。
已修并用 [`tests/test_contract.py`](../tests/test_contract.py) 钉死（见 commit f42128e）。

## 模型与训练

- ResNet18 ×2（agentview、eye-in-hand，ImageNet 初始化）+ 本体感知 MLP → 1088 维 → MLP → 7 维
  （前 6 维 SmoothL1，夹爪 BCE logit；无语言输入 —— 单任务不需要）
- DrQ 式随机平移增强（pad 6 随机裁剪）、AdamW 3e-4、cosine、batch 64、30 epochs
- 45 条训练 / 5 条验证；MPS 上 1,157 s
- 推理 7–15 ms/步（MPS），环境 24 ms/步 —— 合起来仍比真 VLA 的单步前向快一个量级
- 实际 1,157 s 是在与两次 Gazebo 实验并行时测的；单独跑约 32 s/epoch ≈ 16 分钟

## 结果（LIBERO `libero_spatial` 任务 0，官方初始状态 0..49，220 步上限）

| 指标 | 值 |
|---|---|
| **成功率** | **25 / 50 = 50 %** |
| 平均步数（成功回合） | 102（上限 220） |
| 校验器 | 0 次夹取；1 个回合（初始状态 41）被校验器以「夹爪抖动：6 步 4 次开合」拒绝并终止 —— 裸策略在同一状态同样超时失败（全程 11 次开合），所以运行时与裸策略成功率一致；这是校验器第一次在真实学习策略上触发 |

对照：LIBERO 论文里 ResNet-T 单任务 BC 在 spatial 套件的成功率量级为 60–80%（更大模型、
更长训练、GPU）；OpenVLA 官方 finetuned 检查点为 84.7%（整个套件、语言条件）。
**本基线的数字不可与它们直接比较** —— 单任务、小预算、本机 —— 它的意义是把流水线跑通并给出
一个诚实的本地参考点。评测报告 `runs/bc-eval-*/eval.json` 的 `provenance` 字段写明了这一切。

**校验器第一次在真实学习策略上触发。** 初始状态 41 上，BC 策略在第 58–70 步把夹爪
开合成 `1 1 1 0 1 1 0 1 0 0 1 0 1` —— 正是神经网络策略的典型病理。`ActionValidator`
以"6 步内 4 次翻转"拒绝了它，`max_recoveries=0` 下回合终止为 `unsafe_action`。裸策略
（绕过校验器）在同一状态也是 220 步超时失败、全程 11 次翻转 —— 校验器没有"偷走"任何
成功，只是提前 156 步说出了结论。这是这道检查的设计初衷（为 VLA 输出准备的），在一个
本机训练的真策略上得到了第一个真实样本。

**训练曲线**：val loss 第 15 epoch 最低（0.0655），第 30 epoch 0.0846，夹爪准确率 0.978 ——
45 条示教的轻微过拟合。保存的是最终权重而非最优 val（动作 loss 与 rollout 成功率相关性有限，
不值得为此加早停）；想试的话 `make bc-train EPOCHS=15`。

烟雾测试值得一提：只训 **2 个 epoch**（66 s）的检查点已经能在初始状态 0 上成功（87 步）——
单任务 BC 的"形状"很好学，难的是跨初始状态的稳定性，这正是成功率要回答的。

## 文件

- [`src/rvc/policies/bc_libero.py`](../src/rvc/policies/bc_libero.py) — 数据加载（约定转换）、模型、训练、`LiberoBCPolicy`
- [`src/rvc/runners/bc.py`](../src/rvc/runners/bc.py) — `data / train / eval` 子命令
- [`tests/test_bc.py`](../tests/test_bc.py) — 约定转换与策略契约
- 产物：`models/bc-libero-spatial-0.pt`（gitignored）+ `.json` 训练元数据；`runs/bc-eval-*/`（eval.json、前 2 回合 GIF）
