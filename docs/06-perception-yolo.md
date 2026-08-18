# 06 · 学习型感知：YOLO11n on 合成数据

> ✅ 2026-08-14 在本机（M3, Apple MPS）实跑验证。`make yolo` 一条命令复现，约 10 分钟。

## 目的

颜色阈值检测（`ColorDetector`）对教学仿真够用、完全可解释，但真实机器人面对的是有纹理、
光照变化的物体，学习型检测器是标准答案。这一步把 `Detector` 这条接缝**做实**：跑完之后，
Agent 的 PERCEIVE 状态可以由神经网络检测器支撑，其余代码零改动（`--detector yolo`）。

## 方法：仿真器给自己的帧打标签

仿真器知道自己把每个物体画在哪（[`TabletopSim.ground_truth_boxes()`](../src/rvc/envs/tabletop.py)
与 `render()` 用同一套几何），所以**不需要人工标注**：

1. `randomize_layout()` 随机散布方块 / 盒子 / 夹爪，15% 帧带遮挡板（"看不见方块"也在分布内），
   部分帧方块被抬起（更小、带阴影）
2. 写成 YOLO 格式数据集：train 800 / val 100 / test 150
3. 微调 `yolo11n.pt`（ultralytics 8.4，imgsz 256，batch 32，40 epochs，MPS）
4. **用 agent 自己的 `Detector` 接口**在 held-out test 集上评测（IoU ≥ 0.5），并排对比阈值检测器
5. 最优权重复制到 `models/yolo-tabletop.pt`（gitignored，可再生）

这就是"带免费真值的合成数据"—— sim-to-real 流水线里规模化使用的同一个技巧。

## 结果（held-out 合成 test 集，150 帧 / 272 个目标）

| 检测器 | precision | recall | 时延 (MPS) |
|---|---|---|---|
| **YOLO11n 微调** | **0.996** | **0.996** (tp 271 / fp 1 / fn 1) | p50 8.9 ms · p95 9.6 ms |
| 颜色阈值 | 0.967 | 0.963 (tp 262 / fp 9 / fn 10) | < 1 ms |

训练 579 s（40 epochs）。ultralytics 自带验证：mAP50 0.993，mAP50-95 0.967。

端到端：`make demo-libero DETECTOR=yolo INJECT={none,target_lost,grasp_slip}` 三种情形
与颜色检测器行为一致（28 / 40 / 45 步，恢复 0 / 2 / 1 次）—— 遮挡板出现时 YOLO 同样返回
"无方块"，RECOVER 正常触发。

**这不是**对真实世界泛化能力的声明 —— `models/yolo-tabletop.report.json` 的 `provenance`
字段写明了训练与测试都在合成数据上。

## 一个真实 bug：ultralytics 把 numpy 数组当 BGR

第一次评测 40-epoch 模型：precision **0.417**、fp 182 —— 而 ultralytics 自己的验证说
mAP50 0.993。矛盾的根因：**ultralytics 对裸 numpy 数组按 OpenCV 惯例当 BGR 处理**，本项目
全链路是 RGB → 红蓝互换 → 训练充分、真正学会了颜色的模型把红方块认成蓝盒子。更早那个
只训 12 epoch 的模型"看起来正常"（P 0.99）恰恰因为它还没学会颜色、只靠形状大小分类。

修复：`YoloDetector` 把帧包成 PIL Image 再喂模型（PIL 无歧义为 RGB）。
[`tests/test_yolo_detector.py`](../tests/test_yolo_detector.py) 钉住了这个行为。

教训：**评测框架给出的数字要和训练框架自带指标交叉核对；两者矛盾时先怀疑数据管线而不是模型。**

## 文件

- [`src/rvc/perception/yolo_train.py`](../src/rvc/perception/yolo_train.py) — 数据 → 训练 → 评测 → 报告
- [`src/rvc/perception/detector.py`](../src/rvc/perception/detector.py) — `ColorDetector` / `YoloDetector`，同一接口
- [`tests/test_yolo_data.py`](../tests/test_yolo_data.py) — 真值框与渲染一致性、数据集格式（无需 ultralytics）
- [`tests/test_yolo_detector.py`](../tests/test_yolo_detector.py) — RGB 回归（有权重时运行）
