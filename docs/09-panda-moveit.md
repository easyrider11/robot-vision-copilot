# 09 · Panda 机械臂：从悬浮夹爪到 7-DOF 真臂

> ✅ 2026-08-20 在本机容器里实跑验证：**完整 pick-and-place 32.9 秒，放置偏差 7px**
> （复跑 17px），零恢复。`make ros-panda` 复现。
> 九阶段序列器与视觉伺服**零改动** —— 换掉的只是执行链，这正是当初分层的意义。

## 最终架构

```
agent_node (九阶段序列器，同悬浮夹爪版)
    │ 世界系速度指令 (10 Hz)
resolved-rate 控制律 (rvc_agent/arm_control.py, ~40 行, 4 个单元测试)
    │ dq = Jᵀ(JJᵀ+λ²I)⁻¹ [v; w_ori]   ← moveit_py 雅可比 + 阻尼最小二乘
    │ 绝对关节位置 (每拍从实测 q 重新出发，无开环积分)
JointGroupPositionController (ros2_control)
    │ gz_ros2_control (position_proportional_gain=60)
Gazebo Panda (moveit_resources URDF + 运行时手术)
```

姿态由控制律中的小角度校正项锁定朝下（吸附面），z 反馈来自 TF（本体感知），
物体/目标位置只来自相机 —— 感知契约与之前完全一致。

## MoveIt Servo 为什么被退役

原计划是 `TwistStamped → MoveIt Servo → 关节速度`。在这套容器（Jazzy 二进制 +
自造动力学的规划 URDF）里，servo 表现出**不可复现的行为**，全部实测记录：

| 实验 | 结果 |
|---|---|
| 手动 link0 系 −z（10 Hz，零时戳） | EE 下降 ✓（0.589→0.527） |
| 同命令再跑一次（另一会话） | EE **上升**（0.591→0.646） |
| link8 系 −z | EE 上升（变换正确性 ✓） |
| agent 发 sim 时钟戳 twist | EE 上升 |
| 零时戳 + 100 Hz 重发 | EE 上升 |
| **POSE 模式，目标冻结在当前位姿** | EE **仍然上升**（对冻结目标的"跟踪"自漂） |

零指令下仍漂移 = 问题不在我们的命令链。与其调一个黑盒，不如用 40 行可单测的
numpy 实现同样的数学（DLS resolved-rate）—— 每拍从实测关节状态重新求解，
**没有任何开环积分**，行为完全确定。servo 的 15 条参数调试经验没有浪费：
全部沉淀在下面的坑清单里。

## 首跑修正全记录（14 个，每个都有实测证据）

| # | 症状 | 根因与修复 |
|---|---|---|
| 1 | launch 秒挂：`KeyError '/**'` | servo 自带 yaml 是裸键值，没有 ros__parameters 包装 |
| 2 | 模型 spawn 失败"至少要有一个 link" | 规划用 URDF 大量 link 无 `<inertial>`，urdf2sdf 全部丢弃 → 给每个缺惯性的 link 注入名义惯量 |
| 3 | `gz_ros2_control-system` 找不到 | ExecuteProcess 不跑 ament env hook → 显式设 `GZ_SIM_SYSTEM_PLUGIN_PATH` `GZ_SIM_RESOURCE_PATH` |
| 4 | DetachableJoint 找不到 `panda_hand` | 手指固定后 sdformat 合并固定关节链 → `preserveFixedJoint` |
| 5 | servo 崩溃 "requires 0 variable values" | SRDF hand 组 group_state 引用被固定的手指关节 → 运行时剥除 |
| 6 | servo 崩溃 `update_period 未初始化` | 自带配置启用的平滑插件参数由 demo launch 提供 → 关平滑 |
| 7 | 卡 INIT，z 永远 None | `rclpy.time.Time` 的 AttributeError 被宽泛 except **静默吞掉** → 显式导入 + 窄化异常 + 限流告警 |
| 8 | 臂趴在工作区上、遮住一切 | 全零关节 = 垂直奇异位形 → `initial_value` 设标准 ready 位形 |
| 9 | 相机里全是臂 | 1.9m 相机下臂身放大 1.8× → 升到 3.4m/512px（放大 1.15×），逐点重标定 k≈200px/m |
| 10 | 绿标记永不可见 | 球体标记被腕部网格吞没；顶视下盘也被 14cm 腕影盖住 → r=9cm 绿盘装腕上方 10cm |
| 11 | j4 精确停在 −2.95、servo 永久拦停 | **spawn 与控制器激活之间关节无人认领，重力把臂拽瘫**，位置控制器激活后把瘫姿锁死 → homing 阶段 + 后来重力整臂关闭 |
| 12 | kp=60 稳态差 0.16 rad；kp=300 臂甩上天 | gz 位置接口是裸 P（默认增益 0.1，参数挂在 `/gz_ros_control` 节点上）；我编的轻惯量+零阻尼在高增益下失稳 → 自洽假动力学（惯量/阻尼/增益联合设计） |
| 13 | 臂整体翻滚漂移 (0.37,−0.57,1.11) | **base 没焊到世界** —— link0 是自由刚体，此前靠重力趴住；重力一关，接触冲量守恒让全臂永久漂移 → URDF `world` link + 固定关节，安装位姿写进关节原点（顺带绕开 spawn 负数参数） |
| 14 | 放置偏 22–55px | 运输截止误差 + 抓取悬挂偏移叠加 → **末段导引换成被携带的方块本身**（对消偏移与视差）→ 7px |

另有两个"遮挡即到达"新变体（旧原理新场景，都先在玩具仿真里写了测试）：
被携带的方块在抓取偏移处露边不算"留在原地"（阈值 30→50px）；
蓝垫被携带物+手完全遮住且误差已小 = 到达，进 LOWER。

## 教学要点（面试可讲）

1. **分层的回报**：换整条执行链（悬浮夹爪→7-DOF 臂 + 逆解），序列器/伺服/状态机零改动。
2. **诚实的假动力学**：规划 URDF 没有真实惯量，选择"运动学木偶"（零重力+阻尼+world 焊接）
   是与悬浮夹爪一脉相承的**显式**简化，而不是调一堆假参数假装动力学正确。
3. **黑盒 vs 40 行白盒**：servo 六次矛盾实验后，用可单测的 DLS 替换 —— 同样的数学，
   确定的行为，每拍闭环于实测状态。
4. **静默吞异常的代价**：坑 #7 让"z=None"伪装成"TF 还没好"，浪费一轮调试。
   except 要窄，吞掉前要打印。

## 复现

```bash
make ros-panda        # 容器内: colcon build + ros2 launch rvc_agent panda.launch.py
```

产物：`runs/panda-run/`（launch.log 相位轨迹、percept.log 检测轨迹、逐帧 PNG）。
