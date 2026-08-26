# MoveIt Servo POSE-mode drift: does NOT reproduce on stock

Date: 2026-08-26

## Setup — stock only, no custom anything

- Image: `ros:jazzy-ros-base` + `ros-jazzy-moveit-servo` **2.12.4-1noble.20260617.160856**
  (`ros-jazzy-moveit-core` 2.12.4-1noble.20260617.143031)
- Robot: `moveit_resources_panda_moveit_config` (upstream URDF, upstream inertias)
- Plant: **ros2_control FakeSystem** — no Gazebo, no physics engine, no gravity
- Config: stock `moveit_servo/config/panda_simulated_config.yaml`, unmodified
- Launch: stock `demo_ros_api.launch.py` with rviz removed and servo standalone
- Host: macOS arm64 / colima
- EE start pose: panda_link0 -> panda_link8 = (0.307020, -0.000000, 0.590270)

## Results — 20 s per condition, EE sampled from TF

| # | Condition | dz | total displacement | verdict |
|---|---|---|---|---|
| A | baseline: servo up, nothing published | +0.000000 | 0.000000 m | held still |
| B | TWIST mode, zero TwistStamped @ ~50 Hz, valid stamps | +0.000000 | 0.000000 m | held still |
| C | **POSE mode, PoseStamped frozen at the t=0 EE pose** | **+0.000000** | **0.000000 m** | **held still** |
| D | TWIST mode, all-zero twist, **zero timestamp**, ~100 Hz resend | +0.000000 | 0.000000 m | held still |

## Positive control — the harness is not silently inert

Identical code path, target deliberately offset by -0.05 m in z:

```
t=0   EE z = 0.590270      target z = 0.540270
t=1.0s  EE z = +0.540270   moved = -0.050000 m
... held at -0.050000 for the remaining 14 s
servo status codes seen: [(0, 'No warnings')]
```

Servo converged to the commanded target in under 1 s, hit it exactly, then held
it dead still. So conditions A-D are genuine null results, not a broken test.

## Conclusion

The behaviour recorded in `robot-vision-copilot/docs/09-panda-moveit.md` — POSE
mode self-drifting against a target frozen at the current pose — **does not
reproduce against stock MoveIt Servo on the stock Panda with FakeSystem
hardware.** Servo's pose tracking is well behaved here: it converges exactly and
holds with zero steady-state error and no oscillation.

Therefore the drift was **not a servo core defect**. It came from the local
environment: the Gazebo plant, the fabricated inertias injected into the
planning URDF, the `gz_ros2_control` position interface (a bare P velocity
controller), or their interaction.

**No upstream issue should be filed for this.** Holding it back pending this
test was the right call.

## Hypothesis for what actually happened (NOT verified — do not report as fact)

Under FakeSystem the commanded joint positions are echoed back as the measured
state, so tracking error is identically zero. Under Gazebo with a P velocity
controller and hand-made dynamics, the measured state lags the command
persistently. Servo's pose tracking closes on the *measured* state, and the
`demo_pose` pattern re-derives the target from that state every cycle, so a
persistent lag can integrate into monotonic drift. Testing this would require
reproducing on Gazebo + gz_ros2_control, which is a different experiment.

## Reproduce

```
docker build -t servo-repro:jazzy .
docker run -d --name servo-repro-run -e ROS_DOMAIN_ID=77 servo-repro:jazzy bash -lc 'sleep infinity'
docker cp repro.launch.py servo-repro-run:/repro/ ; docker cp frozen_target.py servo-repro-run:/repro/
docker exec -d servo-repro-run bash -lc 'source /opt/ros/jazzy/setup.bash; cd /repro && ros2 launch ./repro.launch.py > launch.log 2>&1'
docker exec servo-repro-run bash -lc 'source /opt/ros/jazzy/setup.bash; python3 /repro/frozen_target.py pose 20'
docker exec servo-repro-run bash -lc 'source /opt/ros/jazzy/setup.bash; python3 /repro/positive_control.py'
```
