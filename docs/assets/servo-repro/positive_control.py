#!/usr/bin/env python3
"""Positive control: does the SAME harness move the arm when the target is offset?

If this moves the EE, the frozen-target null result is real.
If this does not move it, the frozen-target test proved nothing.
"""
import time
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE, TIP, POSE = "panda_link0", "panda_link8", 2

rclpy.init()
n = Node("positive_control")
buf = Buffer(); TransformListener(buf, n)
pub = n.create_publisher(PoseStamped, "/servo_node/pose_target_cmds", 10)
statuses = []
n.create_subscription(ServoStatus, "/servo_node/status", lambda m: statuses.append((m.code, m.message)), 10)
cli = n.create_client(ServoCommandType, "/servo_node/switch_command_type")

def ee():
    for _ in range(100):
        rclpy.spin_once(n, timeout_sec=0.1)
        try:
            return buf.lookup_transform(BASE, TIP, rclpy.time.Time()).transform
        except Exception:
            pass
    raise RuntimeError("no TF")

t0 = ee(); p0 = t0.translation
print(f"t=0   EE z = {p0.z:.6f}")

cli.wait_for_service(timeout_sec=10.0)
req = ServoCommandType.Request(); req.command_type = POSE
f = cli.call_async(req); rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
print(f"switch to POSE -> {f.result().success}")

tgt = PoseStamped()
tgt.header.frame_id = BASE
tgt.pose.position.x = p0.x
tgt.pose.position.y = p0.y
tgt.pose.position.z = p0.z - 0.05      # <-- 5 cm DOWN, deliberately offset
tgt.pose.orientation = t0.rotation
print(f"target z = {tgt.pose.position.z:.6f}  (offset -0.05 m)")

start = time.time(); nxt = 0.0
while time.time() - start < 15.0:
    tgt.header.stamp = n.get_clock().now().to_msg()
    pub.publish(tgt)
    rclpy.spin_once(n, timeout_sec=0.01)
    el = time.time() - start
    if el >= nxt:
        z = ee().translation.z
        print(f"  t={el:5.1f}s  EE z={z:+.6f}  moved={z-p0.z:+.6f} m", flush=True)
        nxt = el + 1.0
    time.sleep(0.02)

zf = ee().translation.z
print(f"\ntotal movement: {zf - p0.z:+.6f} m  (commanded -0.050000)")
print(f"HARNESS: {'WORKS - servo acted on the command' if abs(zf-p0.z) > 1e-4 else 'BROKEN - servo never acted'}")
uniq = sorted(set(statuses))
print(f"servo status codes seen: {uniq if uniq else 'NONE (no status published!)'}")
rclpy.shutdown()
