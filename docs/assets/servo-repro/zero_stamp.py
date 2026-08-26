#!/usr/bin/env python3
"""Their table had 'zero timestamp + 100 Hz resend -> EE rose'. Test on stock."""
import time
import rclpy
from geometry_msgs.msg import TwistStamped
from builtin_interfaces.msg import Time
from moveit_msgs.srv import ServoCommandType
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE, TIP, TWIST = "panda_link0", "panda_link8", 1
rclpy.init(); n = Node("zero_stamp"); buf = Buffer(); TransformListener(buf, n)
pub = n.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
cli = n.create_client(ServoCommandType, "/servo_node/switch_command_type")

def ee():
    for _ in range(100):
        rclpy.spin_once(n, timeout_sec=0.1)
        try: return buf.lookup_transform(BASE, TIP, rclpy.time.Time()).transform.translation
        except Exception: pass
    raise RuntimeError("no TF")

p0 = ee(); print(f"t=0  EE z = {p0.z:.6f}")
cli.wait_for_service(timeout_sec=10.0)
r = ServoCommandType.Request(); r.command_type = TWIST
f = cli.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
print(f"switch to TWIST -> {f.result().success}")

tw = TwistStamped(); tw.header.frame_id = BASE
tw.header.stamp = Time(sec=0, nanosec=0)      # <-- ZERO timestamp, never updated
start, nxt = time.time(), 0.0
while time.time() - start < 20.0:
    pub.publish(tw)                            # all-zero twist, zero stamp, ~100 Hz
    rclpy.spin_once(n, timeout_sec=0.001)
    el = time.time() - start
    if el >= nxt:
        z = ee().z; print(f"  t={el:5.1f}s  EE z={z:+.6f}  dz={z-p0.z:+.6f}", flush=True); nxt = el + 2.0
    time.sleep(0.01)
zf = ee().z
print(f"\ndz over 20s: {zf-p0.z:+.6f} m")
print(f"VERDICT: {'DRIFT' if abs(zf-p0.z) > 1e-4 else 'held still'}")
rclpy.shutdown()
