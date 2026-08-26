#!/usr/bin/env python3
"""Does MoveIt Servo hold still when told to hold still?

Three conditions against the stock panda demo (ros2_control FakeSystem, no
physics engine, no gravity, no custom URDF):

  baseline : servo left in its default state, nothing published
  twist    : TWIST mode, zero TwistStamped published continuously
  pose     : POSE mode, a PoseStamped frozen at the EE pose measured at t=0

In all three the end effector should not move. Reports EE translation from the
t=0 sample over the run.
"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE, TIP = "panda_link0", "panda_link8"
JOINT_JOG, TWIST, POSE = 0, 1, 2


class Probe(Node):
    def __init__(self, condition, duration):
        super().__init__("servo_drift_probe")
        self.condition = condition
        self.duration = duration
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.pose_pub = self.create_publisher(PoseStamped, "/servo_node/pose_target_cmds", 10)
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.switch = self.create_client(ServoCommandType, "/servo_node/switch_command_type")

    def ee(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                t = self.buf.lookup_transform(BASE, TIP, rclpy.time.Time()).transform.translation
                return (t.x, t.y, t.z)
            except Exception:
                continue
        raise RuntimeError(f"no TF {BASE}->{TIP} within {timeout}s")

    def set_mode(self, mode):
        if not self.switch.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("switch_command_type service never appeared")
        req = ServoCommandType.Request()
        req.command_type = mode
        fut = self.switch.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = fut.result() is not None and fut.result().success
        self.get_logger().info(f"switch_command_type({mode}) -> {ok}")
        return ok

    def run(self):
        p0 = self.ee()
        self.get_logger().info(f"t=0  EE = ({p0[0]:.6f}, {p0[1]:.6f}, {p0[2]:.6f})")

        frozen = None
        if self.condition == "pose":
            self.set_mode(POSE)
            frozen = PoseStamped()
            frozen.header.frame_id = BASE
            frozen.pose.position.x, frozen.pose.position.y, frozen.pose.position.z = p0
            # orientation measured at t=0 too
            tf = self.buf.lookup_transform(BASE, TIP, rclpy.time.Time()).transform.rotation
            frozen.pose.orientation = tf
        elif self.condition == "twist":
            self.set_mode(TWIST)

        rows, start, next_sample = [], time.time(), 0.0
        while time.time() - start < self.duration:
            now = time.time() - start
            if self.condition == "pose":
                frozen.header.stamp = self.get_clock().now().to_msg()
                self.pose_pub.publish(frozen)
            elif self.condition == "twist":
                tw = TwistStamped()
                tw.header.frame_id = BASE
                tw.header.stamp = self.get_clock().now().to_msg()
                self.twist_pub.publish(tw)   # all zeros
            rclpy.spin_once(self, timeout_sec=0.01)
            if now >= next_sample:
                try:
                    p = self.ee(timeout=0.5)
                    d = ((p[0]-p0[0])**2 + (p[1]-p0[1])**2 + (p[2]-p0[2])**2) ** 0.5
                    rows.append((now, p, d))
                    print(f"  t={now:5.1f}s  EE=({p[0]:+.6f},{p[1]:+.6f},{p[2]:+.6f})  |d|={d:.6f} m", flush=True)
                except Exception as e:
                    print(f"  t={now:5.1f}s  TF lookup failed: {e}", flush=True)
                next_sample = now + 1.0
            time.sleep(0.02)

        if rows:
            _, plast, dlast = rows[-1]
            print("\n=== RESULT ===", flush=True)
            print(f"condition       : {self.condition}", flush=True)
            print(f"start EE        : ({p0[0]:+.6f}, {p0[1]:+.6f}, {p0[2]:+.6f})", flush=True)
            print(f"end   EE        : ({plast[0]:+.6f}, {plast[1]:+.6f}, {plast[2]:+.6f})", flush=True)
            print(f"dz              : {plast[2]-p0[2]:+.6f} m", flush=True)
            print(f"total displace  : {dlast:.6f} m over {self.duration:.0f}s", flush=True)
            print(f"VERDICT         : {'DRIFT' if dlast > 1e-4 else 'held still'}", flush=True)


def main():
    condition = sys.argv[1] if len(sys.argv) > 1 else "pose"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    rclpy.init()
    node = Probe(condition, duration)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
