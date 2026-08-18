"""Grab frames from a ROS image topic to PNG files - the Stage 3 'eyes'.

Runs INSIDE the container, writes into /ws/runs (host-mounted), so the person
on the host can see what Gazebo is rendering without any GUI:

    python3 /ws/src/rvc_agent/rvc_agent/frame_grab.py \
        --topic /camera/image_raw --out /ws/runs/stage3 --count 20 --every 0.5

Also prints per-frame ColorDetector results, which is exactly what the agent's
PERCEIVE state sees - if labels are missing here, the thresholds need tuning
against Gazebo's lighting, and that is a real calibration finding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image

from rvc.perception.detector import ColorDetector


class FrameGrab(Node):
    def __init__(self, topic: str, out: Path, count: int, every: float) -> None:
        super().__init__("frame_grab")
        self.bridge = CvBridge()
        self.detector = ColorDetector()
        self.out = out
        self.count = count
        self.every = every
        self.saved = 0
        self.last_save = 0.0
        out.mkdir(parents=True, exist_ok=True)
        self.create_subscription(Image, topic, self._on_image, 5)

    def _on_image(self, msg: Image) -> None:
        now = time.monotonic()
        if now - self.last_save < self.every:
            return
        self.last_save = now
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        path = self.out / f"grab_{self.saved:03d}.png"
        PILImage.fromarray(np.asarray(img)).save(path)

        dets = self.detector.detect(
            np.asarray(img), ("red_block", "blue_box", "gripper_marker")
        )
        found = {d.label: (round(d.center_px[0]), round(d.center_px[1])) for d in dets}
        print(f"[{self.saved:03d}] {path.name}  {img.shape}  detections={found}", flush=True)
        self.saved += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/image_raw")
    ap.add_argument("--out", default="/ws/runs/stage3")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    rclpy.init()
    node = FrameGrab(args.topic, Path(args.out), args.count, args.every)
    deadline = time.monotonic() + args.timeout
    while node.saved < args.count and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    ok = node.saved > 0
    print(f"saved {node.saved} frame(s) -> {args.out}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
