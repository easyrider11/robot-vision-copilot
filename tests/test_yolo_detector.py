"""Regression test for the learned detector's colour handling.

Skips unless ultralytics is installed AND `make yolo` has produced weights.
Pins the bug that made a well-trained model look broken: ultralytics reads a
bare numpy array as BGR, this project is RGB end to end - a channel swap turns
the red block blue and the blue box red. YoloDetector must feed the model RGB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rvc.envs.tabletop import TabletopSim
from rvc.perception.yolo_train import _iou

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "yolo-tabletop.pt"
pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="run `make yolo` first")


def test_yolo_detector_gets_colours_right_on_rgb_frames():
    from rvc.perception.detector import YoloDetector

    det = YoloDetector(str(WEIGHTS), conf=0.25)
    env, rng = TabletopSim(seed=7), np.random.default_rng(7)
    hits, total = 0, 0
    for _ in range(12):
        env.randomize_layout(rng)
        env._occlusion_window = (-1, -1)
        env.t = 0
        gt = env.ground_truth_boxes()
        preds = {d.label: d for d in det.detect(env.render(), ("red_block", "blue_box"))}
        for label, box in gt.items():
            total += 1
            if label in preds and _iou(preds[label].bbox_px, box) >= 0.5:
                hits += 1
    assert hits / total >= 0.9, f"only {hits}/{total} objects found with the right class"
