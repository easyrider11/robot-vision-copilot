"""Synthetic-dataset generator tests (no ultralytics needed).

The learned detector is only as good as its labels, and the labels come from
`TabletopSim.ground_truth_boxes()`. These tests pin that the geometry-derived
boxes agree with what is actually rendered, and that the YOLO-format writer
produces well-formed, in-range label files.
"""

from __future__ import annotations

import numpy as np

from rvc.envs.tabletop import IMG_SIZE, TabletopSim
from rvc.perception.detector import ColorDetector
from rvc.perception.yolo_train import CLASSES, _iou, build_dataset


def test_ground_truth_boxes_match_rendered_pixels():
    env, rng, det = TabletopSim(seed=0), np.random.default_rng(0), ColorDetector()
    ious = []
    for _ in range(40):
        env.randomize_layout(rng)
        env._occlusion_window = (-1, -1)
        env.t = 0
        img = env.render()
        gt = env.ground_truth_boxes()
        assert set(gt) == {"red_block", "blue_box"}
        for d in det.detect(img, ("red_block", "blue_box")):
            ious.append(_iou(d.bbox_px, gt[d.label]))
    assert len(ious) >= 70, "detector should see nearly every rendered object"
    assert min(ious) > 0.7 and float(np.mean(ious)) > 0.9


def test_occluded_frames_have_no_block_label():
    env, rng = TabletopSim(seed=1), np.random.default_rng(1)
    env.randomize_layout(rng)
    env._occlusion_window = (0, 1)
    env.t = 0
    assert "red_block" not in env.ground_truth_boxes()
    assert "blue_box" in env.ground_truth_boxes()


def test_yolo_dataset_writer_is_well_formed(tmp_path):
    yaml = build_dataset(tmp_path / "d", n_train=6, n_val=2, n_test=2)
    assert yaml.exists() and "names:" in yaml.read_text()
    for split, n in (("train", 6), ("val", 2), ("test", 2)):
        imgs = sorted((tmp_path / "d" / "images" / split).glob("*.png"))
        lbls = sorted((tmp_path / "d" / "labels" / split).glob("*.txt"))
        assert len(imgs) == len(lbls) == n
        for lbl in lbls:
            for line in lbl.read_text().splitlines():
                c, *xywh = line.split()
                assert 0 <= int(c) < len(CLASSES)
                vals = [float(v) for v in xywh]
                assert len(vals) == 4 and all(0.0 <= v <= 1.0 for v in vals)
                # boxes should be a plausible size for a 256px frame
                assert 8 / IMG_SIZE < vals[2] < 0.3
