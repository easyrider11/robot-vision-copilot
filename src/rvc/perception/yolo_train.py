"""`make yolo` - synthetic dataset -> fine-tune yolo11n -> evaluate -> models/yolo-tabletop.pt

WHY THIS EXISTS
---------------
The colour-threshold detector is honest and sufficient for the teaching sims,
but a real robot sees textured objects under changing light, and a learned
detector is the standard answer. This script shows the full loop end to end
on a laptop, with no manual labelling: the simulator knows exactly where it
drew every object (`TabletopSim.ground_truth_boxes()`), so it labels its own
frames. That is "synthetic data with free ground truth" - the same trick used
at scale in sim-to-real pipelines.

WHAT IT DOES
------------
    1. render N random tabletop layouts (block/box/gripper scattered, some
       occluded, some with the block lifted) -> YOLO-format dataset
    2. fine-tune ultralytics `yolo11n.pt` on it (Apple MPS if available)
    3. evaluate on a held-out split with the SAME `Detection` interface the
       agent uses (`YoloDetector`), reporting precision / recall at IoU 0.5
    4. copy best weights to models/yolo-tabletop.pt for `--detector yolo`

WHAT IT IS NOT
--------------
It is not a claim that this detector generalises beyond the sim - the report
JSON says exactly what it was trained and tested on. Its purpose is to make the
`Detector` seam real: after this runs, the agent's PERCEIVE state can be backed
by a neural detector instead of thresholds, with zero changes elsewhere.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rvc.envs.tabletop import IMG_SIZE, TabletopSim

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSES = ["red_block", "blue_box"]


# --- 1. dataset -------------------------------------------------------------


def _write_split(root: Path, split: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    env = TabletopSim(seed=seed)
    for i in range(n):
        env.randomize_layout(rng)
        # a slice of frames get the occluder, so "no block visible" is in-distribution
        env._occlusion_window = (0, 1) if rng.random() < 0.15 else (-1, -1)
        env.t = 0
        img = env.render()
        Image.fromarray(img).save(root / "images" / split / f"{i:05d}.png")
        lines = []
        for label, (x0, y0, x1, y1) in env.ground_truth_boxes().items():
            cx = (x0 + x1) / 2 / IMG_SIZE
            cy = (y0 + y1) / 2 / IMG_SIZE
            w = (x1 - x0) / IMG_SIZE
            h = (y1 - y0) / IMG_SIZE
            lines.append(f"{CLASSES.index(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (root / "labels" / split / f"{i:05d}.txt").write_text("\n".join(lines))


def build_dataset(root: Path, n_train: int, n_val: int, n_test: int) -> Path:
    if root.exists():
        shutil.rmtree(root)
    _write_split(root, "train", n_train, seed=1)
    _write_split(root, "val", n_val, seed=2)
    _write_split(root, "test", n_test, seed=3)
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    return yaml


# --- 3. evaluation through the agent's own interface -------------------------


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(weights: Path, root: Path, split: str = "test") -> dict:
    from rvc.perception.detector import ColorDetector, YoloDetector

    yolo = YoloDetector(str(weights), conf=0.25)
    color = ColorDetector()
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("yolo", "color")}
    lat = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        img = np.asarray(Image.open(images_dir / f"{lbl.stem}.png").convert("RGB"))
        gt = []
        for line in lbl.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) * IMG_SIZE for v in (cx, cy, w, h))
            gt.append((CLASSES[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        for name, det in (("yolo", yolo), ("color", color)):
            t0 = time.perf_counter()
            preds = det.detect(img, tuple(CLASSES))
            if name == "yolo":
                lat.append((time.perf_counter() - t0) * 1000)
            matched = set()
            for p in preds:
                hit = None
                for j, (glabel, gbox) in enumerate(gt):
                    if j in matched or glabel != p.label:
                        continue
                    if _iou(p.bbox_px, gbox) >= 0.5:
                        hit = j
                        break
                if hit is None:
                    stats[name]["fp"] += 1
                else:
                    matched.add(hit)
                    stats[name]["tp"] += 1
            stats[name]["fn"] += len(gt) - len(matched)
    out = {}
    for name, s in stats.items():
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        out[name] = {**s, "precision": round(p, 4), "recall": round(r, 4)}
    out["yolo"]["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 2)
    out["yolo"]["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 2)
    return out


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "runs" / "yolo-data"))
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "yolo-tabletop.pt"))
    ap.add_argument("--skip-train", action="store_true", help="only rebuild data + evaluate")
    args = ap.parse_args(argv)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"✗ ultralytics/torch not installed: {exc}\n  uv pip install -e '.[vision]'")
        return 1

    root = Path(args.data_dir)
    print(f"[1/4] 生成合成数据集 -> {root}  (train={args.train} val={args.val} test={args.test})")
    yaml = build_dataset(root, args.train, args.val, args.test)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    if not args.skip_train:
        print(f"[2/4] 微调 yolo11n  device={device} epochs={args.epochs}")
        t0 = time.time()
        model = YOLO("yolo11n.pt")
        res = model.train(
            data=str(yaml), epochs=args.epochs, imgsz=IMG_SIZE, batch=32, device=device,
            workers=0, plots=False, verbose=False, project=str(root / "train"), name="run",
            exist_ok=True, seed=0, deterministic=True,
        )
        train_s = time.time() - t0
        best = Path(res.save_dir) / "weights" / "best.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out)
        print(f"      训练 {train_s:.0f}s -> {out}")
    else:
        train_s = 0.0
        if not out.exists():
            print(f"✗ {out} 不存在，去掉 --skip-train")
            return 1

    print("[3/4] 用 agent 的 Detector 接口在 held-out test 集上评测（IoU≥0.5）")
    ev = evaluate(out, root, "test")
    for name in ("yolo", "color"):
        s = ev[name]
        print(f"      {name:<6} precision={s['precision']:.3f} recall={s['recall']:.3f} "
              f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    p50, p95 = ev["yolo"]["latency_ms_p50"], ev["yolo"]["latency_ms_p95"]
    print(f"      yolo latency p50={p50}ms p95={p95}ms on {device}")

    report = {
        "provenance": (
            "yolo11n fine-tuned on SYNTHETIC TabletopSim renders auto-labelled from "
            "simulator geometry; evaluated on a held-out synthetic split. Not a claim "
            "about real-world generalisation."
        ),
        "classes": CLASSES, "device": device, "epochs": args.epochs,
        "n_train": args.train, "n_val": args.val, "n_test": args.test,
        "train_seconds": round(train_s, 1), "weights": str(out), "eval": ev,
    }
    rp = out.parent / "yolo-tabletop.report.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"[4/4] 报告 -> {rp}\n      现在可以: make demo-libero DETECTOR=yolo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
