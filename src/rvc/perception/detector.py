"""Perception layer.

Stage 1 ships a pure-numpy colour detector so the PERCEIVE state does real work
on real pixels: when the occluder slides over the block, the detector genuinely
returns nothing and the agent genuinely enters RECOVER. Nothing here reads
simulator state.

Stage 3 swaps this for YOLO behind the same `Detector` interface - see
`YoloDetector` at the bottom, which is wired but not installed by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from rvc.envs.tabletop import IMG_SIZE, X_MAX, X_MIN, Y_MAX, Y_MIN


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1
    center_px: tuple[float, float]
    center_world: tuple[float, float]
    area_px: int


def _px_to_world(u: float, v: float) -> tuple[float, float]:
    """Inverse of `rvc.envs.tabletop._to_px` (top-down orthographic camera)."""
    x = u / IMG_SIZE * (X_MAX - X_MIN) + X_MIN
    y = Y_MAX - v / IMG_SIZE * (Y_MAX - Y_MIN)
    return float(x), float(y)


class ColorDetector:
    """Threshold-in-RGB detector. Deliberately simple and fully inspectable."""

    name = "color-threshold"

    #: label -> (predicate on r,g,b float arrays)
    SPECS = {
        "red_block": lambda r, g, b: (r > 130) & (r - g > 55) & (r - b > 55),
        "blue_box": lambda r, g, b: (b > 90) & (b - r > 30) & (b - g > 15),
        # Stage 3: the Gazebo floating gripper carries a green top face so the
        # visual-servo loop can localize it from the overhead camera alone.
        "gripper_marker": lambda r, g, b: (g > 130) & (g - r > 55) & (g - b > 55),
    }

    def __init__(self, min_area: int = 60) -> None:
        self.min_area = min_area

    def detect(
        self, image: np.ndarray, labels: tuple[str, ...] = ("red_block",)
    ) -> list[Detection]:
        img = np.asarray(image)
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)

        out: list[Detection] = []
        for label in labels:
            spec = self.SPECS.get(label)
            if spec is None:
                continue
            mask = spec(r, g, b)
            area = int(mask.sum())
            if area < self.min_area:
                continue
            ys, xs = np.nonzero(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cu, cv = float(xs.mean()), float(ys.mean())
            # Confidence: how solidly the mask fills its own bounding box.
            fill = area / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            out.append(
                Detection(
                    label=label,
                    confidence=round(min(0.99, 0.45 + 0.55 * fill), 3),
                    bbox_px=(x0, y0, x1, y1),
                    center_px=(cu, cv),
                    center_world=_px_to_world(cu, cv),
                    area_px=area,
                )
            )
        return out

    def find(self, image: np.ndarray, label: str) -> Detection | None:
        d = self.detect(image, (label,))
        return d[0] if d else None


def draw_overlay(
    image: np.ndarray,
    detections: list[Detection],
    header: str = "",
    footer: str = "",
) -> np.ndarray:
    """Annotate a frame with detection boxes + state text (saved to runs/)."""
    im = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    pad_top, pad_bot = 20, 18
    canvas = Image.new("RGB", (im.width, im.height + pad_top + pad_bot), (18, 20, 25))
    canvas.paste(im, (0, pad_top))
    d = ImageDraw.Draw(canvas)

    colors = {"red_block": (255, 210, 90), "blue_box": (120, 235, 190)}
    for det in detections:
        x0, y0, x1, y1 = det.bbox_px
        c = colors.get(det.label, (255, 255, 255))
        d.rectangle([x0 - 1, y0 - 1 + pad_top, x1 + 1, y1 + 1 + pad_top], outline=c, width=2)
        d.text((x0, max(0, y0 - 10) + pad_top), f"{det.label} {det.confidence:.2f}", fill=c)

    if header:
        d.text((6, 5), header[:64], fill=(235, 238, 245))
    if footer:
        d.text((6, im.height + pad_top + 4), footer[:72], fill=(160, 170, 185))
    return np.asarray(canvas, dtype=np.uint8)


class YoloDetector:  # pragma: no cover - Stage 3, requires `.[vision]`
    """Same interface, ultralytics backend. Installed only in Stage 3."""

    name = "yolo"

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25) -> None:
        from ultralytics import YOLO  # imported lazily on purpose

        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image: np.ndarray, labels: tuple[str, ...] = ()) -> list[Detection]:
        # ultralytics treats a bare numpy array as BGR (OpenCV convention);
        # everything in this project is RGB. Handing it a PIL image removes the
        # ambiguity. Found the hard way: a well-trained model scored P=0.42 on
        # RGB-as-BGR input because red and blue swapped - the *under*-trained
        # 12-epoch model looked fine only because it hadn't learned colour yet.
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        res = self.model.predict(pil, conf=self.conf, verbose=False)[0]
        out: list[Detection] = []
        for box in res.boxes:
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            name = res.names[int(box.cls[0])]
            if labels and name not in labels:
                continue
            cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
            out.append(
                Detection(
                    label=name,
                    confidence=round(float(box.conf[0]), 3),
                    bbox_px=(x0, y0, x1, y1),
                    center_px=(cu, cv),
                    center_world=_px_to_world(cu, cv),
                    area_px=(x1 - x0) * (y1 - y0),
                )
            )
        return out

    def find(self, image: np.ndarray, label: str) -> Detection | None:
        d = self.detect(image, (label,))
        return d[0] if d else None
