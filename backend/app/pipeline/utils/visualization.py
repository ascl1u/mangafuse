from __future__ import annotations

from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


def generate_distinct_colors(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    colors = []
    for _ in range(num_colors):
        color = tuple(int(c) for c in rng.integers(low=64, high=255, size=3))
        colors.append((color[2], color[1], color[0]))
    return colors


def make_overlay(
    image_bgr: np.ndarray,
    instance_masks: List[np.ndarray],
    polygons: List[List[Tuple[float, float]]],
    confidences: Optional[List[float]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors_bgr = generate_distinct_colors(len(instance_masks))
    for idx, mask in enumerate(instance_masks):
        color = colors_bgr[idx]
        colored = np.zeros_like(image_bgr)
        colored[:, :] = color
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(image_bgr[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0)
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), thickness=1)
        if polygons and idx < len(polygons):
            poly = polygons[idx]
            if len(poly) > 0:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
                label = None
                if confidences and idx < len(confidences):
                    label = f"{confidences[idx]:.2f}"
                if label is not None:
                    cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return overlay



