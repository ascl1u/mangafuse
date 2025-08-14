from __future__ import annotations

from pathlib import Path
from typing import List

import cv2  # type: ignore
import numpy as np

from .io import ensure_dir, save_png


def save_masks(
    masks_dir: Path,
    instance_masks: List[np.ndarray],
    combined_mask_path: Path,
    image_height: int,
    image_width: int,
) -> None:
    ensure_dir(masks_dir)
    combined = None
    for idx, mask in enumerate(instance_masks, start=1):
        per_path = masks_dir / f"{idx}.png"
        mask_u8 = (mask.astype(np.uint8) * 255)
        save_png(per_path, cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR))
        if combined is None:
            combined = mask.astype(np.uint8)
        else:
            combined = np.maximum(combined, mask.astype(np.uint8))

    if combined is None:
        combined = np.zeros((image_height, image_width), dtype=np.uint8)

    save_png(combined_mask_path, cv2.cvtColor((combined * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))



