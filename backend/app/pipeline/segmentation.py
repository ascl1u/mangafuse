from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def run_segmentation(
    image_bgr: np.ndarray,
    seg_model_path: Path,
) -> Dict:
    try:
        from ultralytics import YOLO  # import here to avoid import cost if unused
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics (YOLOv8) is required for segmentation.\n"
            "Install AI requirements listed in backend/requirements-ai.txt"
        ) from exc

    if not seg_model_path.exists() or not seg_model_path.is_file():
        raise FileNotFoundError(
            f"Segmentation model not found at '{seg_model_path}'. Provide a local YOLOv8-seg weights file."
        )

    model = YOLO(str(seg_model_path))

    # Ultralytics accepts BGR numpy arrays; ensure shape HxWx3
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a color image with shape HxWx3 (BGR)")

    # Use a higher inference size and retina masks for better segmentation quality
    img_h, img_w = image_bgr.shape[:2]

    def _round_to_multiple_of_32(n: int) -> int:
        return int(np.clip((n + 31) // 32 * 32, 320, 1536))

    imgsz = _round_to_multiple_of_32(max(img_h, img_w))

    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        retina_masks=True,
        conf=0.15,  # slightly lower to reduce missed small bubbles
        verbose=False,
    )
    if not results:
        return {"polygons": [], "masks": []}

    result = results[0]

    polygons: List[List[Tuple[float, float]]] = []
    instance_masks: List[np.ndarray] = []
    confidences: List[float] = []

    # result.masks can be None if no instances
    if getattr(result, "masks", None) is None:
        return {"polygons": [], "masks": []}

    # Extract raw masks first
    if hasattr(result.masks, "data") and result.masks.data is not None:
        for mask_tensor in result.masks.data.cpu().numpy():
            instance_masks.append((mask_tensor > 0.5).astype(np.uint8))
    else:
        # If data missing and xy present, we will synthesize masks from polygons below
        instance_masks = []

    # Resize masks to original image size if needed
    img_h, img_w = image_bgr.shape[:2]
    aligned_masks: List[np.ndarray] = []
    import cv2
    for m in instance_masks:
        if m.shape != (img_h, img_w):
            resized = cv2.resize(m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            m = (resized > 0).astype(np.uint8)
        aligned_masks.append(m)
    instance_masks = aligned_masks

    # Prefer polygons from xy (already in original coordinates)
    if hasattr(result.masks, "xy") and result.masks.xy is not None and len(result.masks.xy) > 0:
        for poly in result.masks.xy:
            polygons.append([(float(x), float(y)) for x, y in poly])
    else:
        # Fallback: compute polygons from the (now) resized masks
        import cv2
        for m in instance_masks:
            mask_u8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea).squeeze(1)
            polygons.append([(float(x), float(y)) for x, y in contour])

    # Extract confidences, aligned with instances when available
    if hasattr(result, "boxes") and getattr(result.boxes, "conf", None) is not None:
        try:
            confidences = [float(c) for c in result.boxes.conf.cpu().numpy().tolist()]
        except Exception:
            confidences = []

    # Align counts defensively
    n = min(len(polygons), len(instance_masks))
    polygons = polygons[:n]
    instance_masks = instance_masks[:n]
    if confidences:
        confidences = confidences[:n]

    return {"polygons": polygons, "masks": instance_masks, "confidences": confidences}
    
