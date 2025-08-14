from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def run_segmentation(
    image_bgr: np.ndarray,
    seg_model_path: Path,
) -> Dict:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics (YOLOv8) is required for segmentation.\nInstall AI requirements listed in backend/requirements-ai.txt"
        ) from exc

    if not seg_model_path.exists() or not seg_model_path.is_file():
        raise FileNotFoundError(
            f"Segmentation model not found at '{seg_model_path}'. Provide a local YOLOv8-seg weights file."
        )

    model = YOLO(str(seg_model_path))

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a color image with shape HxWx3 (BGR)")

    img_h, img_w = image_bgr.shape[:2]

    def _round_to_multiple_of_32(n: int) -> int:
        return int(np.clip((n + 31) // 32 * 32, 320, 1536))

    imgsz = _round_to_multiple_of_32(max(img_h, img_w))

    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        retina_masks=True,
        conf=0.15,
        verbose=False,
    )
    if not results:
        return {"polygons": [], "masks": []}

    result = results[0]

    polygons: List[List[Tuple[float, float]]] = []
    instance_masks: List[np.ndarray] = []
    confidences: List[float] = []

    if getattr(result, "masks", None) is None:
        return {"polygons": [], "masks": []}

    if hasattr(result.masks, "data") and result.masks.data is not None:
        for mask_tensor in result.masks.data.cpu().numpy():
            instance_masks.append((mask_tensor > 0.5).astype(np.uint8))
    else:
        instance_masks = []

    img_h, img_w = image_bgr.shape[:2]
    aligned_masks: List[np.ndarray] = []
    import cv2  # type: ignore
    for m in instance_masks:
        if m.shape != (img_h, img_w):
            resized = cv2.resize(m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            m = (resized > 0).astype(np.uint8)
        aligned_masks.append(m)
    instance_masks = aligned_masks

    if hasattr(result.masks, "xy") and result.masks.xy is not None and len(result.masks.xy) > 0:
        for poly in result.masks.xy:
            polygons.append([(float(x), float(y)) for x, y in poly])
    else:
        import cv2  # type: ignore
        for m in instance_masks:
            mask_u8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea).squeeze(1)
            polygons.append([(float(x), float(y)) for x, y in contour])

    if hasattr(result, "boxes") and getattr(result.boxes, "conf", None) is not None:
        try:
            confidences = [float(c) for c in result.boxes.conf.cpu().numpy().tolist()]
        except Exception:
            confidences = []

    n = min(len(polygons), len(instance_masks))
    polygons = polygons[:n]
    instance_masks = instance_masks[:n]
    if confidences:
        confidences = confidences[:n]

    def _mask_iou(a, b):
        import numpy as _np
        inter = _np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = _np.logical_or(a, b).sum()
        return float(inter) / float(max(1, union))

    if instance_masks:
        order = list(range(len(instance_masks)))
        if confidences and len(confidences) == len(instance_masks):
            order.sort(key=lambda i: confidences[i], reverse=True)
        keep: list[int] = []
        iou_thresh = 0.80
        for i in order:
            mask_i = instance_masks[i]
            drop = False
            for j in keep:
                if _mask_iou(mask_i.astype(bool), instance_masks[j].astype(bool)) >= iou_thresh:
                    drop = True
                    break
            if not drop:
                keep.append(i)
        if len(keep) < len(instance_masks):
            instance_masks = [instance_masks[i] for i in keep]
            polygons = [polygons[i] for i in keep]
            if confidences and len(confidences) >= max(keep) + 1:
                confidences = [confidences[i] for i in keep]

    return {"polygons": polygons, "masks": instance_masks, "confidences": confidences}


