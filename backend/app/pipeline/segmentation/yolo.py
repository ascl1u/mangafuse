from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2  # type: ignore
import numpy as np


def _parse_yolo_results(result: Any) -> Dict[str, List]:
    """
    Parses the raw result object from a YOLOv8 prediction.

    Args:
        result: The YOLOv8 result object for a single image.

    Returns:
        A dictionary containing lists of masks, polygons, and confidences.
    """
    masks: List[np.ndarray] = []
    polygons: List[List[Tuple[float, float]]] = []
    confidences: List[float] = []

    if getattr(result, "masks", None) is None:
        return {"masks": [], "polygons": [], "confidences": []}

    # Prefer vector polygons; avoid allocating full-resolution masks
    if hasattr(result.masks, "xy") and result.masks.xy is not None and len(result.masks.xy) > 0:
        for poly in result.masks.xy:
            polygons.append([(float(x), float(y)) for x, y in poly])

    # Extract confidences
    if hasattr(result, "boxes") and getattr(result.boxes, "conf", None) is not None:
        confidences = [float(c) for c in result.boxes.conf.cpu().numpy()]

    # Polygons-only: avoid materializing full-resolution masks to reduce memory.
    masks = []
    n = min(len(polygons), len(confidences))
    return {
        "masks": [],
        "polygons": polygons[:n],
        "confidences": confidences[:n],
    }


def run_segmentation(
    image_bgr: np.ndarray,
    seg_model_path: Path,
    conf_thresh: float = 0.8,
    nms_iou_thresh: float = 0.2,
    yolo_model: Optional[Any] = None,
) -> Dict[str, List]:
    """
    Performs speech bubble segmentation on a manga page.

    Args:
        image_bgr: The input image in BGR format (as a NumPy array).
        seg_model_path: The local file path to the YOLOv8-seg model weights.
        conf_thresh: The confidence threshold for object detection.
        nms_iou_thresh: The Mask IoU threshold for Non-Maximum Suppression.

    Returns:
        A dictionary containing the filtered lists of polygons, masks, and confidences.
    """
    if yolo_model is None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics (YOLOv8) is required for segmentation. "
                "Install AI requirements: pip install -r backend/requirements-ai-gpu.txt"
            ) from exc

        if not seg_model_path.is_file():
            raise FileNotFoundError(f"Segmentation model not found at '{seg_model_path}'.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a color image with shape HxWx3 (BGR).")

    # 1. Initialize Model and Prepare Image Size
    model = yolo_model if yolo_model is not None else YOLO(str(seg_model_path))
    img_h, img_w = image_bgr.shape[:2]

    # 2. Run Prediction (use Ultralytics' own NMS via iou)
    # Force GPU usage (device=0). Fail fast if CUDA is not available.
    results = model.predict(
        source=image_bgr,
        imgsz=640,
        retina_masks=False,  # less memory; we'll use polygons and rasterize when needed
        conf=conf_thresh,
        iou=nms_iou_thresh,
        device=0,
        half=True, #use fp16 for faster inference
        verbose=False,
    )
    if not results:
        return {"polygons": [], "confidences": []}

    # 3. Parse Raw Results
    # This step extracts masks, polygons, and confidences from the YOLO object.
    parsed_data = _parse_yolo_results(results[0])

    # 4. Ultralytics has already applied NMS; return polygons/confidences
    return {
        "polygons": parsed_data["polygons"],
        "confidences": parsed_data["confidences"],
    }