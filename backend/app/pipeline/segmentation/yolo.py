from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2  # type: ignore
import numpy as np


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) for two binary masks."""
    # Ensure masks are boolean
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0

    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(max(1, union))


def _apply_mask_nms(
    masks: List[np.ndarray],
    polygons: List[List[Tuple[float, float]]],
    confidences: List[float],
    iou_thresh: float,
) -> Dict[str, List]:
    """
    Applies Non-Maximum Suppression using Mask IoU to filter overlapping detections.

    Args:
        masks: A list of binary mask arrays.
        polygons: A list of polygon coordinate lists.
        confidences: A list of detection confidence scores.
        iou_thresh: The IoU threshold to use for suppression.

    Returns:
        A dictionary containing the filtered lists of masks, polygons, and confidences.
    """
    if not masks:
        return {"masks": [], "polygons": [], "confidences": []}

    # Sort indices by confidence in descending order
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    keep_indices: List[int] = []
    for i in order:
        is_overlapping = False
        for j in keep_indices:
            if _mask_iou(masks[i], masks[j]) >= iou_thresh:
                is_overlapping = True
                break
        if not is_overlapping:
            keep_indices.append(i)

    # Filter all lists based on the kept indices
    return {
        "masks": [masks[i] for i in keep_indices],
        "polygons": [polygons[i] for i in keep_indices],
        "confidences": [confidences[i] for i in keep_indices],
    }


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

    # Extract binary masks directly from the tensor data
    if hasattr(result.masks, "data") and result.masks.data is not None:
        for mask_tensor in result.masks.data.cpu().numpy():
            # Binarize the mask using the 0.5 threshold
            masks.append((mask_tensor > 0.5).astype(np.uint8))

    # Extract polygons, with a fallback to generate them from masks
    if hasattr(result.masks, "xy") and result.masks.xy is not None and len(result.masks.xy) > 0:
        for poly in result.masks.xy:
            polygons.append([(float(x), float(y)) for x, y in poly])
    else:
        # Fallback: if no polygons are in the result, generate them from the masks
        for mask in masks:
            mask_u8 = mask * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # Use the largest contour as the polygon
            contour = max(contours, key=cv2.contourArea).squeeze(1)
            polygons.append([(float(x), float(y)) for x, y in contour])

    # Extract confidences
    if hasattr(result, "boxes") and getattr(result.boxes, "conf", None) is not None:
        confidences = [float(c) for c in result.boxes.conf.cpu().numpy()]

    # Ensure all lists are of the same length to prevent errors
    n = min(len(masks), len(polygons), len(confidences))
    return {
        "masks": masks[:n],
        "polygons": polygons[:n],
        "confidences": confidences[:n],
    }


def run_segmentation(
    image_bgr: np.ndarray,
    seg_model_path: Path,
    conf_thresh: float = 0.15,
    nms_iou_thresh: float = 0.80,
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
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics (YOLOv8) is required for segmentation. "
            "Install AI requirements: pip install -r backend/requirements-ai.txt"
        ) from exc

    if not seg_model_path.is_file():
        raise FileNotFoundError(f"Segmentation model not found at '{seg_model_path}'.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be a color image with shape HxWx3 (BGR).")

    # 1. Initialize Model and Prepare Image Size
    model = YOLO(str(seg_model_path))
    img_h, img_w = image_bgr.shape[:2]

    # Calculate optimal image size for YOLOv8 (multiple of 32)
    def _round_to_multiple_of_32(n: int) -> int:
        return int(np.clip((n + 31) // 32 * 32, 320, 1536))

    imgsz = _round_to_multiple_of_32(max(img_h, img_w))

    # 2. Run Prediction
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        retina_masks=True,  # Crucial for high-res masks at original image size
        conf=conf_thresh,
        verbose=False,
    )
    if not results:
        return {"polygons": [], "masks": [], "confidences": []}

    # 3. Parse Raw Results
    # This step extracts masks, polygons, and confidences from the YOLO object.
    parsed_data = _parse_yolo_results(results[0])

    # 4. Apply Non-Maximum Suppression
    # This step filters out overlapping detections based on mask IoU.
    filtered_data = _apply_mask_nms(
        masks=parsed_data["masks"],
        polygons=parsed_data["polygons"],
        confidences=parsed_data["confidences"],
        iou_thresh=nms_iou_thresh,
    )

    return {
        "polygons": filtered_data["polygons"],
        "masks": filtered_data["masks"],
        "confidences": filtered_data["confidences"],
    }