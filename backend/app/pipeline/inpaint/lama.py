from __future__ import annotations
import cv2  # type: ignore
import numpy as np
from typing import List, Tuple


def _binarize_mask(mask_gray: np.ndarray) -> np.ndarray:
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _dilate(mask_bin: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return mask_bin
    k = max(1, int(dilation_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask_bin, kernel, iterations=1)


def _find_rois(mask_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # OpenCV findContours expects white foreground
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        if cnt is None or len(cnt) == 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h <= 0:
            continue
        rects.append((x, y, w, h))
    return rects


def _merge_rects(rects: List[Tuple[int, int, int, int]], gap: int) -> List[Tuple[int, int, int, int]]:
    # Expand each rect by gap and merge overlapping ones iteratively
    if not rects:
        return []
    expanded = []
    for x, y, w, h in rects:
        expanded.append((x - gap, y - gap, w + 2 * gap, h + 2 * gap))

    changed = True
    boxes = expanded
    while changed:
        changed = False
        merged: List[Tuple[int, int, int, int]] = []
        taken = [False] * len(boxes)
        for i in range(len(boxes)):
            if taken[i]:
                continue
            xi, yi, wi, hi = boxes[i]
            x0, y0, x1, y1 = xi, yi, xi + wi, yi + hi
            for j in range(i + 1, len(boxes)):
                if taken[j]:
                    continue
                xj, yj, wj, hj = boxes[j]
                u0, v0, u1, v1 = xj, yj, xj + wj, yj + hj
                if not (x1 < u0 or u1 < x0 or y1 < v0 or v1 < y0):
                    # overlap -> merge
                    x0 = min(x0, u0)
                    y0 = min(y0, v0)
                    x1 = max(x1, u1)
                    y1 = max(y1, v1)
                    taken[j] = True
                    changed = True
            taken[i] = True
            merged.append((x0, y0, x1 - x0, y1 - y0))
        boxes = merged

    # Clamp back to non-expanded approximate by shrinking gap (not strictly necessary for composition)
    return boxes


def _resize_roi(img: np.ndarray, mask: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img, mask, scale


def run_inpainting(image_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """
    Run LaMa inpainting on regions of interest (ROIs) derived from the mask, and composite
    results back into the full image. This drastically reduces peak memory compared to
    inpainting the entire image at once.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for inpainting stage") from exc

    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "simple-lama-inpainting is required. Install AI deps from backend/requirements-ai.txt"
        ) from exc

    h, w = image_bgr.shape[:2]
    if mask_gray.shape != (h, w):
        raise ValueError(f"Mask shape {mask_gray.shape} must match image shape {(h, w)}")

    # Parameters tuned for MVP stability and quality
    DILATION_PX = 8              # add local context around masked text
    MERGE_GAP_PX = 8             # merge nearby boxes to reduce number of model calls
    MIN_ROI_AREA = 32 * 32       # ignore tiny specks
    SMALL_ROI_AREA = 48 * 48     # use OpenCV inpaint for very small regions
    MAX_ROI_SIDE = 1600          # downscale large ROIs to cap memory

    # Fast exit when no work is needed
    mask_bin = _binarize_mask(mask_gray)
    if int(mask_bin.sum()) == 0:
        return image_bgr.copy()

    # Build ROIs from the mask with dilation for context and merging for stability
    dilated = _dilate(mask_bin, DILATION_PX)
    rects = _find_rois(dilated)
    # Filter by area and merge
    rects = [r for r in rects if r[2] * r[3] >= MIN_ROI_AREA]
    rects = _merge_rects(rects, MERGE_GAP_PX)
    if not rects:
        return image_bgr.copy()

    # Prepare model once and force CPU map_location to reduce peak memory fragmentation
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for inpainting stage") from exc

    original_jit_load = torch.jit.load

    def _jit_load_cpu(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return original_jit_load(path, *args, **kwargs)

    torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
    try:
        model = SimpleLama()
    except Exception as exc:  # noqa: BLE001
        torch.jit.load = original_jit_load  # restore early
        raise RuntimeError("SimpleLama inpainting failed to initialize") from exc

    # Compose results
    result = image_bgr.copy()

    try:
        for (x, y, rw, rh) in rects:
            # Clamp ROI to image bounds
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(w, x + rw)
            y1 = min(h, y + rh)
            if x1 <= x0 or y1 <= y0:
                continue

            roi_img = result[y0:y1, x0:x1].copy()
            roi_mask = mask_bin[y0:y1, x0:x1]

            area = roi_img.shape[0] * roi_img.shape[1]
            if area <= SMALL_ROI_AREA:
                # Cheap fallback for tiny regions
                radius = 3
                roi_inpainted = cv2.inpaint(roi_img, (roi_mask > 0).astype(np.uint8), radius, cv2.INPAINT_TELEA)
            else:
                # Downscale large ROIs to bound memory, then upsample back
                roi_proc_img, roi_proc_mask, scale = _resize_roi(roi_img, roi_mask, MAX_ROI_SIDE)

                pil_roi = Image.fromarray(cv2.cvtColor(roi_proc_img, cv2.COLOR_BGR2RGB))
                pil_mask = Image.fromarray(roi_proc_mask, mode="L")
                pil_out = model(pil_roi, pil_mask)
                roi_inpainted = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)

                # Ensure the inpainted ROI exactly matches the original ROI size.
                # Some backends may return off-by-few-pixels sizes; normalize by resizing.
                target_h, target_w = roi_img.shape[:2]
                if roi_inpainted.shape[0] != target_h or roi_inpainted.shape[1] != target_w:
                    roi_inpainted = cv2.resize(roi_inpainted, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            # Composite only masked pixels
            m = (roi_mask > 0)
            roi_target = result[y0:y1, x0:x1]
            roi_target[m] = roi_inpainted[m]
            result[y0:y1, x0:x1] = roi_target

    finally:
        # Restore torch jit loader
        torch.jit.load = original_jit_load  # type: ignore[assignment]

    return result