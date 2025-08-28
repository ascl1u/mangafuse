from __future__ import annotations
import cv2  # type: ignore
import numpy as np
from typing import List, Tuple, Any, Optional


def _binarize_mask(mask_gray: np.ndarray) -> np.ndarray:
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def run_inpainting(image_bgr: np.ndarray, mask_gray: np.ndarray, lama_model: Optional[Any] = None) -> np.ndarray:
    """
    Single-pass LaMa inpainting over the full image using the provided mask.

    The mask should be a single-channel (uint8) image where non-zero pixels
    indicate regions to inpaint.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for inpainting stage") from exc

    if lama_model is None:
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "simple-lama-inpainting is required. Install AI deps from backend/requirements-ai.txt"
            ) from exc

    h, w = image_bgr.shape[:2]
    if mask_gray.shape != (h, w):
        raise ValueError(f"Mask shape {mask_gray.shape} must match image shape {(h, w)}")

    # Fast exit when no work is needed
    mask_bin = _binarize_mask(mask_gray)
    if int(mask_bin.sum()) == 0:
        return image_bgr.copy()

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for inpainting stage") from exc

    # Prepare PIL inputs
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_bin, mode="L")

    # Choose model instance
    if lama_model is None:
        model = SimpleLama()
    else:
        model = lama_model

    if torch.cuda.is_available():
        try:
            with torch.autocast("cuda", dtype=torch.float16):  # type: ignore[attr-defined]
                pil_out = model(pil_img, pil_mask)
        except Exception:
            pil_out = model(pil_img, pil_mask)
    else:
        pil_out = model(pil_img, pil_mask)

    out_bgr = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)
    # Normalize size if backend returns slightly different dims
    if out_bgr.shape[:2] != (h, w):
        out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return out_bgr