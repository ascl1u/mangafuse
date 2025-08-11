"""
MangaFuse AI Pipeline — Phase 2 Step 2.1: Bubble Segmentation

Implements the segmentation stage using a local YOLOv8 segmentation model.
Produces:
- artifacts/segmentation_overlay.png
- artifacts/masks/{id}.png and artifacts/masks/all_mask.png
- artifacts/text.json containing polygon geometry per detected bubble

Notes:
- Requires a local YOLOv8 segmentation weights file (no auto-download).
- CPU-only PyTorch is expected for Phase 2.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from typing import Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def save_png(path: Path, image_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def generate_distinct_colors(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    colors = []
    for _ in range(num_colors):
        # Bright, diverse colors in BGR for OpenCV
        color = tuple(int(c) for c in rng.integers(low=64, high=255, size=3))
        colors.append((color[2], color[1], color[0]))  # convert RGB->BGR permutation
    return colors


def run_segmentation(
    image_bgr: np.ndarray,
    seg_model_path: Path,
) -> Dict:
    try:
        from ultralytics import YOLO  # import here to avoid import cost if unused
    except Exception as exc:
        raise RuntimeError("Ultralytics (YOLOv8) is required for segmentation.\n"
                           "Install AI requirements listed in backend/requirements-ai.txt") from exc

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
        # draw contour for visual clarity
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), thickness=1)
        # annotate confidence near polygon centroid if available
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
        # Save as binary 0/255 for visibility
        mask_u8 = (mask.astype(np.uint8) * 255)
        save_png(per_path, cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR))
        if combined is None:
            combined = mask.astype(np.uint8)
        else:
            combined = np.maximum(combined, mask.astype(np.uint8))

    if combined is None:
        # No instances -> empty mask aligned to the input image size
        combined = np.zeros((image_height, image_width), dtype=np.uint8)

    save_png(combined_mask_path, cv2.cvtColor((combined * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))


def write_text_json(
    json_path: Path,
    polygons: List[List[Tuple[float, float]]],
) -> None:
    ensure_dir(json_path.parent)
    records = []
    for idx, poly in enumerate(polygons, start=1):
        records.append(
            {
                "id": idx,
                "polygon": [[float(x), float(y)] for x, y in poly],
            }
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"bubbles": records}, f, ensure_ascii=False, indent=2)


def segment_stage(
    image_path: Path,
    out_dir: Path,
    seg_model_path: Path,
    force: bool,
    debug: bool,
) -> None:
    overlay_path = out_dir / "segmentation_overlay.png"
    masks_dir = out_dir / "masks"
    combined_mask_path = masks_dir / "all_mask.png"
    json_path = out_dir / "text.json"

    if not force and overlay_path.exists() and combined_mask_path.exists() and json_path.exists():
        print(json.dumps({"stage": "seg", "status": "skip", "reason": "artifacts_exist"}))
        return

    image_bgr = read_image_bgr(image_path)
    result = run_segmentation(image_bgr=image_bgr, seg_model_path=seg_model_path)

    polygons = result["polygons"]
    masks = result["masks"]
    confidences = result.get("confidences", [])

    # Save overlay with confidences
    overlay_bgr = make_overlay(image_bgr, masks, polygons, confidences)
    save_png(overlay_path, overlay_bgr)

    # Save per-instance masks and combined mask
    h, w = image_bgr.shape[:2]
    save_masks(masks_dir, masks, combined_mask_path, image_height=h, image_width=w)

    # Initialize text.json with polygons only
    write_text_json(json_path, polygons)

    if debug:
        # Save an additional debug with IDs
        debug_path = out_dir / "segmentation_overlay_ids.png"
        debug_img = overlay_bgr.copy()
        for idx, poly in enumerate(polygons, start=1):
            if len(poly) == 0:
                continue
            # Put ID near the first vertex
            x0, y0 = int(poly[0][0]), int(poly[0][1])
            cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        save_png(debug_path, debug_img)

    print(json.dumps({"stage": "seg", "status": "ok", "num_bubbles": len(polygons)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MangaFuse AI Pipeline (Phase 2)")
    parser.add_argument("--image", type=str, required=True, help="Path to input page image")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["seg", "ocr", "translate", "inpaint", "typeset", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(Path("assets/models/model.pt")),
        help="Path to local YOLOv8 segmentation weights (.pt)",
    )
    parser.add_argument("--debug", action="store_true", help="Save additional debug outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")

    # Future flags (parsed but unused in 2.1)
    parser.add_argument("--font", type=str, default=str(Path("assets/fonts/AnimeAce.ttf")), help="Path to TTF font")
    parser.add_argument("--use-placeholder-text", action="store_true", help="Bypass OCR/translation during typeset stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    seg_model_path = Path(args.seg_model)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    ensure_dir(out_dir)
    ensure_dir(out_dir / "masks")

    stage = args.stage
    if stage == "seg" or stage == "all":
        segment_stage(
            image_path=image_path,
            out_dir=out_dir,
            seg_model_path=seg_model_path,
            force=args.force,
            debug=args.debug,
        )
        # If only 'seg' requested, return early
        if stage == "seg":
            return

    # Placeholders for future steps (to be implemented in Steps 2.2–2.3)
    if stage in ("ocr", "translate", "inpaint", "typeset"):
        raise NotImplementedError(
            f"Stage '{stage}' is not implemented yet. Complete Step 2.1 first for segmentation."
        )


if __name__ == "__main__":
    main()

