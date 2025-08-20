from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple


def run_pipeline(
    job_id: str,
    image_path: str,
    *,
    depth: Literal["cleaned", "full"] = "cleaned",
    debug: bool = False,
    force: bool = False,
    seg_model_path: Optional[str] = None,
    font_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Execute the full pipeline and return the compact result payload.

    Heavy dependencies are imported lazily within this function to keep the worker import fast.
    The function writes artifacts to artifacts/jobs/{task_id}/ based on the Celery task id
    that should be embedded in the job_dir path by the caller.

    The caller is responsible for establishing the job directory (via task id resolution)
    and passing the proper `image_path`.
    """
    # Lazy imports to avoid heavy deps at module import time
    import os
    import json
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from app.pipeline.utils.io import ensure_dir, read_image_bgr, save_png
    from app.pipeline.segmentation.yolo import run_segmentation
    from app.pipeline.utils.masks import save_masks
    from app.pipeline.utils.visualization import make_overlay
    from app.pipeline.utils.textio import write_text_json, read_text_json, save_text_records
    from app.pipeline.ocr.crops import tight_crop_from_mask
    from app.pipeline.ocr.preprocess import binarize_for_ocr
    from app.pipeline.ocr.engine import MangaOcrEngine
    from app.pipeline.translate.gemini import GeminiTranslator
    from app.pipeline.inpaint.lama import run_inpainting
    from app.pipeline.typeset.model import BubbleText
    from app.pipeline.typeset.render import render_typeset
    from app.pipeline.inpaint.text_mask import build_text_inpaint_mask

    # Determine job directory from image path location convention:
    # tasks write under artifacts/jobs/{job_id}/; maintain this here by resolving relative
    input_image_path = Path(image_path)
    if not input_image_path.exists():
        raise FileNotFoundError(f"input image not found: {input_image_path}")

    # Infer job_dir from our standard layout (..../artifacts/jobs/<id>/uploads/<file> -> job_dir)
    # However, for clarity we derive job_dir from artifacts path alongside the image file
    # If the image is uploaded under artifacts/uploads, we will resolve task id later.
    # The caller should provide the correct job_dir in practice; here we mirror tasks logic.
    # We compute the repo root from the known backend layout.
    from app.core.paths import get_assets_root, get_job_dir


    # job_dir is provided by the caller task via an environment variable (minimal churn design)
    # The task sets TASK_JOB_DIR to artifacts/jobs/{task_id}

    # Prepare progress reporting
    def _update(stage: str, progress: float) -> None:
        if progress_callback:
            try:
                progress_callback(stage, progress)
            except Exception:
                pass

    # Establish artifact dirs based on the provided job id
    job_dir = get_job_dir(job_id)
    masks_dir = job_dir / "masks"
    ensure_dir(job_dir)
    ensure_dir(masks_dir)

    # Resolve assets
    assets_root = get_assets_root()
    default_seg_model = assets_root / "models" / "model.pt"
    default_font = assets_root / "fonts" / "animeace2_reg.ttf"
    seg_model = Path(seg_model_path) if seg_model_path else default_seg_model
    font = Path(font_path) if font_path else default_font

    # Prepare common paths
    overlay_path = job_dir / "segmentation_overlay.png"
    combined_mask_path = masks_dir / "all_mask.png"
    text_mask_path = masks_dir / "text_mask.png"
    json_path = job_dir / "text.json"
    cleaned_path = job_dir / "cleaned.png"
    final_path = job_dir / "final.png"
    typeset_debug_path = job_dir / "typeset_debug.png"

    stage_completed: List[str] = []

    # Load once for dimensions and for segmentation
    image_bgr = read_image_bgr(input_image_path)
    height, width = image_bgr.shape[:2]

    # 1) Segmentation
    _update("segmentation", 0.1)
    if force or not (overlay_path.exists() and combined_mask_path.exists() and json_path.exists()):
        result = run_segmentation(image_bgr=image_bgr, seg_model_path=seg_model)
        polygons = result.get("polygons", [])
        instance_masks = result.get("masks", [])
        confidences = result.get("confidences", [])
        overlay_bgr = make_overlay(image_bgr, instance_masks, polygons, confidences)
        save_png(overlay_path, overlay_bgr)
        save_masks(masks_dir, instance_masks, combined_mask_path, image_height=height, image_width=width)
        write_text_json(json_path, polygons)
        if debug:
            dbg = overlay_bgr.copy()
            for idx, poly in enumerate(polygons, start=1):
                if not poly:
                    continue
                x0, y0 = int(poly[0][0]), int(poly[0][1])
                cv2.putText(dbg, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(dbg, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            save_png(job_dir / "segmentation_overlay_ids.png", dbg)
    stage_completed.append("segmentation")
    _update("segmentation_complete", 0.2)

    # Early exit if no bubbles: still provide minimal payload
    data = read_text_json(json_path)
    bubbles = data.get("bubbles", [])

    # 2) OCR (full only)
    if depth == "full" and bubbles:
        _update("ocr", 0.35)
        ocr_engine = MangaOcrEngine()
        for rec in bubbles:
            has_ja = isinstance(rec.get("ja_text"), str) and rec["ja_text"].strip() != ""
            if has_ja and not force:
                continue
            bubble_id = int(rec.get("id"))
            polygon = rec.get("polygon") or []
            mask_path = masks_dir / f"{bubble_id}.png"
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_gray, polygon)
            try:
                bin_img = binarize_for_ocr(crop_bgr)
                ja_text = ocr_engine.run(bin_img)
            except Exception:
                ja_text = ocr_engine.run(crop_bgr)
            rec["ja_text"] = ja_text
        save_text_records(json_path, bubbles)
        stage_completed.append("ocr")
        _update("ocr_complete", 0.45)

    # 3) Translate (full only)
    if depth == "full" and bubbles:
        _update("translate", 0.5)
        api_key = os.getenv("GOOGLE_API_KEY", "")
        translator = GeminiTranslator(api_key=api_key)
        indices: List[int] = []
        texts: List[str] = []
        # Ensure ja_text exists when forced or missing
        if force:
            ocr_engine = MangaOcrEngine()
            for rec in bubbles:
                if not isinstance(rec.get("ja_text"), str) or rec["ja_text"].strip() == "":
                    bubble_id = int(rec.get("id"))
                    polygon = rec.get("polygon") or []
                    mask_path = masks_dir / f"{bubble_id}.png"
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_gray, polygon)
                    try:
                        rec["ja_text"] = ocr_engine.run(binarize_for_ocr(crop_bgr))
                    except Exception:
                        rec["ja_text"] = ocr_engine.run(crop_bgr)
        for idx, rec in enumerate(bubbles):
            ja = (rec.get("ja_text") or "").strip()
            has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip() != ""
            if not ja:
                continue
            if has_en and not force:
                continue
            indices.append(idx)
            texts.append(ja)
        if texts:
            en_list = translator.translate_batch(texts)
            for i, en in zip(indices, en_list):
                bubbles[i]["en_text"] = en
            save_text_records(json_path, bubbles)
        stage_completed.append("translate")
        _update("translate_complete", 0.6)

    # 4) Inpaint (always for cleaned, and for full after translate)
    if depth in ("cleaned", "full"):
        _update("inpaint", 0.7)
        # Prefer text-only mask; build if missing or forcing
        if force or not text_mask_path.exists():
            data = read_text_json(json_path)
            bubbles_for_mask = data.get("bubbles", [])
            # Use in-memory masks if available from the segmentation stage; otherwise, load from disk
            if 'instance_masks' in locals() and isinstance(instance_masks, list) and instance_masks:
                # Ensure masks are single-channel uint8 (0/255)
                imasks = [
                    (m.astype(np.uint8) if m.dtype != np.uint8 else m)
                    for m in instance_masks
                ]
            else:
                imasks = []
                for rec in bubbles_for_mask:
                    try:
                        bubble_id = int(rec.get("id"))
                    except Exception:
                        bubble_id = None
                    if bubble_id is None:
                        # maintain index alignment with a blank mask
                        imasks.append(np.zeros((height, width), dtype=np.uint8))
                        continue
                    mpath = masks_dir / f"{bubble_id}.png"
                    m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        m = np.zeros((height, width), dtype=np.uint8)
                    elif m.ndim == 3 and m.shape[2] == 3:
                        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                    elif m.ndim == 3 and m.shape[2] == 4:
                        m = cv2.cvtColor(m, cv2.COLOR_BGRA2GRAY)
                    imasks.append(m)
            text_mask = build_text_inpaint_mask(image_bgr, imasks, bubbles_for_mask)
            save_png(text_mask_path, cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR))
        # Choose mask: text_mask if non-empty else combined
        # Prepare grayscale mask for inpainting (ensure strictly 2D HxW)
        def _to_gray_2d(img):  # type: ignore[no-redef]
            if img is None:
                return None
            if getattr(img, "ndim", 2) == 2:
                return img
            if img.ndim == 3:
                if img.shape[2] == 1:
                    return img[:, :, 0]
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return None

        text_mask_img = cv2.imread(str(text_mask_path), cv2.IMREAD_UNCHANGED)
        text_mask_gray = _to_gray_2d(text_mask_img)
        if text_mask_gray is not None and int(text_mask_gray.sum()) > 0:
            mask_for_inpaint = text_mask_gray
        else:
            combined_img = cv2.imread(str(combined_mask_path), cv2.IMREAD_UNCHANGED)
            combined_gray = _to_gray_2d(combined_img)
            mask_for_inpaint = combined_gray if combined_gray is not None else np.zeros((height, width), dtype=np.uint8)
        result_bgr = run_inpainting(image_bgr, mask_for_inpaint)
        save_png(cleaned_path, result_bgr)
        stage_completed.append("inpaint")
        _update("inpaint_complete", 0.85)

    # 5) Typeset (full only)
    if depth == "full":
        _update("typeset", 0.9)
        if not cleaned_path.exists():
            raise FileNotFoundError("cleaned.png not found after inpaint stage")
        data = read_text_json(json_path)
        bubbles2 = data.get("bubbles", [])
        records: List[BubbleText] = []
        for rec in bubbles2:
            text = (rec.get("en_text") or rec.get("ja_text") or "").strip()
            records.append(
                BubbleText(
                    bubble_id=int(rec.get("id")),
                    polygon=rec.get("polygon") or [],
                    text=text,
                    font_size=(int(rec.get("font_size")) if isinstance(rec.get("font_size"), (int, float)) else None),
                )
            )
        cleaned_image = read_image_bgr(cleaned_path)
        used_sizes, used_rects = render_typeset(
            cleaned_image_bgr=cleaned_image,
            output_final_path=final_path,
            records=records,
            font_path=font,
            debug=debug,
            debug_overlay_path=typeset_debug_path if debug else None,
            text_layer_output_path=job_dir / "text_layer.png",
        )
        # Persist computed font sizes back into text.json for parity with editor
        try:
            if used_sizes and isinstance(used_sizes, dict):
                for rec in bubbles2:
                    bid = int(rec.get("id"))
                    if bid in used_sizes:
                        rec["font_size"] = int(used_sizes[bid])
                    if used_rects and bid in used_rects:
                        rect = used_rects[bid]
                        rec["rect"] = {
                            "x": int(rect["x"]),
                            "y": int(rect["y"]),
                            "w": int(rect["w"]),
                            "h": int(rect["h"]),
                        }
                save_text_records(json_path, bubbles2)
        except Exception:
            pass
        stage_completed.append("typeset")
        _update("typeset_complete", 1.0)

    # Build result payload
    num_bubbles = len(bubbles)
    # Provide a minimal editor payload by default; will be overwritten below if we can build a richer one
    default_image_url = (
        f"/artifacts/jobs/{job_id}/cleaned.png" if cleaned_path.exists() else (
            f"/artifacts/jobs/{job_id}/final.png" if final_path.exists() else f"/artifacts/jobs/{job_id}/segmentation_overlay.png"
        )
    )
    result: Dict[str, Any] = {
        "stage_completed": stage_completed,
        "width": int(width),
        "height": int(height),
        "num_bubbles": int(num_bubbles),
        "json_url": f"/artifacts/jobs/{job_id}/text.json",
        "overlay_url": f"/artifacts/jobs/{job_id}/segmentation_overlay.png",
        "editor_payload": {
            "image_url": default_image_url,
            "width": int(width),
            "height": int(height),
            "bubbles": [],
        },
    }
    if cleaned_path.exists():
        result["cleaned_url"] = f"/artifacts/jobs/{job_id}/cleaned.png"
    if final_path.exists():
        result["final_url"] = f"/artifacts/jobs/{job_id}/final.png"
    if debug and typeset_debug_path.exists():
        result["typeset_debug_url"] = f"/artifacts/jobs/{job_id}/typeset_debug.png"

    # Provide artifact paths for worker upload
    artifacts: Dict[str, str] = {}
    if cleaned_path.exists():
        artifacts["CLEANED_PAGE"] = str(cleaned_path)
    if final_path.exists():
        artifacts["FINAL_PNG"] = str(final_path)
    text_layer_path = job_dir / "text_layer.png"
    if text_layer_path.exists():
        artifacts["TEXT_LAYER_PNG"] = str(text_layer_path)
    if artifacts:
        result["artifacts"] = artifacts

    # Write editor_payload.json for frontend editor
    try:
        data_for_editor = read_text_json(json_path)
        bubbles_for_editor = data_for_editor.get("bubbles", [])
        normalized: List[Dict[str, Any]] = []
        crops_dir = job_dir / "crops"
        ensure_dir(crops_dir)
        for rec in bubbles_for_editor:
            bubble_id = int(rec.get("id"))
            polygon = rec.get("polygon") or []
            try:
                mask_gray = cv2.imread(str(masks_dir / f"{bubble_id}.png"), cv2.IMREAD_GRAYSCALE)
                crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_gray, polygon)
                save_png(crops_dir / f"{bubble_id}.png", crop_bgr)
                crop_url = f"/artifacts/jobs/{job_dir.name}/crops/{bubble_id}.png"
            except Exception:
                crop_url = None
            rect_payload = rec.get("rect") if isinstance(rec.get("rect"), dict) else None
            normalized.append(
                {
                    "id": bubble_id,
                    "polygon": polygon,
                    "ja_text": rec.get("ja_text"),
                    "en_text": rec.get("en_text"),
                    "font_size": rec.get("font_size"),
                    "rect": rect_payload,
                    "crop_url": crop_url,
                }
            )
        # Use cleaned as background to avoid double-rendering text
        if cleaned_path.exists():
            image_url = f"/artifacts/jobs/{job_id}/cleaned.png"
        elif final_path.exists():
            image_url = f"/artifacts/jobs/{job_id}/final.png"
        else:
            image_url = f"/artifacts/jobs/{job_id}/segmentation_overlay.png"
        editor_payload = {
            "image_url": image_url,
            "width": int(width),
            "height": int(height),
            "bubbles": normalized,
        }
        editor_payload_path = job_dir / "editor_payload.json"
        with open(editor_payload_path, "w", encoding="utf-8") as f:
            json.dump(editor_payload, f, ensure_ascii=False, indent=2)
        result["editor_payload_url"] = f"/artifacts/jobs/{job_id}/editor_payload.json"
        result["editor_payload"] = editor_payload
    except Exception:
        pass

    return result


def apply_edits(
    job_dir: Path,
    edits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply user edits to an existing job and re-typeset the final image.

    Heavy imports are inside to keep worker import light.
    """
    import json
    import cv2  # type: ignore
    from app.pipeline.utils.textio import read_text_json, save_text_records
    from app.pipeline.typeset.model import BubbleText
    from app.pipeline.typeset.render import render_typeset
    from app.pipeline.utils.io import save_png
    from app.core.paths import get_assets_root

    job_id = job_dir.name
    json_path = job_dir / "text.json"
    cleaned_path = job_dir / "cleaned.png"
    final_path = job_dir / "final.png"
    text_layer_path = job_dir / "text_layer.png"
    editor_payload_path = job_dir / "editor_payload.json"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned image missing for job {job_id}")
    if not json_path.exists():
        raise FileNotFoundError(f"text.json missing for job {job_id}")

    # Persist raw edits for audit/debugging
    try:
        with open(job_dir / "edits.json", "w", encoding="utf-8") as f:
            json.dump(edits, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    data = read_text_json(json_path)
    bubbles = data.get("bubbles", [])
    by_id: Dict[int, Dict[str, Any]] = {}
    for rec in bubbles:
        try:
            by_id[int(rec.get("id"))] = rec
        except Exception:
            continue
    for e in edits:
        try:
            bid = int(e.get("id"))
        except Exception:
            continue
        rec = by_id.get(bid)
        if not rec:
            continue
        if isinstance(e.get("en_text"), str):
            rec["en_text"] = e["en_text"]
        if isinstance(e.get("font_size"), (int, float)):
            rec["font_size"] = int(e["font_size"])  # candidate; normalized by typesetter

    save_text_records(json_path, bubbles)

    # Resolve font path (defaults)
    font = get_assets_root() / "fonts" / "animeace2_reg.ttf"

    records: List[BubbleText] = []
    for rec in bubbles:
        text = (rec.get("en_text") or rec.get("ja_text") or "").strip()
        records.append(
            BubbleText(
                bubble_id=int(rec.get("id")),
                polygon=rec.get("polygon") or [],
                text=text,
                font_size=(int(rec.get("font_size")) if isinstance(rec.get("font_size"), (int, float)) else None),
            )
        )

    # Read cleaned image and typeset into memory
    img_cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_COLOR)
    if img_cleaned is None:
        raise FileNotFoundError(f"cleaned image missing for job {job_id}")
    used_sizes, used_rects = render_typeset(
        cleaned_image_bgr=img_cleaned,
        output_final_path=final_path,
        records=records,
        font_path=font,
        debug=False,
        debug_overlay_path=None,
        text_layer_output_path=text_layer_path,
    )

    # Persist normalized font sizes (and rects)
    try:
        if used_sizes and isinstance(used_sizes, dict):
            for rec in bubbles:
                bid = int(rec.get("id"))
                if bid in used_sizes:
                    rec["font_size"] = int(used_sizes[bid])
                if used_rects and bid in used_rects:
                    rect = used_rects[bid]
                    rec["rect"] = {
                        "x": int(rect["x"]),
                        "y": int(rect["y"]),
                        "w": int(rect["w"]),
                        "h": int(rect["h"]),
                    }
            save_text_records(json_path, bubbles)
    except Exception:
        pass

    # Update editor payload to include latest en_text/font_size
    normalized: List[Dict[str, Any]] = []
    for rec in bubbles:
        bubble_id = int(rec.get("id"))
        polygon = rec.get("polygon") or []
        crop_url = None
        crop_path = job_dir / "crops" / f"{bubble_id}.png"
        if crop_path.exists():
            crop_url = f"/artifacts/jobs/{job_id}/crops/{bubble_id}.png"
        rect_payload = rec.get("rect") if isinstance(rec.get("rect"), dict) else None
        normalized.append(
            {
                "id": bubble_id,
                "polygon": polygon,
                "ja_text": rec.get("ja_text"),
                "en_text": rec.get("en_text"),
                "font_size": rec.get("font_size"),
                "rect": rect_payload,
                "crop_url": crop_url,
            }
        )
    image_url = (
        f"/artifacts/jobs/{job_id}/cleaned.png" if cleaned_path.exists() else f"/artifacts/jobs/{job_id}/final.png"
    )
    editor_payload = {
        "image_url": image_url,
        "width": None,
        "height": None,
        "bubbles": normalized,
    }
    # Best-effort infer dimensions from cleaned image
    try:
        img = cv2.imread(str(cleaned_path))
        if img is not None:
            h, w = img.shape[:2]
            editor_payload["width"] = int(w)
            editor_payload["height"] = int(h)
    except Exception:
        pass
    try:
        with open(editor_payload_path, "w", encoding="utf-8") as f:
            json.dump(editor_payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    result: Dict[str, Any] = {
        "task_id": job_id,
        "final_url": f"/artifacts/jobs/{job_id}/final.png" if final_path.exists() else None,
        "text_layer_url": f"/artifacts/jobs/{job_id}/text_layer.png" if text_layer_path.exists() else None,
        "editor_payload_url": f"/artifacts/jobs/{job_id}/editor_payload.json" if editor_payload_path.exists() else None,
        "json_url": f"/artifacts/jobs/{job_id}/text.json" if json_path.exists() else None,
        "cleaned_url": f"/artifacts/jobs/{job_id}/cleaned.png" if cleaned_path.exists() else None,
    }
    # Expose local artifact file paths for the worker to persist via storage
    artifacts: Dict[str, str] = {}
    if final_path.exists():
        artifacts["FINAL_PNG"] = str(final_path)
    if text_layer_path.exists():
        artifacts["TEXT_LAYER_PNG"] = str(text_layer_path)
    if cleaned_path.exists():
        artifacts["CLEANED_PAGE"] = str(cleaned_path)
    if artifacts:
        result["artifacts"] = artifacts
    return result
