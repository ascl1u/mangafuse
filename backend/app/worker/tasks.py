from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2  # type: ignore

from app.worker.celery_app import celery_app
from app.pipeline.io import ensure_dir, read_image_bgr, save_png
from app.pipeline.segmentation import run_segmentation
from app.pipeline.masks import save_masks
from app.pipeline.visualization import make_overlay
from app.pipeline.textio import write_text_json, read_text_json, save_text_records
from app.pipeline.crops import tight_crop_from_mask
from app.pipeline.preprocess import binarize_for_ocr
from app.pipeline.ocr_engine import MangaOcrEngine
from app.pipeline.translator import GeminiTranslator
from app.pipeline.inpaint import run_inpainting
from app.pipeline.typeset import BubbleText, render_typeset
from app.pipeline.text_mask import build_text_inpaint_mask


logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.demo_task")
def demo_task(self, duration_s: int = 5) -> Dict[str, Any]:
    """A simple demo task that sleeps then returns a payload."""
    duration = max(0, int(duration_s or 0))
    logger.info("task_started", extra={"task": "demo_task", "duration_s": duration})
    time.sleep(duration)
    result: Dict[str, Any] = {"status": "completed", "slept_seconds": duration}
    logger.info("task_completed", extra={"task": "demo_task", "result": result})
    return result


def _artifact_url(task_id: str, *parts: str) -> str:
    joined = "/".join(["artifacts", "jobs", task_id] + list(parts))
    return f"/{joined}"


@celery_app.task(bind=True, name="app.worker.tasks.process_page_task")
def process_page_task(
    self,
    image_path: str,
    *,
    depth: Literal["cleaned", "full"] = "cleaned",
    debug: bool = False,
    force: bool = False,
    seg_model_path: Optional[str] = None,
    font_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the MangaFuse pipeline for a single page according to depth.

    Artifacts are written to artifacts/jobs/{task_id}/ and a compact JSON result is returned.
    """
    task_id: str = str(getattr(self.request, "id", "unknown"))
    # Write artifacts under repo_root so FastAPI's static mount can serve them in dev
    repo_root = Path(__file__).resolve().parents[3]
    job_dir = repo_root / "artifacts" / "jobs" / task_id
    masks_dir = job_dir / "masks"
    ensure_dir(job_dir)
    ensure_dir(masks_dir)

    # Resolve assets
    repo_root = Path(__file__).resolve().parents[3]
    default_seg_model = repo_root / "assets" / "models" / "model.pt"
    default_font = repo_root / "assets" / "fonts" / "animeace2_reg.ttf"
    seg_model = Path(seg_model_path) if seg_model_path else default_seg_model
    font = Path(font_path) if font_path else default_font

    # Prepare common paths
    input_image_path = Path(image_path)
    overlay_path = job_dir / "segmentation_overlay.png"
    combined_mask_path = masks_dir / "all_mask.png"
    text_mask_path = masks_dir / "text_mask.png"
    json_path = job_dir / "text.json"
    cleaned_path = job_dir / "cleaned.png"
    final_path = job_dir / "final.png"
    typeset_debug_path = job_dir / "typeset_debug.png"

    # Progress helper
    def _update(stage: str, progress: float) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "progress": progress})

    stage_completed: List[str] = []

    # Validate input
    if not input_image_path.exists():
        raise FileNotFoundError(f"input image not found: {input_image_path}")

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
        image_bgr = image_bgr  # already loaded
        for rec in bubbles:
            has_ja = isinstance(rec.get("ja_text"), str) and rec["ja_text"].strip() != ""
            if has_ja and not force:
                continue
            bubble_id = int(rec.get("id"))
            polygon = rec.get("polygon") or []
            mask_path = masks_dir / f"{bubble_id}.png"
            crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_path, polygon)
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
            # re-OCR missing ja_text
            ocr_engine = MangaOcrEngine()
            for rec in bubbles:
                if not isinstance(rec.get("ja_text"), str) or rec["ja_text"].strip() == "":
                    bubble_id = int(rec.get("id"))
                    polygon = rec.get("polygon") or []
                    mask_path = masks_dir / f"{bubble_id}.png"
                    crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_path, polygon)
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
            image_bgr = image_bgr  # already loaded
            data = read_text_json(json_path)
            bubbles_for_mask = data.get("bubbles", [])
            text_mask = build_text_inpaint_mask(image_bgr, masks_dir, bubbles_for_mask)
            save_png(text_mask_path, cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR))
        # Choose mask: text_mask if non-empty else combined
        use_mask_path = combined_mask_path
        text_mask_gray = cv2.imread(str(text_mask_path), cv2.IMREAD_GRAYSCALE)
        if text_mask_gray is not None and int(text_mask_gray.sum()) > 0:
            use_mask_path = text_mask_path
        result_bgr = run_inpainting(input_image_path, use_mask_path)
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
                )
            )
        render_typeset(
            cleaned_path=cleaned_path,
            output_final_path=final_path,
            records=records,
            font_path=font,
            margin_px=6,
            debug=debug,
            debug_overlay_path=typeset_debug_path if debug else None,
        )
        stage_completed.append("typeset")
        _update("typeset_complete", 1.0)

    # Build result payload
    num_bubbles = len(bubbles)
    result: Dict[str, Any] = {
        "task_id": task_id,
        "stage_completed": stage_completed,
        "width": int(width),
        "height": int(height),
        "num_bubbles": int(num_bubbles),
        "json_url": _artifact_url(task_id, "text.json"),
        "overlay_url": _artifact_url(task_id, "segmentation_overlay.png"),
    }
    if cleaned_path.exists():
        result["cleaned_url"] = _artifact_url(task_id, "cleaned.png")
    if final_path.exists():
        result["final_url"] = _artifact_url(task_id, "final.png")
    if debug and typeset_debug_path.exists():
        result["typeset_debug_url"] = _artifact_url(task_id, "typeset_debug.png")

    logger.info("task_completed", extra={"task": "process_page_task", "result": result})
    return result


__all__ = ["demo_task", "process_page_task"]


