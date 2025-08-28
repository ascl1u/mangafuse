from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
import logging
import time


class PipelineOrchestrator:
    """
    Manages the state and execution of the multi-stage AI pipeline for a single image.

    This class provides a structured, reusable, and testable implementation of the
    pipeline logic, acting as the single source of truth for the system.
    """

    def __init__(
        self,
        job_id: str,
        image_path: str,
        *,
        depth: Literal["cleaned", "full"] = "cleaned",
        debug: bool = False,
        force: bool = False,
        seg_model_path: Optional[str] = None,
        font_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        include_typeset: bool = True,
        include_translate: bool = True,
        job_dir_override: Optional[str | Path] = None,
        models: Any | None = None,
    ):
        # Configuration
        self.job_id = job_id
        self.input_image_path = Path(image_path)
        self.depth = depth
        self.debug = debug
        self.force = force
        self.progress_callback = progress_callback
        self.include_typeset = include_typeset
        self.include_translate = include_translate

        # State - initialized lazily
        self.image_bgr: Optional[Any] = None
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.bubbles: List[Dict[str, Any]] = []
        self.stage_completed: List[str] = []

        # Path Setup
        if job_dir_override:
            self.job_dir = Path(job_dir_override)
        else:
            from app.core.paths import get_job_dir
            self.job_dir = get_job_dir(self.job_id)

        from app.core.paths import get_assets_root
        self.masks_dir = self.job_dir / "masks"
        self.crops_dir = self.job_dir / "crops"

        assets_root = get_assets_root()
        default_seg_model = assets_root / "models" / "model.pt"
        default_font = assets_root / "fonts" / "animeace2_reg.ttf"
        self.seg_model = Path(seg_model_path) if seg_model_path else default_seg_model
        self.font = Path(font_path) if font_path else default_font
        self.models = models

        # Artifact Paths
        self.overlay_path = self.job_dir / "segmentation_overlay.png"
        self.combined_mask_path = self.masks_dir / "all_mask.png"
        self.text_mask_path = self.masks_dir / "text_mask.png"
        self.json_path = self.job_dir / "text.json"
        self.cleaned_path = self.job_dir / "cleaned.png"
        self.final_path = self.job_dir / "final.png"
        self.typeset_debug_path = self.job_dir / "typeset_debug.png"
        self.editor_payload_path = self.job_dir / "editor_payload.json"

    def _update_progress(self, stage: str, progress: float) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(stage, progress)
            except Exception:
                logging.getLogger(__name__).exception(
                    "progress_callback_failed", extra={"job_id": self.job_id, "stage": stage}
                )

    def _load_initial_data(self) -> None:
        """Loads the source image and prepares initial directories."""
        from app.pipeline.utils.io import ensure_dir, read_image_bgr

        if not self.input_image_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.input_image_path}")

        ensure_dir(self.job_dir)
        ensure_dir(self.masks_dir)

        self.image_bgr = read_image_bgr(self.input_image_path)
        self.height, self.width = self.image_bgr.shape[:2]

    def _run_segmentation(self) -> None:
        """Stage 1: Detects speech bubbles and saves polygons and masks."""
        import cv2
        import numpy as np
        from app.pipeline.segmentation.yolo import run_segmentation
        from app.pipeline.utils.io import save_png
        from app.pipeline.utils.masks import save_masks
        from app.pipeline.utils.textio import read_text_json, write_text_json
        from app.pipeline.utils.visualization import make_overlay

        self._update_progress("segmentation", 0.1)
        t0 = time.perf_counter()
        if self.force or not (self.overlay_path.exists() and self.json_path.exists()):
            yolo_model = getattr(self.models, "yolo_model", None) if self.models else None
            result = run_segmentation(image_bgr=self.image_bgr, seg_model_path=self.seg_model, yolo_model=yolo_model)
            polygons = result.get("polygons", [])
            confidences = result.get("confidences", [])

            temp_masks = []
            for poly in polygons:
                m = np.zeros((self.height, self.width), dtype=np.uint8)
                if poly:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(m, [pts], 1)
                temp_masks.append(m)

            overlay_bgr = make_overlay(self.image_bgr, temp_masks, polygons, confidences)
            save_png(self.overlay_path, overlay_bgr)
            save_masks(self.masks_dir, temp_masks, self.combined_mask_path, self.height, self.width)
            write_text_json(self.json_path, polygons)

        self.bubbles = read_text_json(self.json_path).get("bubbles", [])
        self.stage_completed.append("segmentation")
        self._update_progress("segmentation_complete", 0.2)
        t1 = time.perf_counter()
        logging.getLogger(__name__).info(
            "stage_timing",
            extra={
                "job_id": self.job_id,
                "stage": "segmentation",
                "ms": int((t1 - t0) * 1000),
                "num_bubbles": len(self.bubbles),
            },
        )

    def _run_ocr(self) -> None:
        """Stage 2: Extracts Japanese text from segmented bubbles."""
        if self.depth != "full" or not self.bubbles:
            return

        import cv2
        import numpy as np
        from app.pipeline.ocr.crops import tight_crop_from_mask
        from app.pipeline.ocr.engine import MangaOcrEngine
        from app.pipeline.ocr.preprocess import binarize_for_ocr
        from app.pipeline.utils.textio import save_text_records

        self._update_progress("ocr", 0.35)
        t0 = time.perf_counter()
        if self.models and getattr(self.models, "ocr_engine", None) is not None:
            ocr_engine = self.models.ocr_engine
        else:
            ocr_engine = MangaOcrEngine()

        for rec in self.bubbles:
            has_ja = isinstance(rec.get("ja_text"), str) and rec["ja_text"].strip()
            if has_ja and not self.force:
                continue

            polygon = rec.get("polygon") or []
            mask_gray = np.zeros((self.height, self.width), dtype=np.uint8)
            if polygon:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask_gray, [pts], 1)

            crop_bgr, _ = tight_crop_from_mask(self.image_bgr, mask_gray, polygon)
            try:
                ja_text = ocr_engine.run(binarize_for_ocr(crop_bgr))
            except Exception:
                logging.getLogger(__name__).exception(
                    "ocr_binarize_failed", extra={"job_id": self.job_id, "bubble_id": int(rec.get("id") or -1)}
                )
                ja_text = ocr_engine.run(crop_bgr) # Fallback
            rec["ja_text"] = ja_text

        save_text_records(self.json_path, self.bubbles)
        self.stage_completed.append("ocr")
        self._update_progress("ocr_complete", 0.45)
        t1 = time.perf_counter()
        logging.getLogger(__name__).info(
            "stage_timing",
            extra={
                "job_id": self.job_id,
                "stage": "ocr",
                "ms": int((t1 - t0) * 1000),
                "num_bubbles": len(self.bubbles),
            },
        )

    def _run_translation(self) -> None:
        """Stage 3: Translates Japanese text to English using an external API."""
        if self.depth != "full" or not self.include_translate or not self.bubbles:
            return

        from app.pipeline.translate.gemini import GeminiTranslator
        from app.pipeline.utils.textio import save_text_records

        self._update_progress("translate", 0.5)
        api_key = os.getenv("GOOGLE_API_KEY", "")
        translator = GeminiTranslator(api_key=api_key)
        indices_to_translate: list[int] = []
        texts_to_translate: list[str] = []
        try:
            for idx, rec in enumerate(self.bubbles):
                ja_text = (rec.get("ja_text") or "").strip()
                has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip()
                if ja_text and (not has_en or self.force):
                    indices_to_translate.append(idx)
                    texts_to_translate.append(ja_text)

            t0 = time.perf_counter()
            if texts_to_translate:
                en_list = translator.translate_batch(texts_to_translate)
                for i, en in zip(indices_to_translate, en_list):
                    self.bubbles[i]["en_text"] = en
                save_text_records(self.json_path, self.bubbles)

            self.stage_completed.append("translate")
            self._update_progress("translate_complete", 0.6)
            t1 = time.perf_counter()
            logging.getLogger(__name__).info(
                "stage_timing",
                extra={
                    "job_id": self.job_id,
                    "stage": "translate",
                    "ms": int((t1 - t0) * 1000),
                    "num_translated": len(texts_to_translate),
                },
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "translate_failed",
                extra={
                    "job_id": self.job_id,
                    "num_to_translate": len(texts_to_translate),
                },
            )
            raise

    def _run_inpaint(self) -> None:
        """Stage 4: Removes original text from the image, creating a 'cleaned' version."""
        import cv2
        import numpy as np
        from app.pipeline.inpaint.lama import run_inpainting
        from app.pipeline.inpaint.text_mask import build_text_inpaint_mask
        from app.pipeline.utils.io import read_image_bgr, save_png

        self._update_progress("inpaint", 0.7)
        t0 = time.perf_counter()

        if self.force or not self.text_mask_path.exists():
            imasks = []
            for rec in self.bubbles:
                poly = rec.get("polygon") or []
                m = np.zeros((self.height, self.width), dtype=np.uint8)
                if poly:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(m, [pts], 1)
                imasks.append(m)
            text_mask = build_text_inpaint_mask(self.image_bgr, imasks, self.bubbles)
            save_png(self.text_mask_path, text_mask)

        text_mask_img = read_image_bgr(self.text_mask_path)
        text_mask_gray = cv2.cvtColor(text_mask_img, cv2.COLOR_BGR2GRAY)

        if np.any(text_mask_gray):
            mask_for_inpaint = text_mask_gray
        else:
            combined_img = cv2.imread(str(self.combined_mask_path), cv2.IMREAD_UNCHANGED)
            mask_for_inpaint = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY) if combined_img is not None else np.zeros((self.height, self.width), dtype=np.uint8)

        lama_model = getattr(self.models, "lama_model", None) if self.models else None
        result_bgr = run_inpainting(self.image_bgr, mask_for_inpaint, lama_model=lama_model)
        save_png(self.cleaned_path, result_bgr)
        self.stage_completed.append("inpaint")
        self._update_progress("inpaint_complete", 0.85)
        t1 = time.perf_counter()
        logging.getLogger(__name__).info(
            "stage_timing",
            extra={
                "job_id": self.job_id,
                "stage": "inpaint",
                "ms": int((t1 - t0) * 1000),
            },
        )

    def _run_typeset(self) -> None:
        """Stage 5: Renders the translated text onto the cleaned image."""
        if self.depth != "full" or not self.include_typeset:
            return

        from app.pipeline.typeset.model import BubbleText
        from app.pipeline.typeset.render import render_typeset
        from app.pipeline.utils.io import read_image_bgr
        from app.pipeline.utils.textio import save_text_records

        self._update_progress("typeset", 0.9)
        t0 = time.perf_counter()
        try:
            if not self.cleaned_path.exists():
                raise FileNotFoundError(f"cleaned.png not found for job {self.job_id}")

            cleaned_image = read_image_bgr(self.cleaned_path)
            records = [
                BubbleText(
                    bubble_id=int(rec.get("id")),
                    polygon=rec.get("polygon") or [],
                    text=(rec.get("en_text") or rec.get("ja_text") or "").strip(),
                    font_size=(int(rec.get("font_size")) if isinstance(rec.get("font_size"), (int, float)) else None),
                )
                for rec in self.bubbles
            ]

            used_sizes, used_rects = render_typeset(
                cleaned_image_bgr=cleaned_image,
                output_final_path=self.final_path,
                records=records,
                font_path=self.font,
                debug=self.debug,
                debug_overlay_path=self.typeset_debug_path if self.debug else None,
                text_layer_output_path=self.job_dir / "text_layer.png",
            )

            for rec in self.bubbles:
                bid = int(rec.get("id"))
                if used_sizes and bid in used_sizes:
                    rec["font_size"] = int(used_sizes[bid])
                if used_rects and bid in used_rects:
                    rect = used_rects[bid]
                    rec["rect"] = {k: int(v) for k, v in rect.items()}
            save_text_records(self.json_path, self.bubbles)

            self.stage_completed.append("typeset")
            self._update_progress("typeset_complete", 1.0)
            t1 = time.perf_counter()
            logging.getLogger(__name__).info(
                "stage_timing",
                extra={
                    "job_id": self.job_id,
                    "stage": "typeset",
                    "ms": int((t1 - t0) * 1000),
                    "num_bubbles": len(self.bubbles),
                },
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "typeset_failed",
                extra={
                    "job_id": self.job_id,
                    "num_bubbles": len(self.bubbles),
                },
            )
            raise
    
    def _build_final_payload(self) -> Dict[str, Any]:
        """Constructs the final result dictionary with paths and metadata."""
        result: Dict[str, Any] = {
            "job_id": self.job_id,
            "stage_completed": self.stage_completed,
            "width": int(self.width),
            "height": int(self.height),
            "num_bubbles": len(self.bubbles),
            "paths": {
                "json": str(self.json_path),
                "overlay": str(self.overlay_path),
                "cleaned": str(self.cleaned_path) if self.cleaned_path.exists() else None,
                "final": str(self.final_path) if self.final_path.exists() else None,
            },
        }
        return result

    def _generate_editor_assets(self) -> None:
        """Generates bubble crops and the editor_payload.json file for the frontend."""
        import cv2
        import json
        from app.pipeline.utils.io import ensure_dir, save_png
        from app.pipeline.ocr.crops import tight_crop_from_mask

        if not self.bubbles:
            return

        ensure_dir(self.crops_dir)
        normalized_bubbles: List[Dict[str, Any]] = []

        for rec in self.bubbles:
            bubble_id = int(rec["id"])
            polygon = rec.get("polygon") or []
            crop_url = None
            crop_path = self.crops_dir / f"{bubble_id}.png"

            try:
                # Use the individual mask file which should have been saved during segmentation
                mask_path = self.masks_dir / f"{bubble_id}.png"
                if mask_path.exists():
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    # Crop from the original image, not the cleaned one
                    crop_bgr, _ = tight_crop_from_mask(self.image_bgr, mask_gray, polygon)
                    save_png(crop_path, crop_bgr)
                    crop_url = f"/artifacts/jobs/{self.job_id}/crops/{bubble_id}.png"
            except Exception:
                logging.getLogger(__name__).exception(
                    "crop_generation_failed", extra={"job_id": self.job_id, "bubble_id": bubble_id}
                )

            normalized_bubbles.append(
                {
                    "id": bubble_id,
                    "polygon": polygon,
                    "ja_text": rec.get("ja_text"),
                    "en_text": rec.get("en_text"),
                    "font_size": rec.get("font_size"),
                    "rect": rec.get("rect"),
                    "crop_url": crop_url,
                }
            )

        # Use the cleaned image as the main background for the editor
        image_url = f"/artifacts/jobs/{self.job_id}/cleaned.png"

        editor_payload = {
            "image_url": image_url,
            "width": self.width,
            "height": self.height,
            "bubbles": normalized_bubbles,
        }

        with open(self.editor_payload_path, "w", encoding="utf-8") as f:
            json.dump(editor_payload, f, ensure_ascii=False, indent=2)

    def run(self) -> Dict[str, Any]:
        """Executes the full pipeline in the correct order."""
        self._load_initial_data()
        self._run_segmentation()

        if self.depth == "full":
            self._run_ocr()
            self._run_translation()

        self._run_inpaint()

        if self.depth == "full":
            self._run_typeset()
        
        self._generate_editor_assets()

        return self._build_final_payload()


def run_pipeline(
    job_id: str,
    image_path: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    High-level wrapper to execute the full pipeline via the orchestrator class.
    
    This maintains a simple functional API for consumers like Celery tasks.
    """
    orchestrator = PipelineOrchestrator(job_id, image_path, **kwargs)
    return orchestrator.run()


def apply_edits(
    job_dir: Path,
    edits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import json
    import cv2  # type: ignore
    from app.pipeline.utils.textio import read_text_json, save_text_records
    from app.pipeline.typeset.model import BubbleText
    from app.pipeline.typeset.render import render_typeset
    from app.core.paths import get_assets_root

    job_id = job_dir.name
    json_path = job_dir / "text.json"
    cleaned_path = job_dir / "cleaned.png"
    final_path = job_dir / "final.png"
    text_layer_path = job_dir / "text_layer.png"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned image missing for job {job_id}")
    if not json_path.exists():
        raise FileNotFoundError(f"text.json missing for job {job_id}")

    # Persist raw edits for audit/debugging
    with open(job_dir / "edits.json", "w", encoding="utf-8") as f:
        json.dump(edits, f, ensure_ascii=False, indent=2)

    data = read_text_json(json_path)
    bubbles = data.get("bubbles", [])
    
    bubbles_by_id: Dict[int, Dict[str, Any]] = {int(b["id"]): b for b in bubbles if "id" in b}

    for edit in edits:
        try:
            bubble_id = int(edit["id"])
            if bubble_id in bubbles_by_id:
                bubble_record = bubbles_by_id[bubble_id]
                if "en_text" in edit:
                    bubble_record["en_text"] = edit["en_text"]
                if "font_size" in edit:
                    val = edit.get("font_size")
                    if isinstance(val, (int, float)):
                        bubble_record["font_size"] = int(val)
                    elif isinstance(val, str):
                        vs = val.strip()
                        if vs.isdigit():
                            bubble_record["font_size"] = int(vs)
        except (ValueError, KeyError):
            continue
    
    save_text_records(json_path, bubbles)

    font = get_assets_root() / "fonts" / "animeace2_reg.ttf"

    records: List[BubbleText] = [
        BubbleText(
            bubble_id=int(rec.get("id")),
            polygon=rec.get("polygon") or [],
            text=(rec.get("en_text") or rec.get("ja_text") or "").strip(),
            font_size=(int(rec.get("font_size")) if isinstance(rec.get("font_size"), (int, float)) else None),
        )
        for rec in bubbles
    ]
    
    img_cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_COLOR)
    if img_cleaned is None:
        raise FileNotFoundError(f"Failed to read cleaned image for job {job_id}")

    used_sizes, used_rects = render_typeset(
        cleaned_image_bgr=img_cleaned,
        output_final_path=final_path,
        records=records,
        font_path=font,
        debug=False,
        debug_overlay_path=None,
        text_layer_output_path=text_layer_path,
    )

    if used_sizes or used_rects:
        for rec in bubbles:
            bid = int(rec.get("id"))
            if used_sizes and bid in used_sizes:
                rec["font_size"] = int(used_sizes[bid])
            if used_rects and bid in used_rects:
                rect = used_rects[bid]
                rec["rect"] = {k: int(v) for k, v in rect.items()}
        save_text_records(json_path, bubbles)
        
    # Build/update editor payload for the frontend editor and worker persistence
    editor_payload_path = job_dir / "editor_payload.json"
    try:
        normalized: List[Dict[str, Any]] = []
        crops_dir = job_dir / "crops"

        for rec in bubbles:
            bubble_id = int(rec.get("id"))
            polygon = rec.get("polygon") or []
            rect_payload = rec.get("rect") if isinstance(rec.get("rect"), dict) else None
            crop_path = crops_dir / f"{bubble_id}.png"
            crop_url = f"/artifacts/jobs/{job_id}/crops/{bubble_id}.png" if crop_path.exists() else None

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

        # Prefer cleaned image as background for accurate editor preview
        if cleaned_path.exists():
            image_url = f"/artifacts/jobs/{job_id}/cleaned.png"
        elif final_path.exists():
            image_url = f"/artifacts/jobs/{job_id}/final.png"
        else:
            image_url = f"/artifacts/jobs/{job_id}/segmentation_overlay.png"

        # Infer width/height from cleaned image where possible
        width_val: Optional[int] = None
        height_val: Optional[int] = None
        try:
            img = cv2.imread(str(cleaned_path))
            if img is not None:
                h, w = img.shape[:2]
                width_val = int(w)
                height_val = int(h)
        except Exception:
            pass

        editor_payload: Dict[str, Any] = {
            "image_url": image_url,
            "width": width_val,
            "height": height_val,
            "bubbles": normalized,
        }
        with open(editor_payload_path, "w", encoding="utf-8") as f:
            json.dump(editor_payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Do not fail the job if editor payload write has issues
        pass

    # Expose local artifact file paths for the worker to persist via storage
    artifacts: Dict[str, str] = {}
    if final_path.exists():
        artifacts["FINAL_PNG"] = str(final_path)
    if text_layer_path.exists():
        artifacts["TEXT_LAYER_PNG"] = str(text_layer_path)
    if cleaned_path.exists():
        artifacts["CLEANED_PAGE"] = str(cleaned_path)

    result: Dict[str, Any] = {
        "task_id": job_id,
        "final_url": f"/artifacts/jobs/{job_id}/final.png" if final_path.exists() else None,
        "text_layer_url": f"/artifacts/jobs/{job_id}/text_layer.png" if text_layer_path.exists() else None,
        "editor_payload_url": f"/artifacts/jobs/{job_id}/editor_payload.json" if editor_payload_path.exists() else None,
    }
    if artifacts:
        result["artifacts"] = artifacts
    return result