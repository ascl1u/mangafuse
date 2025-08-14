# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
import cv2

# --- Path Setup ---
# Ensure the backend package root is on sys.path when running from repo root
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# --- Local Imports ---
from app.pipeline.utils.io import ensure_dir, read_image_bgr, save_png
from app.pipeline.utils.masks import save_masks
from app.pipeline.segmentation.yolo import run_segmentation
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

# --- Configuration & State ---

@dataclass
class PipelineConfig:
    """A centralized configuration object for the pipeline."""
    image_path: Path
    out_dir: Path
    seg_model_path: Path
    font_path: Path
    stage: str
    debug: bool
    force: bool
    use_placeholder_text: bool

    # Derived Paths
    masks_dir: Path = field(init=False)
    json_path: Path = field(init=False)
    overlay_path: Path = field(init=False)
    combined_mask_path: Path = field(init=False)
    text_mask_path: Path = field(init=False)
    cleaned_path: Path = field(init=False)
    final_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths after the main fields are set."""
        self.masks_dir = self.out_dir / "masks"
        self.json_path = self.out_dir / "text.json"
        self.overlay_path = self.out_dir / "segmentation_overlay.png"
        self.combined_mask_path = self.masks_dir / "all_mask.png"
        self.text_mask_path = self.masks_dir / "text_mask.png"
        self.cleaned_path = self.out_dir / "cleaned.png"
        self.final_path = self.out_dir / "final.png"

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaFuse AI Pipeline (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input page image")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["seg", "ocr", "translate", "inpaint", "typeset", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(Path("assets/models/model.pt")),
        help="Path to local YOLOv8 segmentation weights (.pt)",
    )
    parser.add_argument("--font", type=str, default=str(Path("assets/fonts/animeace2_reg.ttf")), help="Path to TTF font")
    parser.add_argument("--debug", action="store_true", help="Save additional debug outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--use-placeholder-text", action="store_true", help="Use placeholder text for typesetting")
    return parser.parse_args()


# --- Pipeline Stages ---

def run_segmentation_stage(config: PipelineConfig, image_bgr: np.ndarray) -> Dict[str, Any]:
    """Detects speech bubbles and returns their data in-memory."""
    print("--- Running Stage: Segmentation ---")
    
    print(f"Running bubble segmentation on {config.image_path.name}...")
    result = run_segmentation(image_bgr=image_bgr, seg_model_path=config.seg_model_path)
    
    bubbles = [{"id": i, "polygon": p} for i, p in enumerate(result.get("polygons", []), 1)]
    result["bubbles"] = bubbles
    
    if config.debug:
        print("Debug mode: Saving segmentation artifacts to disk...")
        h, w = image_bgr.shape[:2]
        save_masks(config.masks_dir, result.get("masks", []), config.combined_mask_path, image_height=h, image_width=w)
        save_text_records(config.json_path, bubbles)
        overlay_bgr = make_overlay(image_bgr, result.get("masks", []), result.get("polygons", []), result.get("confidences", []))
        save_png(config.overlay_path, overlay_bgr)

    print(f"Segmentation complete. Found {len(result.get('polygons', []))} bubbles.")
    return result

def run_ocr_stage(config: PipelineConfig, image_bgr: np.ndarray, seg_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs OCR on in-memory bubble data."""
    print("--- Running Stage: OCR ---")
    
    bubbles = seg_result.get("bubbles", [])
    masks = seg_result.get("masks", [])
    if not bubbles:
        print("No bubbles found, skipping OCR.")
        return []

    ocr_engine = MangaOcrEngine()
    updated_count = 0
    
    print(f"Performing OCR on {len(bubbles)} bubbles...")
    for i, rec in enumerate(bubbles):
        if not (rec.get("ja_text") is None or config.force):
            continue
        
        mask_array = masks[i]
        polygon = rec.get("polygon", [])
        
        crop_bgr, _ = tight_crop_from_mask(image_bgr, mask_array, polygon)
        
        try:
            ja_text = ocr_engine.run(binarize_for_ocr(crop_bgr))
        except Exception:
            ja_text = ocr_engine.run(crop_bgr)
            
        rec["ja_text"] = ja_text
        updated_count += 1
        print(f"   - Bubble {rec['id']}: {ja_text[:30]}...")

    if config.debug:
        print("Debug mode: Saving updated text records to disk...")
        save_text_records(config.json_path, bubbles)

    print(f"OCR complete. Updated {updated_count} bubbles.")
    return bubbles

def run_translation_stage(config: PipelineConfig, bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translates Japanese text from in-memory bubble data."""
    print("--- Running Stage: Translation ---")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    if not bubbles:
        print("No bubbles to translate.")
        return []

    texts_to_translate = []
    indices_to_update = []
    
    for i, rec in enumerate(bubbles):
        ja_text = (rec.get("ja_text") or "").strip()
        if ja_text and (rec.get("en_text") is None or config.force):
            texts_to_translate.append(ja_text)
            indices_to_update.append(i)

    if not texts_to_translate:
        print("Skipping translation: All bubbles already have English text. Use --force to re-translate.")
        return bubbles
        
    print(f"Translating text for {len(texts_to_translate)} bubbles...")
    translator = GeminiTranslator(api_key=api_key)
    translated_texts = translator.translate_batch(texts_to_translate)

    for i, en_text in zip(indices_to_update, translated_texts):
        bubbles[i]["en_text"] = en_text
        print(f"   - Bubble {bubbles[i]['id']}: \"{bubbles[i]['ja_text']}\" -> \"{en_text}\"")

    if config.debug:
        print("Debug mode: Saving updated text records to disk...")
        save_text_records(config.json_path, bubbles)

    print(f"Translation complete. Generated {len(translated_texts)} English translations.")
    return bubbles

def run_inpaint_stage(config: PipelineConfig, image_bgr: np.ndarray, seg_result: Dict[str, Any], bubble_data: List[Dict[str, Any]]) -> np.ndarray:
    """Removes original text from the image using an in-memory workflow."""
    print("--- Running Stage: Inpainting ---")
    
    if not config.force and config.cleaned_path.exists():
        print(f"Skipping inpainting: {config.cleaned_path.name} already exists. Use --force to override.")
        return read_image_bgr(config.cleaned_path)

    print("Building precise text mask for inpainting...")
    text_mask = build_text_inpaint_mask(
        image_bgr=image_bgr,
        instance_masks=seg_result.get("masks", []),
        bubbles=bubble_data
    )
    
    if config.debug:
        print(f"Debug mode: Saving text inpaint mask to {config.text_mask_path}")
        save_png(config.text_mask_path, text_mask)

    use_mask = text_mask
    if use_mask.sum() == 0:
        print("Text mask is empty, falling back to combined bubble mask for inpainting.")
        masks = seg_result.get("masks", [])
        if masks:
            combined_mask = np.maximum.reduce(masks).astype(np.uint8) * 255
            use_mask = combined_mask
    
    print(f"Running inpainting on {config.image_path.name}...")
    cleaned_bgr = run_inpainting(image_bgr, use_mask)
    
    save_png(config.cleaned_path, cleaned_bgr)
    
    print(f"Inpainting complete. Saved cleaned image to {config.cleaned_path}")
    return cleaned_bgr

def run_typeset_stage(config: PipelineConfig, cleaned_bgr: np.ndarray, bubble_data: List[Dict[str, Any]]):
    """Renders the translated text back into the bubbles using an in-memory image."""
    print("--- Running Stage: Typesetting ---")

    records = []
    if config.use_placeholder_text:
        print("Using placeholder text for typesetting.")
        records = [
            BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text="Placeholder Text")
            for rec in bubble_data
        ]
    else:
        for rec in bubble_data:
            text = (rec.get("en_text") or rec.get("ja_text") or "").strip()
            records.append(
                BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text=text)
            )

    print(f"Typesetting text for {len(records)} bubbles...")
    debug_overlay_path = config.out_dir / "typeset_debug.png" if config.debug else None
    
    # --- REFACTOR: Pass the in-memory NumPy array directly ---
    render_typeset(
        cleaned_image_bgr=cleaned_bgr,
        output_final_path=config.final_path,
        records=records,
        font_path=config.font_path,
        debug=config.debug,
        debug_overlay_path=debug_overlay_path,
    )

    print(f"Typesetting complete. Saved final image to {config.final_path}")
    if config.debug and debug_overlay_path:
        print(f"Saved debug typesetting overlay to {debug_overlay_path}")

# --- Main Orchestrator ---

def main():
    """Main function to parse arguments and run the selected pipeline stage(s)."""
    start_time = time.time()
    args = parse_args()

    backend_env = _BACKEND_DIR / ".env"
    if backend_env.exists():
        load_dotenv(backend_env, override=False)
    load_dotenv(override=False)

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    config = PipelineConfig(
        image_path=Path(args.image),
        out_dir=Path(args.out_dir),
        seg_model_path=Path(args.seg_model),
        font_path=Path(args.font),
        stage=args.stage,
        debug=args.debug,
        force=args.force,
        use_placeholder_text=args.use_placeholder_text
    )
    ensure_dir(config.out_dir)
    ensure_dir(config.masks_dir)
    
    print(f"Starting MangaFuse AI Pipeline for '{config.image_path.name}'...")
    print(f"Output directory: '{config.out_dir}'")
    print(f"Stage(s) to run: '{config.stage}'")

    # --- In-Memory State ---
    image_bgr = read_image_bgr(config.image_path)
    seg_result = {}
    bubble_data = []
    cleaned_bgr = None
    
    # --- Stage-based execution ---
    stages_to_run = ["seg", "ocr", "translate", "inpaint", "typeset"]
    try:
        start_index = stages_to_run.index(config.stage)
    except ValueError:
        start_index = 0 # 'all'

    active_stages = stages_to_run[start_index:] if config.stage != "all" else stages_to_run

    # Run stages sequentially, passing data in memory
    if "seg" in active_stages:
        seg_result = run_segmentation_stage(config, image_bgr)
    
    if "ocr" in active_stages:
        if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
        bubble_data = run_ocr_stage(config, image_bgr, seg_result)

    if "translate" in active_stages:
        if not bubble_data:
            if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
            bubble_data = run_ocr_stage(config, image_bgr, seg_result)
        bubble_data = run_translation_stage(config, bubble_data)

    if "inpaint" in active_stages:
        if not bubble_data:
            if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
            bubble_data = run_ocr_stage(config, image_bgr, seg_result)
            bubble_data = run_translation_stage(config, bubble_data)
        cleaned_bgr = run_inpaint_stage(config, image_bgr, seg_result, bubble_data)
    
    if "typeset" in active_stages:
        if cleaned_bgr is None:
            if not bubble_data:
                if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
                bubble_data = run_ocr_stage(config, image_bgr, seg_result)
                bubble_data = run_translation_stage(config, bubble_data)
            cleaned_bgr = run_inpaint_stage(config, image_bgr, seg_result, bubble_data)
        
        # If running typesetting as a standalone stage, we may still need text.json for debug
        if config.debug or not bubble_data:
             save_text_records(config.json_path, bubble_data)

        run_typeset_stage(config, cleaned_bgr, bubble_data)

    end_time = time.time()
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()