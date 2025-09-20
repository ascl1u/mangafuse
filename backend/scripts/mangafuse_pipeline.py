# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# --- Path Setup ---
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.pipeline.orchestrator import run_pipeline
from app.pipeline.utils.io import ensure_dir
from app.pipeline.model_registry import ModelRegistry
from app.core.paths import get_assets_root

# --- NEW: Script Usage Documentation ---
# This script now has an intelligent two-stage workflow.
#
# STEP 1: GPU Stages (in Docker)
# The script automatically detects it's the first run and tells the pipeline
# to run OCR and inpainting, but skip translation.
#
#   docker run --rm -it --gpus all \
#     -v "./test_output:/test_output" -v "./samples:/samples" \
#     mangafuse \
#     python3 /app/backend/scripts/mangafuse_pipeline.py \
#       --input /samples \
#       --out-dir /test_output --force
#
# STEP 2: CPU Stages (on Host with Conda)
# The script detects the intermediate files and runs only the final
# translation and typesetting stages.
#
#   conda activate your-ai-env
#   python ./backend/scripts/mangafuse_pipeline.py \
#     --input ./samples \
#     --out-dir ./test_output
# ---

@dataclass
class ScriptConfig:
    """A centralized configuration object for the command-line script."""
    image_path: Path
    out_dir: Path
    seg_model_path: Path
    font_path: Path
    debug: bool
    force: bool
    models: ModelRegistry
    # REMOVED run_mode, as the script will now determine the stages itself.

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaFuse AI Pipeline Runner (Hybrid Workflow Enabled)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image or a folder of images")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(Path("assets/models/model.pt")),
        help="Path to local YOLOv8 segmentation weights (.pt)",
    )
    parser.add_argument("--font", type=str, default=str(Path("assets/fonts/animeace2_reg.ttf")), help="Path to TTF font")
    parser.add_argument("--debug", action="store_true", help="Save additional debug outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    return parser.parse_args()


def process_image(config: ScriptConfig):
    """
    Processes a single image by invoking the centralized pipeline orchestrator.
    It automatically detects whether to run GPU-only stages or CPU-finishing stages.
    """
    print(f"Starting MangaFuse AI Pipeline for '{config.image_path.name}'...")
    print(f"Output directory: '{config.out_dir}'")

    job_id = config.out_dir.name

    try:
        # --- MODIFIED: Automatically detect stage and set pipeline flags ---
        cleaned_path = config.out_dir / "cleaned.png"
        json_path = config.out_dir / "text.json"
        
        is_cpu_stage_run = cleaned_path.exists() and json_path.exists() and not config.force

        pipeline_kwargs = {
            "job_id": job_id,
            "image_path": str(config.image_path),
            "debug": config.debug,
            "force": config.force,
            "seg_model_path": str(config.seg_model_path),
            "font_path": str(config.font_path),
            "job_dir_override": str(config.out_dir),
            "models": config.models,
        }

        if is_cpu_stage_run:
            print("Intermediate files found. Running CPU stages: Translation & Typesetting.")
            pipeline_kwargs["depth"] = "full"
            pipeline_kwargs["include_translate"] = True
            pipeline_kwargs["include_typeset"] = True
        else:
            print("Running GPU stages: Segmentation, OCR & Inpainting.")
            # We run in 'full' mode to trigger OCR, but explicitly disable translation
            # to prevent errors inside the GPU container.
            pipeline_kwargs["depth"] = "full"
            pipeline_kwargs["include_translate"] = False
            pipeline_kwargs["include_typeset"] = False
        # --- END MODIFICATION ---

        result = run_pipeline(**pipeline_kwargs)

        print("Pipeline execution successful.")
        print(f"Completed stages: {result.get('stage_completed')}")

        if result.get('paths', {}).get('final'):
            print(f"Final image saved to: {result['paths']['final']}")
        elif result.get('paths', {}).get('cleaned'):
            print(f"Cleaned image saved to: {result['paths']['cleaned']}")

    except Exception as e:
        print(f"\n!!! PIPELINE FAILED for {config.image_path.name} !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to parse arguments and run the pipeline for all images."""
    overall_start_time = time.time()
    args = parse_args()

    load_dotenv(override=False)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_paths = []
    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        supported_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_paths = sorted([p for p in input_path.glob("*") if p.suffix.lower() in supported_extensions])

    if not image_paths:
        print(f"No supported images found in '{input_path}'.")
        return

    total_images = len(image_paths)
    print(f"Found {total_images} image(s) to process.")

    print("Preloading models...")
    assets_root = get_assets_root()
    models = ModelRegistry.load(seg_model_path=assets_root / "models" / "model.pt")
    print("Models loaded successfully.")

    for i, image_path in enumerate(image_paths):
        image_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING IMAGE {i+1}/{total_images}: {image_path.name}")
        print(f"{'='*80}")

        image_out_dir = Path(args.out_dir) / image_path.stem
        ensure_dir(image_out_dir)

        config = ScriptConfig(
            image_path=image_path,
            out_dir=image_out_dir,
            seg_model_path=Path(args.seg_model),
            font_path=Path(args.font),
            debug=args.debug,
            force=args.force,
            models=models,
        )

        process_image(config)

        image_end_time = time.time()
        print(f"\nFinished processing {image_path.name} in {image_end_time - image_start_time:.2f} seconds.")

    overall_end_time = time.time()
    print(f"\n{'='*80}")
    print(f"Pipeline finished processing all {total_images} image(s) in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()