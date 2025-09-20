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
from app.pipeline.model_registry import ModelRegistry # 👈 Add this import
from app.core.paths import get_assets_root # 👈 Add this import

@dataclass
class ScriptConfig:
    """A centralized configuration object for the command-line script."""
    image_path: Path
    out_dir: Path
    seg_model_path: Path
    font_path: Path
    run_mode: str # 'full' or 'cleaned'
    debug: bool
    force: bool
    models: ModelRegistry # 👈 Add this to the config

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaFuse AI Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image or a folder of images")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "cleaned"],
        help="Pipeline depth: 'full' runs all stages, 'cleaned' stops after inpainting.",
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
    return parser.parse_args()


def process_image(config: ScriptConfig):
    """
    Processes a single image by invoking the centralized pipeline orchestrator.
    
    This function is now a simple client. It gathers configuration from the
    command line and passes it to the single source of truth: `run_pipeline`.
    It contains no complex conditional logic about how the pipeline stages run.
    """
    print(f"Starting MangaFuse AI Pipeline for '{config.image_path.name}'...")
    print(f"Output directory: '{config.out_dir}'")
    print(f"Run mode: '{config.run_mode}'")
    
    # The job_id can be derived from the output path for script-based runs.
    # In a web app, this would be a unique ID (e.g., from a database or UUID).
    job_id = config.out_dir.name

    try:
        result = run_pipeline(
            job_id=job_id,
            image_path=str(config.image_path),
            depth=config.run_mode,
            debug=config.debug,
            force=config.force,
            seg_model_path=str(config.seg_model_path),
            font_path=str(config.font_path),
            job_dir_override=str(config.out_dir),
            models=config.models, # 👈 Pass the preloaded models object
        )
        print("Pipeline execution successful.")
        print(f"Completed stages: {result.get('stage_completed')}")
        
        # Now, the path in the result will match the script's output directory
        if result.get('paths', {}).get('final'):
            print(f"Final image saved to: {result['paths']['final']}")
        elif result.get('paths', {}).get('cleaned'):
            print(f"Cleaned image saved to: {result['paths']['cleaned']}")

    except Exception as e:
        print(f"\n!!! PIPELINE FAILED for {config.image_path.name} !!!")
        print(f"Error: {e}")
        # In a real application, use proper logging with tracebacks.
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

    # 🚀 Load all models once before the processing loop begins
    print("Preloading models...")
    assets_root = get_assets_root()
    models = ModelRegistry.load(seg_model_path=assets_root / "models" / "model.pt")
    print("Models loaded successfully.")

    for i, image_path in enumerate(image_paths):
        image_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING IMAGE {i+1}/{total_images}: {image_path.name}")
        print(f"{'='*80}")

        # Each image gets its own output subdirectory within the main out-dir
        image_out_dir = Path(args.out_dir) / image_path.stem
        ensure_dir(image_out_dir)

        config = ScriptConfig(
            image_path=image_path,
            out_dir=image_out_dir,
            seg_model_path=Path(args.seg_model),
            font_path=Path(args.font),
            run_mode=args.mode,
            debug=args.debug,
            force=args.force,
            models=models, # 👈 Pass the preloaded models object
        )
        
        process_image(config)

        image_end_time = time.time()
        print(f"\nFinished processing {image_path.name} in {image_end_time - image_start_time:.2f} seconds.")

    overall_end_time = time.time()
    print(f"\n{'='*80}")
    print(f"Pipeline finished processing all {total_images} image(s) in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()