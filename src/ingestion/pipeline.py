"""Unified Ingestion Pipeline Runner.

Orchestrates document conversion, context-aware multimodal figure captioning,
and section-aware / multimodal chunking.

Usage examples:
    # Run all stages end-to-end:
    python -m src.ingestion.pipeline --all

    # Run only document conversion:
    python -m src.ingestion.pipeline --stage convert

    # Run only figure captioning with a specific vision model:
    python -m src.ingestion.pipeline --stage caption --vlm-model llava:13b

    # Run only chunking:
    python -m src.ingestion.pipeline --stage chunk
"""

import argparse
import logging
from pathlib import Path

from src.ingestion.caption import caption_all_figures
from src.ingestion.chunk import chunk_all_documents
from src.ingestion.convert import convert_all_pdfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    stage: str = "all",
    vlm_model: str = "llava:13b",
    overwrite: bool = False,
):
    """Run specified stages of the ingestion pipeline."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    images_dir = processed_dir / "images"
    image_descriptions_path = processed_dir / "image_descriptions.json"
    chunks_path = processed_dir / "chunks.json"

    stages_to_run = ["convert", "caption", "chunk"] if stage == "all" else [stage]

    logger.info(f"Starting pipeline execution for stages: {stages_to_run}")

    if "convert" in stages_to_run:
        logger.info("=== STAGE 1: Document Conversion & Figure Extraction ===")
        convert_all_pdfs(
            raw_dir=raw_dir,
            output_dir=processed_dir,
            images_dir=images_dir,
            overwrite=overwrite,
        )

    if "caption" in stages_to_run:
        logger.info(f"=== STAGE 2: Context-Aware Multimodal Figure Captioning ({vlm_model}) ===")
        caption_all_figures(
            images_dir=images_dir,
            output_file=image_descriptions_path,
            processed_dir=processed_dir,
            model=vlm_model,
            overwrite=overwrite,
        )

    if "chunk" in stages_to_run:
        logger.info("=== STAGE 3: Section-Aware & Multimodal Hybrid Chunking ===")
        chunk_all_documents(
            processed_dir=processed_dir,
            output_file=chunks_path,
            include_figures=True,
        )

    logger.info("Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Scientific Paper Ingestion Pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "convert", "caption", "chunk"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument("--all", dest="stage", action="store_const", const="all", help="Execute all pipeline stages")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw PDF files")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output folder")
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="llava:13b",
        help="Ollama vision model for figure captioning (default: llava:13b)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force re-processing of existing documents/figures")

    args = parser.parse_args()
    run_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        stage=args.stage,
        vlm_model=args.vlm_model,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
