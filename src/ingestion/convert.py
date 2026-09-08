"""PDF Ingestion and Document Conversion Module.

Converts PDF papers into Markdown and extracts figures and their captions
in a single pass using Docling.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_converter() -> DocumentConverter:
    """Initialize a Docling DocumentConverter configured for PDF parsing and image extraction."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )


def convert_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    images_dir: Optional[Path] = None,
    converter: Optional[DocumentConverter] = None,
    overwrite: bool = False,
) -> dict:
    """Convert a single PDF into Markdown and extract embedded figures in a single pass.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where the Markdown file will be saved.
        images_dir: Directory where figures will be saved (defaults to output_dir / "images").
        converter: Existing DocumentConverter instance (created if None).
        overwrite: If False, skip conversion if output Markdown already exists.

    Returns:
        Dict with conversion summary: paper_id, md_path, figures_count, and figures_meta.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir) if images_dir else output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    paper_id = pdf_path.stem
    md_path = output_dir / f"{paper_id}.md"

    if md_path.exists() and not overwrite:
        logger.info(f"Skipping {pdf_path.name} (already converted: {md_path.name})")
        existing_figures = sorted(images_dir.glob(f"{paper_id}_figure_*.png"))
        return {
            "paper_id": paper_id,
            "md_path": str(md_path),
            "figures_count": len(existing_figures),
            "figures_meta": [
                {
                    "image_id": f.name,
                    "paper_id": paper_id,
                    "figure_index": idx,
                    "image_path": str(f),
                }
                for idx, f in enumerate(existing_figures)
            ],
            "skipped": True,
        }

    logger.info(f"Converting {pdf_path.name}...")
    if converter is None:
        converter = create_converter()

    result = converter.convert(pdf_path)
    doc = result.document

    # 1. Export Markdown with image placeholders
    md_content = doc.export_to_markdown(image_mode="placeholder")
    md_path.write_text(md_content, encoding="utf-8")

    # 2. Extract and save pictures + captions
    figures_meta = []
    figure_count = 0
    for item, _ in doc.iterate_items():
        if item.label == "picture":
            img = item.get_image(doc)
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            if img:
                image_id = f"{paper_id}_figure_{figure_count}.png"
                img_path = images_dir / image_id
                img.save(str(img_path))
                figures_meta.append({
                    "image_id": image_id,
                    "paper_id": paper_id,
                    "figure_index": figure_count,
                    "caption": caption.strip(),
                    "image_path": str(img_path),
                })
                figure_count += 1

    logger.info(f"Converted {paper_id}: saved markdown and {figure_count} figures.")
    return {
        "paper_id": paper_id,
        "md_path": str(md_path),
        "figures_count": figure_count,
        "figures_meta": figures_meta,
        "skipped": False,
    }


def convert_all_pdfs(
    raw_dir: Path,
    output_dir: Path,
    images_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> list[dict]:
    """Convert all PDF files in raw_dir to Markdown and extract figures.

    Args:
        raw_dir: Directory containing raw PDF files.
        output_dir: Target directory for processed Markdown files.
        images_dir: Target directory for extracted figure PNGs.
        overwrite: Whether to overwrite existing files.

    Returns:
        List of conversion summaries.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir) if images_dir else output_dir / "images"

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {raw_dir}")
    converter = create_converter()
    results = []
    all_figures_meta = []

    for pdf_path in pdf_files:
        res = convert_single_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            images_dir=images_dir,
            converter=converter,
            overwrite=overwrite,
        )
        results.append(res)
        all_figures_meta.extend(res.get("figures_meta", []))

    # Save figures metadata index if newly generated
    meta_path = output_dir / "figures_meta.json"
    if all_figures_meta and (not meta_path.exists() or overwrite):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_figures_meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved figures metadata index to {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert scientific PDF papers to Markdown and extract figures.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw PDF directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to processed directory")
    parser.add_argument("--images-dir", type=str, default=None, help="Path to images directory (default: output_dir/images)")
    parser.add_argument("--overwrite", action="store_true", help="Force re-conversion of existing documents")

    args = parser.parse_args()
    convert_all_pdfs(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
