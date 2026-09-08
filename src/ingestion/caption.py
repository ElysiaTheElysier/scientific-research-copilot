"""Multimodal Figure Captioning Module.

Uses an advanced Vision LLM (default: llava:13b) grounded with the paper's
captions and in-text references to capture the full academic and architectural
meaning of figures for high-precision RAG retrieval.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ACADEMIC_FIGURE_PROMPT = """You are an expert scientific researcher and peer reviewer. Analyze this figure in deep academic detail using the provided context from the paper.

Context from the paper:
- Paper ID: {paper_id}
- Section: {section}
- Figure Caption: {caption}
- In-Text Discussion from paper body: {in_text_context}

Provide a comprehensive, evidence-grounded academic analysis for a semantic search database:
1. Conceptual & Architectural Flow: Trace the components, data flows, inputs, outputs, and modules shown in the visual.
2. Grounding in Paper Methodology: Explain how this visual directly supports the authors' technical claims and design choices.
3. Visible Annotations & Labels: Identify and explain all visible labels, tables, parameters, gates, or metrics.
4. Core Empirical / Architectural Takeaway: State concisely what key architecture, mechanism, or empirical finding this figure demonstrates.

Be specific, dense in technical terminology, and faithful to both the visual and the text context."""


def extract_figure_context(
    paper_id: str,
    figure_index: int,
    markdown_text: str,
    known_caption: str = "",
) -> tuple[str, str, str]:
    """Extract figure caption, section name, and in-text discussion paragraphs from paper Markdown.

    Args:
        paper_id: Paper identifier.
        figure_index: Zero-based figure index.
        markdown_text: Converted Markdown content of the paper.
        known_caption: Caption if already extracted during Docling conversion.

    Returns:
        Tuple of (caption, section_name, in_text_context)
    """
    fig_num = figure_index + 1
    caption = known_caption.strip()

    # If caption not provided, extract via regex from Markdown
    if not caption:
        cap_patterns = [
            rf"(?:^|\n)(Figure\s+{fig_num}[:\.\s][^\n]+(?:\n(?![#\n])[^\n]+)*)",
            rf"(?:^|\n)(Fig\.\s*{fig_num}[:\.\s][^\n]+(?:\n(?![#\n])[^\n]+)*)",
        ]
        for pattern in cap_patterns:
            match = re.search(pattern, markdown_text, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                break

    # Extract in-text mentions (paragraphs mentioning Figure X that are not the caption itself)
    paragraphs = markdown_text.split("\n\n")
    mentions = []
    fig_regex = rf"\b(?:Figure|Fig\.)\s*{fig_num}\b"

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if re.search(fig_regex, p_clean, re.IGNORECASE):
            # Exclude paragraph if it is essentially the caption
            if caption and caption[:60].lower() in p_clean.lower():
                continue
            # Keep clean paragraph text
            mentions.append(" ".join(p_clean.split()))

    in_text_context = "\n\n".join(mentions[:3]) if mentions else "Not explicitly discussed in nearby body text."

    # Identify the section where the figure or its discussion appears
    section_name = "Body"
    if caption:
        cap_pos = markdown_text.find(caption[:60])
        if cap_pos != -1:
            # Find the closest preceding ## header
            header_matches = list(re.finditer(r"^##\s+([^\n]+)", markdown_text[:cap_pos], re.MULTILINE))
            if header_matches:
                section_name = header_matches[-1].group(1).strip()

    return caption, section_name, in_text_context


def caption_single_figure(
    img_path: Path,
    paper_id: str,
    caption: str = "",
    section: str = "",
    in_text_context: str = "",
    model: str = "llava:13b",
) -> str:
    """Generate a deep academic description of a figure using a Vision LLM with surrounding paper context."""
    prompt = ACADEMIC_FIGURE_PROMPT.format(
        paper_id=paper_id,
        section=section or "General",
        caption=caption or "No caption provided in manuscript.",
        in_text_context=in_text_context or "No direct discussion found in text.",
    )
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [str(img_path)],
        }],
    )
    return response["message"]["content"]


def caption_all_figures(
    images_dir: Path,
    output_file: Path,
    processed_dir: Optional[Path] = None,
    model: str = "llava:13b",
    overwrite: bool = False,
) -> list[dict]:
    """Caption all figures with paper context, incrementally saving descriptions to JSON.

    Args:
        images_dir: Directory containing figure images (*.png).
        output_file: Destination JSON file for figure descriptions.
        processed_dir: Directory containing Markdown papers and figures_meta.json.
        model: Ollama vision model (default: llava:13b).
        overwrite: If True, re-caption all images.

    Returns:
        List of all image description dicts.
    """
    images_dir = Path(images_dir)
    output_file = Path(output_file)
    processed_dir = Path(processed_dir) if processed_dir else output_file.parent
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing descriptions for incremental processing
    existing_by_id = {}
    if output_file.exists() and not overwrite:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                existing_by_id = {r["image_id"]: r for r in records}
            logger.info(f"Loaded {len(existing_by_id)} existing descriptions from {output_file.name}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing {output_file}, starting fresh.")

    # Load known figures metadata if available
    meta_by_id = {}
    meta_file = processed_dir / "figures_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    meta_by_id[item["image_id"]] = item
        except Exception as e:
            logger.warning(f"Could not read {meta_file}: {e}")

    # Cache Markdown texts to avoid repeated file reads
    markdown_cache = {}
    for md_file in processed_dir.glob("*.md"):
        markdown_cache[md_file.stem] = md_file.read_text(encoding="utf-8")

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        logger.warning(f"No image files found in {images_dir}")
        return list(existing_by_id.values())

    results = []
    processed_count = 0

    for img_path in image_files:
        image_id = img_path.name
        # Parse paper_id and figure index
        parts = image_id.replace(".png", "").split("_figure_")
        paper_id = parts[0]
        try:
            figure_idx = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            figure_idx = 0

        # Skip if already captioned with context and not overwriting
        if image_id in existing_by_id and not overwrite:
            rec = existing_by_id[image_id]
            # If previous record already has caption and deep description, keep it
            if rec.get("caption") and rec.get("in_text_context"):
                results.append(rec)
                continue

        # Extract paper context
        known_caption = meta_by_id.get(image_id, {}).get("caption", "")
        md_text = markdown_cache.get(paper_id, "")
        caption, section, in_text_context = extract_figure_context(
            paper_id=paper_id,
            figure_index=figure_idx,
            markdown_text=md_text,
            known_caption=known_caption,
        )

        logger.info(f"Captioning {image_id} with {model} (Paper: {paper_id}, Fig {figure_idx + 1})...")
        try:
            desc = caption_single_figure(
                img_path=img_path,
                paper_id=paper_id,
                caption=caption,
                section=section,
                in_text_context=in_text_context,
                model=model,
            )
            record = {
                "image_id": image_id,
                "paper_id": paper_id,
                "figure_index": figure_idx,
                "section": section,
                "caption": caption,
                "in_text_context": in_text_context,
                "description": desc,
                "model": model,
            }
            results.append(record)
            existing_by_id[image_id] = record
            processed_count += 1

            # Intermittently write to disk so progress is saved
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to caption {image_id}: {e}")
            if image_id in existing_by_id:
                results.append(existing_by_id[image_id])

    # Final write
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} figure descriptions ({processed_count} newly generated) to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate context-grounded figure descriptions using Vision LLMs.")
    parser.add_argument("--images-dir", type=str, default="data/processed/images", help="Path to figures directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed Markdown directory")
    parser.add_argument("--output", type=str, default="data/processed/image_descriptions.json", help="Path to output JSON")
    parser.add_argument("--model", type=str, default="llava:13b", help="Ollama vision model name (default: llava:13b)")
    parser.add_argument("--overwrite", action="store_true", help="Re-caption all images even if already present")

    args = parser.parse_args()
    caption_all_figures(
        images_dir=Path(args.images_dir),
        output_file=Path(args.output),
        processed_dir=Path(args.processed_dir),
        model=args.model,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
