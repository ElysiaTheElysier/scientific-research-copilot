"""Section-Aware and Multimodal Document Chunking Module.

Splits Markdown papers into semantically coherent sections and applies hybrid chunking:
- Short sections (<= 1000 words) remain intact to preserve full context.
- Long sections (> 1000 words) are split with RecursiveCharacterTextSplitter.
- Noise sections (e.g. references) are filtered out.
- Figures with deep visual analyses, captions, and in-text context are integrated
  as first-class multimodal chunks for hybrid RAG retrieval.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SKIP_SECTIONS = {"references"}


def split_sections(markdown_text: str, paper_id: str) -> list[dict]:
    """Split a Markdown document by level-2 headers (## Section Name).

    Lines prior to the first ## header are labeled as 'front_matter'.

    Args:
        markdown_text: Full text of the Markdown file.
        paper_id: Identifier of the paper.

    Returns:
        List of dicts: [{"paper_id": str, "section": str, "content": str}]
    """
    parts = re.split(r"(?=^##\s)", markdown_text, flags=re.MULTILINE)
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        if lines[0].startswith("## "):
            section_title = lines[0][3:].strip()
            content = "\n".join(lines[1:]).strip()
        else:
            section_title = "front_matter"
            content = part

        if content:
            sections.append({
                "paper_id": paper_id,
                "section": section_title,
                "content": content,
            })

    return sections


def create_splitter(chunk_size: int = 3000, chunk_overlap: int = 300) -> RecursiveCharacterTextSplitter:
    """Create a LangChain recursive text splitter configured for scientific text."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_sections(
    sections: list[dict],
    max_words_per_section: int = 1000,
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
    skip_sections: Optional[set[str]] = None,
) -> list[dict]:
    """Apply section-aware hybrid chunking across extracted sections.

    Args:
        sections: List of section dicts with paper_id, section, content.
        max_words_per_section: Maximum words before a section is split recursively.
        chunk_size: Target characters per chunk for long sections.
        chunk_overlap: Character overlap for long sections.
        skip_sections: Set of lowercase section names to exclude (e.g. references).

    Returns:
        List of structured chunk dicts.
    """
    if skip_sections is None:
        skip_sections = DEFAULT_SKIP_SECTIONS

    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for section in sections:
        section_name = section["section"]
        if section_name.lower().strip() in skip_sections:
            continue

        text = section["content"]
        word_count = len(text.split())

        if word_count <= max_words_per_section:
            # Short section: keep intact
            chunks.append({
                "chunk_type": "text",
                "paper_id": section["paper_id"],
                "section": section_name,
                "chunk_index": 0,
                "content": text,
                "word_count": word_count,
                "char_count": len(text),
            })
        else:
            # Long section: split recursively
            sub_texts = splitter.split_text(text)
            for idx, sub_text in enumerate(sub_texts):
                chunks.append({
                    "chunk_type": "text",
                    "paper_id": section["paper_id"],
                    "section": section_name,
                    "chunk_index": idx,
                    "content": sub_text,
                    "word_count": len(sub_text.split()),
                    "char_count": len(sub_text),
                })

    return chunks


def create_figure_chunks(
    image_descriptions: list[dict],
) -> list[dict]:
    """Transform image descriptions into retrieval-ready multimodal chunks.

    Args:
        image_descriptions: List of dicts from image_descriptions.json.

    Returns:
        List of figure chunk dicts.
    """
    figure_chunks = []

    for item in image_descriptions:
        caption = (item.get("caption") or item.get("original_caption") or "").strip()
        section = item.get("section", "Figures").strip()
        raw_context = item.get("in_text_context") or item.get("in_text_mentions", "")
        if isinstance(raw_context, list):
            in_text = " ".join(raw_context).strip()
        else:
            in_text = str(raw_context).strip()
        desc = item.get("description", "").strip()
        paper_id = item.get("paper_id", "")
        image_id = item.get("image_id", "")
        figure_label = item.get("figure_label", "")

        title = f"{figure_label}: {caption}" if figure_label and caption else (caption or figure_label or f"Figure: {image_id}")

        # Build comprehensive, searchable text content for retriever
        content_parts = [f"### {title} [Scientific Visual Analysis]"]
        content_parts.append(f"**Paper**: {paper_id} | **Section**: {section}")
        if caption:
            content_parts.append(f"**Caption**: {caption}")
        if in_text and in_text != "Not explicitly discussed in nearby body text.":
            content_parts.append(f"**Context from Manuscript**: {in_text}")
        content_parts.append(f"**Detailed Visual & Architectural Breakdown**:\n{desc}")

        full_content = "\n\n".join(content_parts)

        figure_chunks.append({
            "chunk_type": "figure",
            "paper_id": paper_id,
            "section": section,
            "image_id": image_id,
            "caption": caption,
            "chunk_index": 0,
            "content": full_content,
            "word_count": len(full_content.split()),
            "char_count": len(full_content),
        })

    return figure_chunks


def chunk_all_documents(
    processed_dir: Path,
    output_file: Optional[Path] = None,
    max_words_per_section: int = 1000,
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
    include_figures: bool = True,
) -> list[dict]:
    """Load all Markdown papers and figure descriptions, chunk them, and save to JSON.

    Args:
        processed_dir: Directory containing .md files and image_descriptions.json.
        output_file: Optional path to save chunks.json.
        max_words_per_section: Word threshold to trigger sub-chunking.
        chunk_size: Character chunk size for sub-chunking.
        chunk_overlap: Character overlap for sub-chunking.
        include_figures: Whether to include multimodal figure chunks.

    Returns:
        List of all final chunk dicts.
    """
    processed_dir = Path(processed_dir)
    md_files = sorted(processed_dir.glob("*.md"))

    if not md_files:
        logger.warning(f"No .md files found in {processed_dir}")
        return []

    logger.info(f"Chunking {len(md_files)} Markdown files from {processed_dir}...")
    all_sections = []
    for md_file in md_files:
        paper_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        sections = split_sections(text, paper_id)
        all_sections.extend(sections)

    text_chunks = chunk_sections(
        all_sections,
        max_words_per_section=max_words_per_section,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    figure_chunks = []
    if include_figures:
        desc_path = processed_dir / "image_descriptions.json"
        if desc_path.exists():
            try:
                with open(desc_path, "r", encoding="utf-8") as f:
                    image_descriptions = json.load(f)
                figure_chunks = create_figure_chunks(image_descriptions)
                logger.info(f"Generated {len(figure_chunks)} multimodal figure chunks from {desc_path.name}")
            except Exception as e:
                logger.warning(f"Could not load {desc_path}: {e}")

    # Combine text and figure chunks
    all_chunks = text_chunks + figure_chunks

    # Assign unique sequential chunk IDs
    for global_id, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = global_id

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved {len(all_chunks)} chunks ({len(text_chunks)} text, {len(figure_chunks)} figure) to {output_file}"
        )

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Section-aware and multimodal chunking for scientific papers.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed Markdown directory")
    parser.add_argument("--output", type=str, default="data/processed/chunks.json", help="Path to output chunks JSON file")
    parser.add_argument("--max-words", type=int, default=1000, help="Max words in section before recursive splitting")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Character chunk size for sub-splitting")
    parser.add_argument("--chunk-overlap", type=int, default=300, help="Character chunk overlap")
    parser.add_argument("--no-figures", action="store_true", help="Exclude multimodal figure chunks")

    args = parser.parse_args()
    chunk_all_documents(
        processed_dir=Path(args.processed_dir),
        output_file=Path(args.output),
        max_words_per_section=args.max_words,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
