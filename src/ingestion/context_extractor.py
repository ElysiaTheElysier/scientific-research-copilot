"""Academic Context Extraction Module for Scientific Figures.

Extracts paper title, abstract, section structure, author figure captions,
and in-text figure citations to ground Multimodal VLMs in academic context.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_paper_metadata(markdown_text: str) -> dict:
    """Extract paper title and abstract/overview from markdown front matter.

    Args:
        markdown_text: Full text of the Markdown paper.

    Returns:
        Dict containing 'title' and 'abstract'.
    """
    lines = markdown_text.strip().split("\n")
    title = "Unknown Paper"
    abstract = ""

    # Find the first heading as the paper title
    for line in lines:
        cleaned = line.strip()
        if cleaned.startswith("# ") or cleaned.startswith("## "):
            title = re.sub(r"^#+\s*", "", cleaned).strip()
            # Stop if we found a substantive title (longer than single character/symbol)
            if len(title) > 3 and not title.lower().startswith("table"):
                break

    # Look for explicit Abstract section or front matter
    abstract_match = re.search(
        r"(?:^|\n)(?:##\s*)?Abstract[:\s*\n]+(.*?)(?=\n##|\Z)",
        markdown_text,
        re.DOTALL | re.IGNORECASE,
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()
    else:
        # Fallback: grab paragraphs from the front matter before the first ## 1 or ## Introduction
        front_parts = re.split(r"(?=\n##\s*(?:1\b|Introduction))", markdown_text, maxsplit=1, flags=re.IGNORECASE)
        if front_parts:
            paras = [p.strip() for p in front_parts[0].split("\n\n") if len(p.strip()) > 60]
            # Exclude title lines and author lines
            content_paras = [p for p in paras if not p.startswith("#")]
            if content_paras:
                abstract = content_paras[0]

    # Clean up whitespace
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())
    if len(abstract) > 1200:
        abstract = abstract[:1197] + "..."

    return {
        "title": title,
        "abstract": abstract,
    }


def find_in_text_mentions(markdown_text: str, figure_num: int, max_mentions: int = 4) -> list[str]:
    """Find in-text paragraphs or sentences citing a specific figure (e.g. Figure 1 / Fig. 1).

    Args:
        markdown_text: Full text of the paper.
        figure_num: The integer figure index (e.g. 1 for Figure 1).
        max_mentions: Maximum number of distinct discussion snippets to return.

    Returns:
        List of cleaned paragraphs/snippets mentioning this figure.
    """
    # Regex to match mentions like Figure 1, Fig. 1, Fig. 1(a), Figure 1., etc.
    fig_pattern = re.compile(
        rf"\b(?:Figure|Fig\.?)\s*{figure_num}\b(?:\([a-zA-Z0-9]+\))?",
        re.IGNORECASE,
    )

    paragraphs = markdown_text.split("\n\n")
    mentions = []

    for para in paragraphs:
        para_clean = " ".join(para.strip().split())
        if not para_clean:
            continue

        # Skip caption lines themselves (e.g. "Figure 1: ...")
        if re.match(rf"^(?:Figure|Fig\.?)\s*{figure_num}[:\.\-]\s+", para_clean, re.IGNORECASE):
            continue

        if fig_pattern.search(para_clean):
            # Trim excessively long paragraphs to keep context compact
            if len(para_clean) > 500:
                # Find the sentence containing the figure mention
                sentences = re.split(r"(?<=[.!?])\s+", para_clean)
                relevant = [s for s in sentences if fig_pattern.search(s)]
                if relevant:
                    snippet = " ".join(relevant[:2])
                else:
                    snippet = para_clean[:497] + "..."
            else:
                snippet = para_clean

            if snippet not in mentions:
                mentions.append(snippet)

        if len(mentions) >= max_mentions:
            break

    return mentions


def extract_figure_contexts(
    markdown_path: Path,
    images_dir: Optional[Path] = None,
) -> list[dict]:
    """Extract contextual grounding for all figures in a markdown paper.

    Args:
        markdown_path: Path to the paper markdown file.
        images_dir: Optional path to images directory (to verify figure files).

    Returns:
        List of dicts containing rich figure context:
        - image_id: str (e.g. '2608.24921v1_figure_0.png')
        - paper_id: str
        - figure_index: int
        - figure_label: str (e.g. 'Figure 1')
        - section: str
        - paper_title: str
        - paper_abstract: str
        - original_caption: str
        - in_text_mentions: list[str]
    """
    markdown_path = Path(markdown_path)
    paper_id = markdown_path.stem
    content = markdown_path.read_text(encoding="utf-8")

    metadata = extract_paper_metadata(content)
    paper_title = metadata["title"]
    paper_abstract = metadata["abstract"]

    # Split document by section headers (## ...)
    section_splits = re.split(r"(?=^##\s)", content, flags=re.MULTILINE)

    # Global scan for <!-- image --> occurrences
    img_matches = list(re.finditer(r"<!-- image -->", content))
    contexts = []

    for idx, match in enumerate(img_matches):
        img_pos = match.start()
        figure_id = f"{paper_id}_figure_{idx}.png"

        # Determine section where this image appears
        current_section = "Main"
        running_pos = 0
        for sec in section_splits:
            sec_len = len(sec)
            if running_pos <= img_pos < running_pos + sec_len:
                lines = sec.strip().split("\n")
                if lines[0].startswith("## "):
                    current_section = lines[0][3:].strip()
                elif lines[0].startswith("# "):
                    current_section = lines[0][2:].strip()
                break
            running_pos += sec_len

        # Search for caption in adjacent text window (-1500 to +1500 chars)
        window_before = content[max(0, img_pos - 1500) : img_pos]
        window_after = content[match.end() : min(len(content), match.end() + 1500)]

        before_paras = [p.strip() for p in window_before.split("\n\n") if p.strip()]
        after_paras = [p.strip() for p in window_after.split("\n\n") if p.strip()]

        caption_text = ""
        figure_label = f"Figure {idx + 1}"
        figure_num = idx + 1

        caption_regex = re.compile(
            r"(?:Figure|Fig\.?)\s+(\d+[a-zA-Z]?)(?:[:\.\-]\s*(.*?)(?=\n\n|\Z)|$)",
            re.DOTALL | re.IGNORECASE,
        )

        candidates = []
        if before_paras:
            candidates.append(before_paras[-1])
            if len(before_paras) >= 2:
                candidates.append(before_paras[-2])
        if after_paras:
            candidates.append(after_paras[0])
            if len(after_paras) >= 2:
                candidates.append(after_paras[1])

        found = False
        for cand in candidates:
            m = caption_regex.search(cand)
            if m:
                fig_num_str = m.group(1)
                try:
                    figure_num = int(re.sub(r"\D", "", fig_num_str))
                except ValueError:
                    figure_num = idx + 1
                figure_label = f"Figure {fig_num_str}"
                caption_body = m.group(2) if m.group(2) else cand[m.end() :].strip()
                caption_text = " ".join(caption_body.split())
                found = True
                break

        if not found:
            for text_chunk in [window_before, window_after]:
                m = caption_regex.search(text_chunk)
                if m:
                    fig_num_str = m.group(1)
                    try:
                        figure_num = int(re.sub(r"\D", "", fig_num_str))
                    except ValueError:
                        figure_num = idx + 1
                    figure_label = f"Figure {fig_num_str}"
                    caption_body = m.group(2) if m.group(2) else ""
                    caption_text = " ".join(caption_body.split())
                    found = True
                    break

        if not caption_text:
            caption_text = f"Scientific visual in {current_section}."

        in_text_mentions = find_in_text_mentions(content, figure_num=figure_num)

        contexts.append({
            "image_id": figure_id,
            "paper_id": paper_id,
            "figure_index": idx,
            "figure_label": figure_label,
            "section": current_section,
            "paper_title": paper_title,
            "paper_abstract": paper_abstract,
            "original_caption": caption_text,
            "in_text_mentions": in_text_mentions,
        })

    return contexts

