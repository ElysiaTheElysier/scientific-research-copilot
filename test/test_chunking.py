import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunk import chunk_sections, create_figure_chunks, split_sections

SAMPLE_MARKDOWN = """# Paper Title
Chandan Rajah
Abstract text goes here.

## Introduction
This is the introduction section. It provides background context for the paper.

## Methods
Here we describe the experimental setup and model architecture.

## References
[1] Author et al. Paper title. 2024.
"""


def test_split_sections():
    sections = split_sections(SAMPLE_MARKDOWN, paper_id="test_paper")
    assert len(sections) == 4
    assert sections[0]["section"] == "front_matter"
    assert "Paper Title" in sections[0]["content"]
    assert sections[1]["section"] == "Introduction"
    assert sections[2]["section"] == "Methods"
    assert sections[3]["section"] == "References"
    titles = [s["section"] for s in sections]
    assert "Introduction" in titles
    assert "Methods" in titles
    assert "References" in titles


def test_chunk_sections_skips_references():
    sections = split_sections(SAMPLE_MARKDOWN, paper_id="test_paper")
    chunks = chunk_sections(sections, max_words_per_section=500)
    section_names = [c["section"].lower() for c in chunks]
    assert "references" not in section_names
    assert len(chunks) == 3  # front_matter, Introduction, Methods
    assert all("paper_id" in c for c in chunks)
    assert all(c["chunk_type"] == "text" for c in chunks)


def test_chunk_sections_subsplitting():
    long_content = "Word " * 1200
    sections = [
        {"paper_id": "test_paper", "section": "Long Section", "content": long_content}
    ]
    chunks = chunk_sections(sections, max_words_per_section=1000, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(c["section"] == "Long Section" for c in chunks)
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_create_figure_chunks():
    sample_descriptions = [
        {
            "image_id": "paper1_figure_0.png",
            "paper_id": "paper1",
            "caption": "Figure 1: Architecture overview",
            "section": "3 System Architecture",
            "in_text_context": "Figure 1 illustrates the indexing pipeline.",
            "description": "A flowchart showing inputs, embedding models, and PostgreSQL storage.",
        }
    ]
    fig_chunks = create_figure_chunks(sample_descriptions)
    assert len(fig_chunks) == 1
    assert fig_chunks[0]["chunk_type"] == "figure"
    assert "Figure 1: Architecture overview" in fig_chunks[0]["content"]
    assert "PostgreSQL storage" in fig_chunks[0]["content"]
    assert fig_chunks[0]["paper_id"] == "paper1"
    assert fig_chunks[0]["image_id"] == "paper1_figure_0.png"


if __name__ == "__main__":
    test_split_sections()
    test_chunk_sections_skips_references()
    test_chunk_sections_subsplitting()
    test_create_figure_chunks()
    print("All unit tests passed successfully!")
