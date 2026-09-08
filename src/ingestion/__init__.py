"""Ingestion module for scientific research copilot."""

from src.ingestion.caption import caption_all_figures, caption_single_figure
from src.ingestion.chunk import (
    chunk_all_documents,
    chunk_sections,
    create_figure_chunks,
    split_sections,
)
from src.ingestion.context_extractor import (
    extract_figure_contexts,
    extract_paper_metadata,
    find_in_text_mentions,
)
from src.ingestion.convert import convert_all_pdfs, convert_single_pdf

__all__ = [
    "convert_single_pdf",
    "convert_all_pdfs",
    "caption_single_figure",
    "caption_all_figures",
    "extract_figure_contexts",
    "extract_paper_metadata",
    "find_in_text_mentions",
    "split_sections",
    "chunk_sections",
    "create_figure_chunks",
    "chunk_all_documents",
]
