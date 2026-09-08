"""Tests for Academic Context Extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.context_extractor import (
    extract_figure_contexts,
    extract_paper_metadata,
    find_in_text_mentions,
)

SAMPLE_PAPER_MARKDOWN = """# Graph-Based RAG Systems
Chandan Rajah, Jane Doe
Abstract
This paper presents a PostgreSQL-native Graph RAG architecture designed to improve multi-hop retrieval over scientific knowledge graphs.

## 1 Introduction
Traditional passage retrieval misses cross-document connections. As illustrated in Figure 1, our system couples vector search with one-hop graph traversal.

Figure 1: High-level overview of the Graph RAG dual-layer architecture.

<!-- image -->

## 2 Experimental Evaluation
We evaluate retrieval precision across three benchmark corpora. Figure 2 demonstrates that our extraction gates reduce orphan nodes significantly.

Figure 2: Empirical analysis of graph density and predicate distributions.

<!-- image -->

Furthermore, we observe that the gains shown in Fig. 2 are consistent across different LLM backends.
"""


def test_extract_paper_metadata():
    meta = extract_paper_metadata(SAMPLE_PAPER_MARKDOWN)
    assert meta["title"] == "Graph-Based RAG Systems"
    assert "PostgreSQL-native Graph RAG" in meta["abstract"]


def test_find_in_text_mentions():
    mentions_fig1 = find_in_text_mentions(SAMPLE_PAPER_MARKDOWN, figure_num=1)
    assert len(mentions_fig1) >= 1
    assert "As illustrated in Figure 1" in mentions_fig1[0]

    mentions_fig2 = find_in_text_mentions(SAMPLE_PAPER_MARKDOWN, figure_num=2)
    assert len(mentions_fig2) >= 1
    # Check that both 'Figure 2' and 'Fig. 2' are detected in text mentions
    assert any("Figure 2" in m or "Fig. 2" in m for m in mentions_fig2)


def test_extract_figure_contexts(tmp_path):
    md_file = tmp_path / "paper_test.md"
    md_file.write_text(SAMPLE_PAPER_MARKDOWN, encoding="utf-8")

    contexts = extract_figure_contexts(md_file)
    assert len(contexts) == 2

    # Check Figure 1
    fig1 = contexts[0]
    assert fig1["figure_label"] == "Figure 1"
    assert fig1["section"] == "1 Introduction"
    assert "High-level overview" in fig1["original_caption"]
    assert fig1["paper_title"] == "Graph-Based RAG Systems"
    assert len(fig1["in_text_mentions"]) >= 1

    # Check Figure 2
    fig2 = contexts[1]
    assert fig2["figure_label"] == "Figure 2"
    assert fig2["section"] == "2 Experimental Evaluation"
    assert "Empirical analysis" in fig2["original_caption"]
    assert len(fig2["in_text_mentions"]) >= 1


if __name__ == "__main__":
    test_extract_paper_metadata()
    test_find_in_text_mentions()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_extract_figure_contexts(Path(td))
    print("All context extraction tests passed!")

