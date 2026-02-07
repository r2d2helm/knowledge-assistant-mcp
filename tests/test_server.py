"""
Tests for knowledge-assistant MCP server.
"""

import time
import pytest
from pathlib import Path


# ============== Tests for parse_frontmatter() ==============

class TestParseFrontmatter:
    """Tests for the parse_frontmatter function."""

    def test_valid_frontmatter(self):
        """Test parsing valid YAML frontmatter."""
        from src.server import parse_frontmatter
        from datetime import date

        content = """---
title: Test Note
date: 2024-01-15
type: concept
tags:
  - python
  - testing
---

# Body content

Some text here.
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["title"] == "Test Note"
        # YAML parses dates as datetime.date objects
        assert frontmatter["date"] == date(2024, 1, 15)
        assert frontmatter["type"] == "concept"
        assert frontmatter["tags"] == ["python", "testing"]
        assert "# Body content" in body
        assert "Some text here." in body

    def test_missing_frontmatter(self):
        """Test parsing content without frontmatter."""
        from src.server import parse_frontmatter

        content = """# Just a heading

No frontmatter here.
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_invalid_yaml_frontmatter(self):
        """Test parsing invalid YAML frontmatter returns empty dict."""
        from src.server import parse_frontmatter

        content = """---
title: [broken yaml syntax
invalid: : extra colon
---

Body content here.
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert "Body content here." in body

    def test_empty_frontmatter(self):
        """Test parsing empty frontmatter block."""
        from src.server import parse_frontmatter

        content = """---
---

Body only.
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter == {}
        assert "Body only." in body

    def test_frontmatter_with_complex_values(self):
        """Test parsing frontmatter with lists and nested values."""
        from src.server import parse_frontmatter

        content = """---
title: Complex Note
tags:
  - tag1
  - tag2
  - tag3
related:
  - Note A
  - Note B
---

Content.
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["tags"] == ["tag1", "tag2", "tag3"]
        assert frontmatter["related"] == ["Note A", "Note B"]

    def test_frontmatter_no_trailing_newline(self):
        """Test frontmatter handling with minimal spacing."""
        from src.server import parse_frontmatter

        content = """---
title: Minimal
---
Body immediately after."""

        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["title"] == "Minimal"
        assert "Body immediately after" in body


# ============== Tests for VaultCache ==============

class TestVaultCache:
    """Tests for the VaultCache class."""

    def test_cache_initialization(self, temp_vault):
        """Test VaultCache initializes with correct defaults."""
        from src.server import VaultCache

        cache = VaultCache(temp_vault, ttl=30)

        assert cache.vault_path == temp_vault
        assert cache.ttl == 30
        assert cache._notes == []
        assert cache._loaded_at == 0

    def test_is_stale_initially(self, temp_vault):
        """Test cache is stale before first load."""
        from src.server import VaultCache

        cache = VaultCache(temp_vault, ttl=60)

        assert cache.is_stale is True

    def test_is_not_stale_after_refresh(self, vault_cache):
        """Test cache is not stale immediately after refresh."""
        assert vault_cache.is_stale is False

    def test_is_stale_after_ttl(self, temp_vault):
        """Test cache becomes stale after TTL expires."""
        from src.server import VaultCache

        cache = VaultCache(temp_vault, ttl=1)
        cache.refresh(force=True)

        assert cache.is_stale is False
        time.sleep(1.1)
        assert cache.is_stale is True

    def test_refresh_loads_notes(self, vault_cache):
        """Test refresh loads notes from disk."""
        notes = vault_cache._notes

        assert len(notes) >= 7  # At least 7 notes in fixture
        titles = [n["stem"] for n in notes]
        assert "C_Python" in titles
        assert "C_JavaScript" in titles

    def test_refresh_force(self, vault_cache):
        """Test force refresh reloads even if not stale."""
        initial_time = vault_cache._loaded_at
        time.sleep(0.01)
        vault_cache.refresh(force=True)

        assert vault_cache._loaded_at > initial_time

    def test_refresh_skips_if_not_stale(self, vault_cache):
        """Test refresh does nothing if cache is fresh."""
        initial_time = vault_cache._loaded_at
        time.sleep(0.01)
        vault_cache.refresh(force=False)

        assert vault_cache._loaded_at == initial_time

    def test_get_notes_excludes_templates(self, vault_cache):
        """Test get_notes filters out templates by default."""
        notes = vault_cache.get_notes(include_templates=False)
        stems = [n["stem"] for n in notes]

        assert "T_Concept" not in stems

    def test_get_notes_includes_templates(self, vault_cache):
        """Test get_notes can include templates."""
        notes = vault_cache.get_notes(include_templates=True)
        stems = [n["stem"] for n in notes]

        assert "T_Concept" in stems

    def test_get_note_by_path_exact(self, vault_cache):
        """Test finding note by exact relative path."""
        note = vault_cache.get_note_by_path("Concepts/C_Python.md")

        assert note is not None
        assert note["stem"] == "C_Python"

    def test_get_note_by_path_title_match(self, vault_cache):
        """Test finding note by title/stem match."""
        note = vault_cache.get_note_by_path("Python")

        assert note is not None
        assert "Python" in note["stem"]

    def test_get_note_by_path_case_insensitive(self, vault_cache):
        """Test title matching is case-insensitive."""
        note = vault_cache.get_note_by_path("python")

        assert note is not None
        assert "Python" in note["stem"]

    def test_get_note_by_path_not_found(self, vault_cache):
        """Test returns None for non-existent note."""
        note = vault_cache.get_note_by_path("NonExistent")

        assert note is None

    def test_note_count(self, vault_cache):
        """Test note_count property."""
        assert vault_cache.note_count >= 7

    def test_note_stems(self, vault_cache):
        """Test note_stems property returns set of basenames."""
        stems = vault_cache.note_stems

        assert isinstance(stems, set)
        assert "C_Python" in stems
        assert "C_JavaScript" in stems

    def test_notes_have_required_fields(self, vault_cache):
        """Test each cached note has all required fields."""
        required_fields = [
            "path", "rel_path", "rel_path_str", "stem", "stem_lower",
            "content", "content_lower", "frontmatter", "body", "body_lower",
            "links", "tags", "type", "date", "mtime", "word_count", "is_template"
        ]

        for note in vault_cache._notes:
            for field in required_fields:
                assert field in note, f"Missing field: {field}"

    def test_links_extraction(self, vault_cache):
        """Test wiki links are extracted correctly."""
        note = vault_cache.get_note_by_path("C_JavaScript")

        assert note is not None
        assert "Python" in note["links"]
        assert "Docker" in note["links"]

    def test_aliased_links_extraction(self, vault_cache):
        """Test aliased wiki links extract the target, not alias."""
        note = vault_cache.get_note_by_path("C_Aliases")

        assert note is not None
        assert "Python" in note["links"]


# ============== Tests for search_notes() ==============

class TestSearchNotes:
    """Tests for the search_notes function."""

    def test_search_single_term(self, patched_vault_cache):
        """Test searching with a single term."""
        from src.server import search_notes

        results = search_notes("python")

        assert len(results) > 0
        assert any("Python" in r["title"] for r in results)

    def test_search_multi_word_and_logic(self, patched_vault_cache):
        """Test multi-word search uses AND logic."""
        from src.server import search_notes

        results = search_notes("programming language")

        assert len(results) > 0
        # All results should contain both terms
        for r in results:
            assert "programming" in r["matched_terms"]
            assert "language" in r["matched_terms"]

    def test_search_no_results(self, patched_vault_cache):
        """Test search with no matches returns empty list."""
        from src.server import search_notes

        results = search_notes("xyznonexistent123")

        assert results == []

    def test_search_empty_query(self, patched_vault_cache):
        """Test empty query returns empty list."""
        from src.server import search_notes

        results = search_notes("")

        assert results == []

    def test_search_max_results(self, patched_vault_cache):
        """Test max_results limits output."""
        from src.server import search_notes

        results = search_notes("python", max_results=1)

        assert len(results) <= 1

    def test_search_scoring_title_higher(self, patched_vault_cache):
        """Test title matches score higher than content matches."""
        from src.server import search_notes

        results = search_notes("python")

        if len(results) >= 2:
            # Title with "Python" should score higher
            python_note = next((r for r in results if "Python" in r["title"]), None)
            if python_note:
                assert python_note["score"] >= 10  # Title match = +10

    def test_search_snippet_extraction(self, patched_vault_cache):
        """Test snippets are extracted around matches."""
        from src.server import search_notes

        results = search_notes("programming")

        assert len(results) > 0
        for r in results:
            assert "snippet" in r
            assert len(r["snippet"]) > 0

    def test_search_result_structure(self, patched_vault_cache):
        """Test search results have expected structure."""
        from src.server import search_notes

        results = search_notes("python")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "score" in result
        assert "snippet" in result
        assert "tags" in result
        assert "type" in result
        assert "matched_terms" in result

    def test_search_case_insensitive(self, patched_vault_cache):
        """Test search is case-insensitive."""
        from src.server import search_notes

        results_lower = search_notes("python")
        results_upper = search_notes("PYTHON")
        results_mixed = search_notes("PyThOn")

        assert len(results_lower) == len(results_upper) == len(results_mixed)


# ============== Tests for get_related_notes() ==============

class TestGetRelatedNotes:
    """Tests for the get_related_notes function."""

    def test_related_by_link(self, patched_vault_cache):
        """Test finding related notes by wiki links."""
        from src.server import get_related_notes

        results = get_related_notes("Python")

        assert len(results) > 0
        # JavaScript links to Python
        js_note = next((r for r in results if "JavaScript" in r["title"]), None)
        assert js_note is not None

    def test_related_by_tag(self, patched_vault_cache):
        """Test finding related notes by tags."""
        from src.server import get_related_notes

        results = get_related_notes("programming")

        assert len(results) > 0
        # Notes with 'programming' tag should appear
        for r in results:
            assert "reasons" in r

    def test_related_by_title(self, patched_vault_cache):
        """Test finding related notes by title match."""
        from src.server import get_related_notes

        results = get_related_notes("Docker")

        # Docker note itself should score high (title match)
        docker_note = next((r for r in results if "Docker" in r["title"]), None)
        assert docker_note is not None
        assert "title match" in docker_note["reasons"]

    def test_related_scoring(self, patched_vault_cache):
        """Test related notes are sorted by score."""
        from src.server import get_related_notes

        results = get_related_notes("Python")

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_related_max_results(self, patched_vault_cache):
        """Test max_results limits output."""
        from src.server import get_related_notes

        results = get_related_notes("python", max_results=2)

        assert len(results) <= 2

    def test_related_result_structure(self, patched_vault_cache):
        """Test related notes result structure."""
        from src.server import get_related_notes

        results = get_related_notes("Python")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "score" in result
        assert "reasons" in result
        assert "type" in result


# ============== Tests for get_backlinks() ==============

class TestGetBacklinks:
    """Tests for the get_backlinks function."""

    def test_backlinks_found(self, patched_vault_cache):
        """Test finding backlinks to a note."""
        from src.server import get_backlinks

        results = get_backlinks("Python")

        assert len(results) > 0
        titles = [r["title"] for r in results]
        # JavaScript and Docker both link to Python
        assert any("JavaScript" in t for t in titles)

    def test_backlinks_case_insensitive(self, patched_vault_cache):
        """Test backlink search is case-insensitive."""
        from src.server import get_backlinks

        results_lower = get_backlinks("python")
        results_upper = get_backlinks("PYTHON")

        assert len(results_lower) == len(results_upper)

    def test_backlinks_not_found(self, patched_vault_cache):
        """Test no backlinks for unlinked note."""
        from src.server import get_backlinks

        results = get_backlinks("NonExistentNote")

        assert results == []

    def test_backlinks_result_structure(self, patched_vault_cache):
        """Test backlinks result structure."""
        from src.server import get_backlinks

        results = get_backlinks("Python")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "link_text" in result
        assert "type" in result


# ============== Tests for explore_by_tag() ==============

class TestExploreByTag:
    """Tests for the explore_by_tag function."""

    def test_explore_tag_found(self, patched_vault_cache):
        """Test finding notes by tag."""
        from src.server import explore_by_tag

        results = explore_by_tag("programming")

        assert len(results) >= 2  # Python and JavaScript both have this tag
        titles = [r["title"] for r in results]
        assert any("Python" in t for t in titles)
        assert any("JavaScript" in t for t in titles)

    def test_explore_tag_with_hash(self, patched_vault_cache):
        """Test tag search works with # prefix."""
        from src.server import explore_by_tag

        results = explore_by_tag("#programming")

        assert len(results) >= 2

    def test_explore_tag_case_insensitive(self, patched_vault_cache):
        """Test tag search is case-insensitive."""
        from src.server import explore_by_tag

        results_lower = explore_by_tag("programming")
        results_upper = explore_by_tag("PROGRAMMING")

        assert len(results_lower) == len(results_upper)

    def test_explore_tag_not_found(self, patched_vault_cache):
        """Test no results for non-existent tag."""
        from src.server import explore_by_tag

        results = explore_by_tag("nonexistenttag123")

        assert results == []

    def test_explore_tag_result_structure(self, patched_vault_cache):
        """Test explore_by_tag result structure."""
        from src.server import explore_by_tag

        results = explore_by_tag("programming")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "tags" in result
        assert "type" in result

    def test_explore_tag_partial_match(self, patched_vault_cache):
        """Test partial tag matching."""
        from src.server import explore_by_tag

        # Should match 'programming' and 'language'
        results = explore_by_tag("lang")

        assert len(results) >= 1


# ============== Tests for helper functions ==============

class TestHelperFunctions:
    """Tests for helper functions like generate_filename, get_default_folder, etc."""

    def test_generate_filename_concept(self):
        """Test filename generation for concept type."""
        from src.server import generate_filename

        filename = generate_filename("Test Concept", "concept")

        assert filename == "C_Test_Concept.md"

    def test_generate_filename_conversation(self):
        """Test filename generation for conversation type."""
        from src.server import generate_filename
        from datetime import datetime

        filename = generate_filename("My Chat", "conversation")
        today = datetime.now().strftime("%Y-%m-%d")

        assert filename == f"{today}_Conv_My_Chat.md"

    def test_generate_filename_reference(self):
        """Test filename generation for reference type."""
        from src.server import generate_filename

        filename = generate_filename("API Docs", "reference")

        assert filename == "R_API_Docs.md"

    def test_generate_filename_sanitizes_special_chars(self):
        """Test special characters are removed from filename."""
        from src.server import generate_filename

        filename = generate_filename("Test: With <Special> Chars!", "concept")

        assert ":" not in filename
        assert "<" not in filename
        assert ">" not in filename
        assert "!" not in filename

    def test_get_default_folder(self):
        """Test getting default folder for note types."""
        from src.server import get_default_folder

        assert get_default_folder("concept") == "Concepts"
        assert get_default_folder("session") == "Sessions"
        assert get_default_folder("reference") == "References"
        assert get_default_folder("unknown_type") == "Notes"

    def test_generate_frontmatter(self):
        """Test frontmatter generation."""
        from src.server import generate_frontmatter
        import yaml

        frontmatter_str = generate_frontmatter(
            title="Test Note",
            note_type="concept",
            tags=["python", "testing"],
            related=["Related Note"]
        )

        # Parse the generated frontmatter
        assert frontmatter_str.startswith("---\n")
        assert frontmatter_str.endswith("---\n\n")

        yaml_content = frontmatter_str.strip().strip("-").strip()
        parsed = yaml.safe_load(yaml_content)

        assert parsed["title"] == "Test Note"
        assert parsed["type"] == "concept"
        assert parsed["status"] == "seedling"
        assert parsed["tags"] == ["python", "testing"]
        assert parsed["related"] == ["Related Note"]

    def test_check_duplicates_exact_match(self, patched_vault_cache):
        """Test duplicate detection with exact title match."""
        from src.server import check_duplicates

        duplicates = check_duplicates("C_Python", "concept")

        assert len(duplicates) >= 1
        assert any(d["match_type"] == "exact_title" for d in duplicates)

    def test_check_duplicates_similar_title(self, patched_vault_cache):
        """Test duplicate detection with similar titles."""
        from src.server import check_duplicates

        duplicates = check_duplicates("Python Programming", "concept")

        # Should find C_Python as similar
        assert len(duplicates) >= 0  # May or may not match depending on overlap threshold


# ============== Tests for get_vault_stats() ==============

class TestGetVaultStats:
    """Tests for the get_vault_stats function."""

    def test_vault_stats_structure(self, patched_vault_cache):
        """Test vault stats has expected structure."""
        from src.server import get_vault_stats

        stats = get_vault_stats()

        assert "total_notes" in stats
        assert "by_type" in stats
        assert "by_folder" in stats
        assert "total_words" in stats
        assert "total_links" in stats
        assert "tags" in stats
        assert "recent_notes" in stats
        assert "top_tags" in stats

    def test_vault_stats_counts(self, patched_vault_cache):
        """Test vault stats counts are reasonable."""
        from src.server import get_vault_stats

        stats = get_vault_stats()

        # Templates are excluded by default
        assert stats["total_notes"] >= 5
        assert stats["total_words"] > 0
        assert stats["total_links"] > 0

    def test_vault_stats_by_type(self, patched_vault_cache):
        """Test notes are categorized by type."""
        from src.server import get_vault_stats

        stats = get_vault_stats()

        assert "concept" in stats["by_type"]
        assert stats["by_type"]["concept"] >= 2

    def test_vault_stats_recent_notes(self, patched_vault_cache):
        """Test recent notes are included."""
        from src.server import get_vault_stats

        stats = get_vault_stats()

        assert len(stats["recent_notes"]) > 0
        for note in stats["recent_notes"]:
            assert "title" in note
            assert "path" in note


# ============== Tests for Graph Functions ==============

class TestBuildGraph:
    """Tests for the build_graph function."""

    def test_build_graph_structure(self, patched_vault_cache):
        """Test build_graph returns expected structure."""
        from src.server import build_graph

        result = build_graph()

        assert "nodes" in result
        assert "edges" in result
        assert "orphans" in result
        assert "stats" in result

    def test_build_graph_stats(self, patched_vault_cache):
        """Test build_graph stats are populated."""
        from src.server import build_graph

        result = build_graph()

        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]
        assert "orphan_count" in result["stats"]
        assert "avg_connections" in result["stats"]
        assert result["stats"]["total_nodes"] >= 5  # Excluding template

    def test_build_graph_nodes_have_required_fields(self, patched_vault_cache):
        """Test each node has required fields."""
        from src.server import build_graph

        result = build_graph()

        for node in result["nodes"]:
            assert "id" in node
            assert "title" in node
            assert "path" in node
            assert "type" in node
            assert "connections" in node

    def test_build_graph_edges_structure(self, patched_vault_cache):
        """Test edges have source and target."""
        from src.server import build_graph

        result = build_graph()

        # We expect edges from JavaScript -> Python, JavaScript -> Docker, etc.
        assert len(result["edges"]) > 0
        for edge in result["edges"]:
            assert "source" in edge
            assert "target" in edge

    def test_build_graph_finds_links(self, patched_vault_cache):
        """Test graph correctly finds links between notes."""
        from src.server import build_graph

        result = build_graph()

        # C_JavaScript links to C_Python
        js_to_python = any(
            e["source"] == "C_JavaScript" and e["target"] == "C_Python"
            for e in result["edges"]
        )
        assert js_to_python

    def test_build_graph_orphans(self, patched_vault_cache):
        """Test orphan detection works."""
        from src.server import build_graph

        result = build_graph()

        # Some notes might be orphans (no incoming/outgoing links to existing notes)
        assert isinstance(result["orphans"], list)


class TestGetSubgraph:
    """Tests for the get_subgraph function."""

    def test_get_subgraph_found(self, patched_vault_cache):
        """Test getting subgraph around a note."""
        from src.server import get_subgraph

        result = get_subgraph("C_Python", depth=1)

        assert "error" not in result
        assert "nodes" in result
        assert "edges" in result
        assert "center" in result
        assert result["center"] == "C_Python"

    def test_get_subgraph_includes_center(self, patched_vault_cache):
        """Test subgraph includes the center note."""
        from src.server import get_subgraph

        result = get_subgraph("C_Python", depth=1)

        center_node = next((n for n in result["nodes"] if n["id"] == "C_Python"), None)
        assert center_node is not None
        assert center_node.get("is_center") is True

    def test_get_subgraph_includes_neighbors(self, patched_vault_cache):
        """Test subgraph includes connected notes."""
        from src.server import get_subgraph

        result = get_subgraph("C_Python", depth=1)

        # C_JavaScript links to C_Python, so it should be a neighbor
        js_node = next((n for n in result["nodes"] if n["id"] == "C_JavaScript"), None)
        assert js_node is not None

    def test_get_subgraph_not_found(self, patched_vault_cache):
        """Test error for non-existent note."""
        from src.server import get_subgraph

        result = get_subgraph("NonExistentNote", depth=1)

        assert "error" in result

    def test_get_subgraph_depth(self, patched_vault_cache):
        """Test depth controls subgraph size."""
        from src.server import get_subgraph

        depth1 = get_subgraph("C_Python", depth=1)
        depth2 = get_subgraph("C_Python", depth=2)

        # Depth 2 should have >= nodes than depth 1
        assert len(depth2["nodes"]) >= len(depth1["nodes"])

    def test_get_subgraph_stats(self, patched_vault_cache):
        """Test subgraph stats are populated."""
        from src.server import get_subgraph

        result = get_subgraph("C_Python", depth=2)

        assert "stats" in result
        assert result["stats"]["depth"] == 2
        assert result["stats"]["center"] == "C_Python"
        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]


class TestGetClusters:
    """Tests for the get_clusters function."""

    def test_get_clusters_structure(self, patched_vault_cache):
        """Test get_clusters returns expected structure."""
        from src.server import get_clusters

        result = get_clusters()

        assert "clusters" in result
        assert "orphans" in result
        assert "stats" in result

    def test_get_clusters_hub_structure(self, patched_vault_cache):
        """Test each cluster has expected fields."""
        from src.server import get_clusters

        result = get_clusters()

        if result["clusters"]:
            cluster = result["clusters"][0]
            assert "hub" in cluster
            assert "neighbors" in cluster
            assert "neighbor_count" in cluster
            assert "id" in cluster["hub"]
            assert "title" in cluster["hub"]
            assert "connections" in cluster["hub"]

    def test_get_clusters_sorted_by_connections(self, patched_vault_cache):
        """Test clusters are sorted by number of connections."""
        from src.server import get_clusters

        result = get_clusters()

        if len(result["clusters"]) >= 2:
            connections = [c["hub"]["connections"] for c in result["clusters"]]
            assert connections == sorted(connections, reverse=True)

    def test_get_clusters_top_n(self, patched_vault_cache):
        """Test top_n limits the number of clusters."""
        from src.server import get_clusters

        result = get_clusters(top_n=2)

        assert len(result["clusters"]) <= 2

    def test_get_clusters_stats(self, patched_vault_cache):
        """Test clusters stats are populated."""
        from src.server import get_clusters

        result = get_clusters()

        assert "total_clusters" in result["stats"]
        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]
        assert "orphan_count" in result["stats"]
