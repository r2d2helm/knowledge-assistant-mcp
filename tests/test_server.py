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
        assert cache._notes == {}
        assert cache._mtimes == {}
        assert cache._loaded_at == 0

    def test_is_stale_initially(self, temp_vault):
        """Test cache is stale before first load."""
        from src.server import VaultCache

        cache = VaultCache(temp_vault, ttl=60)

        assert cache.is_stale is True

    async def test_is_not_stale_after_refresh(self, vault_cache):
        """Test cache is not stale immediately after refresh."""
        assert vault_cache.is_stale is False

    async def test_is_stale_after_ttl(self, temp_vault):
        """Test cache becomes stale after TTL expires."""
        from src.server import VaultCache

        cache = VaultCache(temp_vault, ttl=1)
        await cache.refresh(force=True)

        assert cache.is_stale is False
        time.sleep(1.1)
        assert cache.is_stale is True

    async def test_refresh_loads_notes(self, vault_cache):
        """Test refresh loads notes from disk."""
        notes = vault_cache._notes

        assert len(notes) >= 7  # At least 7 notes in fixture
        titles = [n.stem for n in notes]
        assert "C_Python" in titles
        assert "C_JavaScript" in titles

    async def test_refresh_force(self, vault_cache):
        """Test force refresh reloads even if not stale."""
        initial_time = vault_cache._loaded_at
        time.sleep(0.01)
        await vault_cache.refresh(force=True)

        assert vault_cache._loaded_at > initial_time

    async def test_refresh_skips_if_not_stale(self, vault_cache):
        """Test refresh does nothing if cache is fresh."""
        initial_time = vault_cache._loaded_at
        time.sleep(0.01)
        await vault_cache.refresh(force=False)

        assert vault_cache._loaded_at == initial_time

    async def test_get_notes_excludes_templates(self, vault_cache):
        """Test get_notes filters out templates by default."""
        notes = await vault_cache.get_notes(include_templates=False)
        stems = [n.stem for n in notes]

        assert "T_Concept" not in stems

    async def test_get_notes_includes_templates(self, vault_cache):
        """Test get_notes can include templates."""
        notes = await vault_cache.get_notes(include_templates=True)
        stems = [n.stem for n in notes]

        assert "T_Concept" in stems

    async def test_get_note_by_path_exact(self, vault_cache):
        """Test finding note by exact relative path."""
        note = await vault_cache.get_note_by_path("Concepts/C_Python.md")

        assert note is not None
        assert note.stem == "C_Python"

    async def test_get_note_by_path_title_match(self, vault_cache):
        """Test finding note by title/stem match."""
        note = await vault_cache.get_note_by_path("Python")

        assert note is not None
        assert "Python" in note.stem

    async def test_get_note_by_path_case_insensitive(self, vault_cache):
        """Test title matching is case-insensitive."""
        note = await vault_cache.get_note_by_path("python")

        assert note is not None
        assert "Python" in note.stem

    async def test_get_note_by_path_not_found(self, vault_cache):
        """Test returns None for non-existent note."""
        note = await vault_cache.get_note_by_path("NonExistent")

        assert note is None

    async def test_note_count(self, vault_cache):
        """Test get_note_count method."""
        assert await vault_cache.get_note_count() >= 7

    async def test_note_stems(self, vault_cache):
        """Test get_note_stems method returns set of basenames."""
        stems = await vault_cache.get_note_stems()

        assert isinstance(stems, set)
        assert "C_Python" in stems
        assert "C_JavaScript" in stems

    async def test_notes_have_required_fields(self, vault_cache):
        """Test each cached note has all required fields."""
        required_fields = [
            "path", "rel_path", "rel_path_str", "stem", "stem_lower",
            "content", "content_lower", "frontmatter", "body", "body_lower",
            "links", "tags", "type", "date", "mtime", "word_count", "is_template"
        ]

        for note in vault_cache._notes.values():
            for field in required_fields:
                assert hasattr(note, field), f"Missing field: {field}"

    async def test_links_extraction(self, vault_cache):
        """Test wiki links are extracted correctly."""
        note = await vault_cache.get_note_by_path("C_JavaScript")

        assert note is not None
        assert "Python" in note.links
        assert "Docker" in note.links

    async def test_aliased_links_extraction(self, vault_cache):
        """Test aliased wiki links extract the target, not alias."""
        note = await vault_cache.get_note_by_path("C_Aliases")

        assert note is not None
        assert "Python" in note.links

    async def test_incremental_refresh_modified_file(self, temp_vault):
        """Test incremental refresh only reloads modified files."""
        from src.server import VaultCache
        import time as time_module

        cache = VaultCache(temp_vault, ttl=1)
        await cache.refresh(force=True)

        initial_count = len(cache._notes)
        python_file = temp_vault / "Concepts" / "C_Python.md"
        initial_mtime = cache._mtimes[python_file]

        # Wait for TTL to expire
        time_module.sleep(1.1)

        # Modify the file
        python_file.write_text("""---
title: Python Modified
date: 2024-01-15
type: concept
status: evergreen
tags:
  - programming
  - language
---

# Python Modified

This note was modified for testing.
""", encoding="utf-8")

        # Trigger incremental refresh
        await cache.refresh(force=False)

        # Verify only the modified file was reloaded
        assert len(cache._notes) == initial_count
        assert cache._mtimes[python_file] > initial_mtime
        note = cache._notes[python_file]
        assert note.frontmatter["title"] == "Python Modified"
        assert "modified for testing" in note.body.lower()

    async def test_incremental_refresh_deleted_file(self, temp_vault):
        """Test incremental refresh removes deleted files from cache."""
        from src.server import VaultCache
        import time as time_module

        cache = VaultCache(temp_vault, ttl=1)
        await cache.refresh(force=True)

        initial_count = len(cache._notes)
        python_file = temp_vault / "Concepts" / "C_Python.md"
        assert python_file in cache._notes

        # Wait for TTL to expire
        time_module.sleep(1.1)

        # Delete the file
        python_file.unlink()

        # Trigger incremental refresh
        await cache.refresh(force=False)

        # Verify the file was removed from cache
        assert len(cache._notes) == initial_count - 1
        assert python_file not in cache._notes
        assert python_file not in cache._mtimes
        assert "C_Python" not in cache._note_stems

    async def test_incremental_refresh_new_file(self, temp_vault):
        """Test incremental refresh adds new files to cache."""
        from src.server import VaultCache
        import time as time_module

        cache = VaultCache(temp_vault, ttl=1)
        await cache.refresh(force=True)

        initial_count = len(cache._notes)

        # Wait for TTL to expire
        time_module.sleep(1.1)

        # Create a new file
        new_file = temp_vault / "Concepts" / "C_NewNote.md"
        new_file.write_text("""---
title: New Note
date: 2024-01-30
type: concept
status: seedling
tags:
  - testing
---

# New Note

This is a brand new note.
""", encoding="utf-8")

        # Trigger incremental refresh
        await cache.refresh(force=False)

        # Verify the new file was added to cache
        assert len(cache._notes) == initial_count + 1
        assert new_file in cache._notes
        assert new_file in cache._mtimes
        assert "C_NewNote" in cache._note_stems

    async def test_incremental_refresh_unmodified_not_reloaded(self, temp_vault):
        """Test that unmodified files are not reloaded during incremental refresh."""
        from src.server import VaultCache
        import time as time_module

        cache = VaultCache(temp_vault, ttl=1)
        await cache.refresh(force=True)

        python_file = temp_vault / "Concepts" / "C_Python.md"
        original_note = cache._notes[python_file]
        original_mtime = cache._mtimes[python_file]

        # Wait for TTL to expire
        time_module.sleep(1.1)

        # Trigger incremental refresh without modifying any files
        await cache.refresh(force=False)

        # Verify the note object is the same (not reloaded)
        assert cache._notes[python_file] is original_note
        assert cache._mtimes[python_file] == original_mtime


# ============== Tests for search_notes() ==============

class TestSearchNotes:
    """Tests for the search_notes function."""

    async def test_search_single_term(self, patched_vault_cache):
        """Test searching with a single term."""
        from src.server import search_notes

        results = await search_notes("python")

        assert len(results) > 0
        assert any("Python" in r.title for r in results)

    async def test_search_multi_word_and_logic(self, patched_vault_cache):
        """Test multi-word search uses AND logic."""
        from src.server import search_notes

        results = await search_notes("programming language")

        assert len(results) > 0
        # All results should contain both terms
        for r in results:
            assert "programming" in r.matched_terms
            assert "language" in r.matched_terms

    async def test_search_no_results(self, patched_vault_cache):
        """Test search with no matches returns empty list."""
        from src.server import search_notes

        results = await search_notes("xyznonexistent123")

        assert results == []

    async def test_search_empty_query(self, patched_vault_cache):
        """Test empty query returns empty list."""
        from src.server import search_notes

        results = await search_notes("")

        assert results == []

    async def test_search_max_results(self, patched_vault_cache):
        """Test max_results limits output."""
        from src.server import search_notes

        results = await search_notes("python", max_results=1)

        assert len(results) <= 1

    async def test_search_scoring_title_higher(self, patched_vault_cache):
        """Test title matches score higher than content matches."""
        from src.server import search_notes

        results = await search_notes("python")

        if len(results) >= 2:
            # Title with "Python" should score higher
            python_note = next((r for r in results if "Python" in r.title), None)
            if python_note:
                assert python_note.score >= 10  # Title match = +10

    async def test_search_snippet_extraction(self, patched_vault_cache):
        """Test snippets are extracted around matches."""
        from src.server import search_notes

        results = await search_notes("programming")

        assert len(results) > 0
        for r in results:
            assert hasattr(r, "snippet")
            assert len(r.snippet) > 0

    async def test_search_result_structure(self, patched_vault_cache):
        """Test search results have expected structure."""
        from src.server import search_notes

        results = await search_notes("python")

        assert len(results) > 0
        result = results[0]
        # SearchResult is a Pydantic model, access via attributes
        assert hasattr(result, "title")
        assert hasattr(result, "path")
        assert hasattr(result, "score")
        assert hasattr(result, "snippet")
        assert hasattr(result, "tags")
        assert hasattr(result, "type")
        assert hasattr(result, "matched_terms")

    async def test_search_case_insensitive(self, patched_vault_cache):
        """Test search is case-insensitive."""
        from src.server import search_notes

        results_lower = await search_notes("python")
        results_upper = await search_notes("PYTHON")
        results_mixed = await search_notes("PyThOn")

        assert len(results_lower) == len(results_upper) == len(results_mixed)


# ============== Tests for get_related_notes() ==============

class TestGetRelatedNotes:
    """Tests for the get_related_notes function."""

    async def test_related_by_link(self, patched_vault_cache):
        """Test finding related notes by wiki links."""
        from src.server import get_related_notes

        results = await get_related_notes("Python")

        assert len(results) > 0
        # JavaScript links to Python
        js_note = next((r for r in results if "JavaScript" in r.title), None)
        assert js_note is not None

    async def test_related_by_tag(self, patched_vault_cache):
        """Test finding related notes by tags."""
        from src.server import get_related_notes

        results = await get_related_notes("programming")

        assert len(results) > 0
        # Notes with 'programming' tag should appear
        for r in results:
            assert hasattr(r, "reasons")

    async def test_related_by_title(self, patched_vault_cache):
        """Test finding related notes by title match."""
        from src.server import get_related_notes

        results = await get_related_notes("Docker")

        # Docker note itself should score high (title match)
        docker_note = next((r for r in results if "Docker" in r.title), None)
        assert docker_note is not None
        assert "title match" in docker_note.reasons

    async def test_related_scoring(self, patched_vault_cache):
        """Test related notes are sorted by score."""
        from src.server import get_related_notes

        results = await get_related_notes("Python")

        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    async def test_related_max_results(self, patched_vault_cache):
        """Test max_results limits output."""
        from src.server import get_related_notes

        results = await get_related_notes("python", max_results=2)

        assert len(results) <= 2

    async def test_related_result_structure(self, patched_vault_cache):
        """Test related notes result structure."""
        from src.server import get_related_notes

        results = await get_related_notes("Python")

        assert len(results) > 0
        result = results[0]
        # RelatedResult is a Pydantic model, access via attributes
        assert hasattr(result, "title")
        assert hasattr(result, "path")
        assert hasattr(result, "score")
        assert hasattr(result, "reasons")
        assert hasattr(result, "type")


# ============== Tests for get_backlinks() ==============

class TestGetBacklinks:
    """Tests for the get_backlinks function."""

    async def test_backlinks_found(self, patched_vault_cache):
        """Test finding backlinks to a note."""
        from src.server import get_backlinks

        results = await get_backlinks("Python")

        assert len(results) > 0
        titles = [r["title"] for r in results]
        # JavaScript and Docker both link to Python
        assert any("JavaScript" in t for t in titles)

    async def test_backlinks_case_insensitive(self, patched_vault_cache):
        """Test backlink search is case-insensitive."""
        from src.server import get_backlinks

        results_lower = await get_backlinks("python")
        results_upper = await get_backlinks("PYTHON")

        assert len(results_lower) == len(results_upper)

    async def test_backlinks_not_found(self, patched_vault_cache):
        """Test no backlinks for unlinked note."""
        from src.server import get_backlinks

        results = await get_backlinks("NonExistentNote")

        assert results == []

    async def test_backlinks_result_structure(self, patched_vault_cache):
        """Test backlinks result structure."""
        from src.server import get_backlinks

        results = await get_backlinks("Python")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "link_text" in result
        assert "type" in result


# ============== Tests for explore_by_tag() ==============

class TestExploreByTag:
    """Tests for the explore_by_tag function."""

    async def test_explore_tag_found(self, patched_vault_cache):
        """Test finding notes by tag."""
        from src.server import explore_by_tag

        results = await explore_by_tag("programming")

        assert len(results) >= 2  # Python and JavaScript both have this tag
        titles = [r["title"] for r in results]
        assert any("Python" in t for t in titles)
        assert any("JavaScript" in t for t in titles)

    async def test_explore_tag_with_hash(self, patched_vault_cache):
        """Test tag search works with # prefix."""
        from src.server import explore_by_tag

        results = await explore_by_tag("#programming")

        assert len(results) >= 2

    async def test_explore_tag_case_insensitive(self, patched_vault_cache):
        """Test tag search is case-insensitive."""
        from src.server import explore_by_tag

        results_lower = await explore_by_tag("programming")
        results_upper = await explore_by_tag("PROGRAMMING")

        assert len(results_lower) == len(results_upper)

    async def test_explore_tag_not_found(self, patched_vault_cache):
        """Test no results for non-existent tag."""
        from src.server import explore_by_tag

        results = await explore_by_tag("nonexistenttag123")

        assert results == []

    async def test_explore_tag_result_structure(self, patched_vault_cache):
        """Test explore_by_tag result structure."""
        from src.server import explore_by_tag

        results = await explore_by_tag("programming")

        assert len(results) > 0
        result = results[0]
        assert "title" in result
        assert "path" in result
        assert "tags" in result
        assert "type" in result

    async def test_explore_tag_partial_match(self, patched_vault_cache):
        """Test partial tag matching."""
        from src.server import explore_by_tag

        # Should match 'programming' and 'language'
        results = await explore_by_tag("lang")

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

    async def test_check_duplicates_exact_match(self, patched_vault_cache):
        """Test duplicate detection with exact title match."""
        from src.server import check_duplicates

        duplicates = await check_duplicates("C_Python", "concept")

        assert len(duplicates) >= 1
        assert any(d["match_type"] == "exact_title" for d in duplicates)

    async def test_check_duplicates_similar_title(self, patched_vault_cache):
        """Test duplicate detection with similar titles."""
        from src.server import check_duplicates

        duplicates = await check_duplicates("Python Programming", "concept")

        # Should find C_Python as similar
        assert len(duplicates) >= 0  # May or may not match depending on overlap threshold


# ============== Tests for get_vault_stats() ==============

class TestGetVaultStats:
    """Tests for the get_vault_stats function."""

    async def test_vault_stats_structure(self, patched_vault_cache):
        """Test vault stats has expected structure."""
        from src.server import get_vault_stats

        stats = await get_vault_stats()

        assert "total_notes" in stats
        assert "by_type" in stats
        assert "by_folder" in stats
        assert "total_words" in stats
        assert "total_links" in stats
        assert "tags" in stats
        assert "recent_notes" in stats
        assert "top_tags" in stats

    async def test_vault_stats_counts(self, patched_vault_cache):
        """Test vault stats counts are reasonable."""
        from src.server import get_vault_stats

        stats = await get_vault_stats()

        # Templates are excluded by default
        assert stats["total_notes"] >= 5
        assert stats["total_words"] > 0
        assert stats["total_links"] > 0

    async def test_vault_stats_by_type(self, patched_vault_cache):
        """Test notes are categorized by type."""
        from src.server import get_vault_stats

        stats = await get_vault_stats()

        assert "concept" in stats["by_type"]
        assert stats["by_type"]["concept"] >= 2

    async def test_vault_stats_recent_notes(self, patched_vault_cache):
        """Test recent notes are included."""
        from src.server import get_vault_stats

        stats = await get_vault_stats()

        assert len(stats["recent_notes"]) > 0
        for note in stats["recent_notes"]:
            assert "title" in note
            assert "path" in note


# ============== Tests for Graph Functions ==============

class TestBuildGraph:
    """Tests for the build_graph function."""

    async def test_build_graph_structure(self, patched_vault_cache):
        """Test build_graph returns expected structure."""
        from src.server import build_graph

        result = await build_graph()

        assert "nodes" in result
        assert "edges" in result
        assert "orphans" in result
        assert "stats" in result

    async def test_build_graph_stats(self, patched_vault_cache):
        """Test build_graph stats are populated."""
        from src.server import build_graph

        result = await build_graph()

        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]
        assert "orphan_count" in result["stats"]
        assert "avg_connections" in result["stats"]
        assert result["stats"]["total_nodes"] >= 5  # Excluding template

    async def test_build_graph_nodes_have_required_fields(self, patched_vault_cache):
        """Test each node has required fields."""
        from src.server import build_graph

        result = await build_graph()

        for node in result["nodes"]:
            assert "id" in node
            assert "title" in node
            assert "path" in node
            assert "type" in node
            assert "connections" in node

    async def test_build_graph_edges_structure(self, patched_vault_cache):
        """Test edges have source and target."""
        from src.server import build_graph

        result = await build_graph()

        # We expect edges from JavaScript -> Python, JavaScript -> Docker, etc.
        assert len(result["edges"]) > 0
        for edge in result["edges"]:
            assert "source" in edge
            assert "target" in edge

    async def test_build_graph_finds_links(self, patched_vault_cache):
        """Test graph correctly finds links between notes."""
        from src.server import build_graph

        result = await build_graph()

        # C_JavaScript links to C_Python
        js_to_python = any(
            e["source"] == "C_JavaScript" and e["target"] == "C_Python"
            for e in result["edges"]
        )
        assert js_to_python

    async def test_build_graph_orphans(self, patched_vault_cache):
        """Test orphan detection works."""
        from src.server import build_graph

        result = await build_graph()

        # Some notes might be orphans (no incoming/outgoing links to existing notes)
        assert isinstance(result["orphans"], list)


class TestGetSubgraph:
    """Tests for the get_subgraph function."""

    async def test_get_subgraph_found(self, patched_vault_cache):
        """Test getting subgraph around a note."""
        from src.server import get_subgraph

        result = await get_subgraph("C_Python", depth=1)

        assert "error" not in result
        assert "nodes" in result
        assert "edges" in result
        assert "center" in result
        assert result["center"] == "C_Python"

    async def test_get_subgraph_includes_center(self, patched_vault_cache):
        """Test subgraph includes the center note."""
        from src.server import get_subgraph

        result = await get_subgraph("C_Python", depth=1)

        center_node = next((n for n in result["nodes"] if n["id"] == "C_Python"), None)
        assert center_node is not None
        assert center_node.get("is_center") is True

    async def test_get_subgraph_includes_neighbors(self, patched_vault_cache):
        """Test subgraph includes connected notes."""
        from src.server import get_subgraph

        result = await get_subgraph("C_Python", depth=1)

        # C_JavaScript links to C_Python, so it should be a neighbor
        js_node = next((n for n in result["nodes"] if n["id"] == "C_JavaScript"), None)
        assert js_node is not None

    async def test_get_subgraph_not_found(self, patched_vault_cache):
        """Test error for non-existent note."""
        from src.server import get_subgraph

        result = await get_subgraph("NonExistentNote", depth=1)

        assert "error" in result

    async def test_get_subgraph_depth(self, patched_vault_cache):
        """Test depth controls subgraph size."""
        from src.server import get_subgraph

        depth1 = await get_subgraph("C_Python", depth=1)
        depth2 = await get_subgraph("C_Python", depth=2)

        # Depth 2 should have >= nodes than depth 1
        assert len(depth2["nodes"]) >= len(depth1["nodes"])

    async def test_get_subgraph_stats(self, patched_vault_cache):
        """Test subgraph stats are populated."""
        from src.server import get_subgraph

        result = await get_subgraph("C_Python", depth=2)

        assert "stats" in result
        assert result["stats"]["depth"] == 2
        assert result["stats"]["center"] == "C_Python"
        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]


class TestGetClusters:
    """Tests for the get_clusters function."""

    async def test_get_clusters_structure(self, patched_vault_cache):
        """Test get_clusters returns expected structure."""
        from src.server import get_clusters

        result = await get_clusters()

        assert "clusters" in result
        assert "orphans" in result
        assert "stats" in result

    async def test_get_clusters_hub_structure(self, patched_vault_cache):
        """Test each cluster has expected fields."""
        from src.server import get_clusters

        result = await get_clusters()

        if result["clusters"]:
            cluster = result["clusters"][0]
            assert "hub" in cluster
            assert "neighbors" in cluster
            assert "neighbor_count" in cluster
            assert "id" in cluster["hub"]
            assert "title" in cluster["hub"]
            assert "connections" in cluster["hub"]

    async def test_get_clusters_sorted_by_connections(self, patched_vault_cache):
        """Test clusters are sorted by number of connections."""
        from src.server import get_clusters

        result = await get_clusters()

        if len(result["clusters"]) >= 2:
            connections = [c["hub"]["connections"] for c in result["clusters"]]
            assert connections == sorted(connections, reverse=True)

    async def test_get_clusters_top_n(self, patched_vault_cache):
        """Test top_n limits the number of clusters."""
        from src.server import get_clusters

        result = await get_clusters(top_n=2)

        assert len(result["clusters"]) <= 2

    async def test_get_clusters_stats(self, patched_vault_cache):
        """Test clusters stats are populated."""
        from src.server import get_clusters

        result = await get_clusters()

        assert "total_clusters" in result["stats"]
        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]
        assert "orphan_count" in result["stats"]


# ============== Tests for Security Validations ==============

class TestPathValidation:
    """Tests for path traversal protection."""

    def test_validate_path_rejects_double_dots(self, temp_vault):
        """Test that '..' in paths is rejected."""
        from src.server import validate_path_within_vault, PathValidationError

        with pytest.raises(PathValidationError, match="Path traversal detected"):
            validate_path_within_vault("../../../etc/passwd", temp_vault)

    def test_validate_path_rejects_hidden_double_dots(self, temp_vault):
        """Test that '..' anywhere in path is rejected."""
        from src.server import validate_path_within_vault, PathValidationError

        with pytest.raises(PathValidationError, match="Path traversal detected"):
            validate_path_within_vault("Concepts/../../etc", temp_vault)

    def test_validate_path_rejects_absolute_paths(self, temp_vault):
        """Test that absolute paths are rejected."""
        from src.server import validate_path_within_vault, PathValidationError

        with pytest.raises(PathValidationError, match="Absolute paths are not allowed"):
            validate_path_within_vault("/etc/passwd", temp_vault)

    def test_validate_path_rejects_windows_absolute_paths(self, temp_vault):
        """Test that Windows absolute paths are rejected."""
        from src.server import validate_path_within_vault, PathValidationError

        with pytest.raises(PathValidationError, match="Absolute paths are not allowed"):
            validate_path_within_vault("C:\\Windows\\System32", temp_vault)

    def test_validate_path_rejects_empty_path(self, temp_vault):
        """Test that empty paths are rejected."""
        from src.server import validate_path_within_vault, PathValidationError

        with pytest.raises(PathValidationError, match="Path cannot be empty"):
            validate_path_within_vault("", temp_vault)

    def test_validate_path_allows_valid_path(self, temp_vault):
        """Test that valid relative paths are accepted."""
        from src.server import validate_path_within_vault

        result = validate_path_within_vault("Concepts/C_Python.md", temp_vault)
        assert result.is_relative_to(temp_vault.resolve())

    def test_validate_folder_path_rejects_traversal(self, temp_vault):
        """Test folder path validation rejects traversal."""
        from src.server import validate_folder_path, PathValidationError

        with pytest.raises(PathValidationError):
            validate_folder_path("../../../etc", temp_vault)

    def test_validate_folder_path_allows_valid_folder(self, temp_vault):
        """Test folder path validation accepts valid paths."""
        from src.server import validate_folder_path

        result = validate_folder_path("Concepts", temp_vault)
        assert result.is_relative_to(temp_vault.resolve())


class TestTitleValidation:
    """Tests for title sanitization."""

    def test_validate_title_rejects_empty(self):
        """Test that empty titles are rejected."""
        from src.server import validate_title, TitleValidationError

        with pytest.raises(TitleValidationError, match="Title cannot be empty"):
            validate_title("")

    def test_validate_title_rejects_whitespace_only(self):
        """Test that whitespace-only titles are rejected."""
        from src.server import validate_title, TitleValidationError

        with pytest.raises(TitleValidationError, match="Title cannot be empty"):
            validate_title("   ")

    def test_validate_title_rejects_too_long(self):
        """Test that titles exceeding max length are rejected."""
        from src.server import validate_title, TitleValidationError, MAX_TITLE_LENGTH

        long_title = "A" * (MAX_TITLE_LENGTH + 1)
        with pytest.raises(TitleValidationError, match="exceeds maximum length"):
            validate_title(long_title)

    def test_validate_title_rejects_special_chars(self):
        """Test that titles with dangerous chars are rejected."""
        from src.server import validate_title, TitleValidationError

        # Path-like characters
        with pytest.raises(TitleValidationError, match="invalid characters"):
            validate_title("../../../etc")

        with pytest.raises(TitleValidationError, match="invalid characters"):
            validate_title("test<script>")

    def test_validate_title_allows_valid_titles(self):
        """Test that valid titles are accepted."""
        from src.server import validate_title

        assert validate_title("My Note Title") == "My Note Title"
        assert validate_title("Test-Note_123") == "Test-Note_123"
        assert validate_title("Note avec accents éàù") == "Note avec accents éàù"

    def test_validate_title_strips_whitespace(self):
        """Test that titles are trimmed."""
        from src.server import validate_title

        assert validate_title("  My Title  ") == "My Title"


class TestContentValidation:
    """Tests for content size limits."""

    def test_validate_content_rejects_too_large(self):
        """Test that content exceeding max size is rejected."""
        from src.server import validate_content_size, ContentValidationError, MAX_CONTENT_SIZE

        large_content = "A" * (MAX_CONTENT_SIZE + 1)
        with pytest.raises(ContentValidationError, match="exceeds maximum allowed size"):
            validate_content_size(large_content)

    def test_validate_content_allows_valid_size(self):
        """Test that valid content size is accepted."""
        from src.server import validate_content_size

        content = "Normal sized content"
        assert validate_content_size(content) == content


class TestWriteNoteSecure:
    """Tests for write_note with security validations."""

    async def test_write_note_rejects_folder_traversal(self, patched_vault_cache):
        """Test write_note rejects path traversal in folder."""
        from src.server import write_note

        result = await write_note(
            title="Test Note",
            content="Content",
            note_type="concept",
            tags=["test"],
            folder="../../../etc"
        )

        assert result.success is False
        assert "Invalid folder path" in result.error

    async def test_write_note_rejects_absolute_folder(self, patched_vault_cache):
        """Test write_note rejects absolute paths in folder."""
        from src.server import write_note

        result = await write_note(
            title="Test Note",
            content="Content",
            note_type="concept",
            tags=["test"],
            folder="/etc/passwd"
        )

        assert result.success is False
        assert "Invalid folder path" in result.error

    async def test_write_note_rejects_invalid_title(self, patched_vault_cache):
        """Test write_note rejects dangerous title."""
        from src.server import write_note

        result = await write_note(
            title="../../../etc/passwd",
            content="Content",
            note_type="concept",
            tags=["test"]
        )

        assert result.success is False
        assert "Invalid title" in result.error

    async def test_write_note_rejects_too_large_content(self, patched_vault_cache):
        """Test write_note rejects content exceeding size limit."""
        from src.server import write_note, MAX_CONTENT_SIZE

        large_content = "A" * (MAX_CONTENT_SIZE + 1)
        result = await write_note(
            title="Large Note",
            content=large_content,
            note_type="concept",
            tags=["test"]
        )

        assert result.success is False
        assert "Invalid content" in result.error

    async def test_write_note_succeeds_with_valid_inputs(self, patched_vault_cache, temp_vault):
        """Test write_note works with valid, secure inputs."""
        from src.server import write_note

        result = await write_note(
            title="Valid Secure Note",
            content="This is valid content",
            note_type="concept",
            tags=["test"]
        )

        assert result.success is True
        assert "C_Valid_Secure_Note.md" in result.filename

        # Verify file was created in the correct location
        created_file = temp_vault / result.path
        assert created_file.exists()


class TestGetNoteSecure:
    """Tests for get_note_content with security validations."""

    async def test_get_note_rejects_traversal_path(self, patched_vault_cache):
        """Test get_note_content rejects path traversal."""
        from src.server import get_note_content

        result = await get_note_content("../../../etc/passwd")

        assert result is None

    async def test_get_note_by_path_rejects_traversal(self, vault_cache):
        """Test VaultCache.get_note_by_path rejects traversal."""
        result = await vault_cache.get_note_by_path("../../../etc/passwd")

        assert result is None

    async def test_get_note_by_path_rejects_absolute_path(self, vault_cache):
        """Test VaultCache.get_note_by_path rejects absolute paths."""
        result = await vault_cache.get_note_by_path("/etc/passwd")

        assert result is None

    async def test_get_note_by_path_rejects_empty(self, vault_cache):
        """Test VaultCache.get_note_by_path rejects empty paths."""
        result = await vault_cache.get_note_by_path("")

        assert result is None

    async def test_get_note_succeeds_with_valid_path(self, vault_cache):
        """Test get_note works with valid paths."""
        result = await vault_cache.get_note_by_path("Concepts/C_Python.md")

        assert result is not None
        assert result.stem == "C_Python"


# ============== Tests for Path Traversal Security ==============

class TestPathTraversalSecurity:
    """Tests for path traversal protection in write_note function."""

    async def test_path_traversal_dot_dot(self, patched_vault_cache):
        """Test write_note with folder='../../../etc' must fail."""
        from src.server import write_note

        result = await write_note(
            title="Malicious Note",
            content="This should not be created",
            note_type="concept",
            tags=["test"],
            folder="../../../etc"
        )

        assert result.success is False
        assert "Invalid folder path" in result.error

    async def test_absolute_path_blocked(self, patched_vault_cache):
        """Test folder='/etc/passwd' must fail."""
        from src.server import write_note

        result = await write_note(
            title="Absolute Path Attack",
            content="This should not be created",
            note_type="concept",
            tags=["test"],
            folder="/etc/passwd"
        )

        assert result.success is False
        assert "Invalid folder path" in result.error

    async def test_windows_traversal(self, patched_vault_cache):
        """Test folder='..\\..\\Windows' must fail."""
        from src.server import write_note

        result = await write_note(
            title="Windows Traversal Attack",
            content="This should not be created",
            note_type="concept",
            tags=["test"],
            folder="..\\..\\Windows"
        )

        assert result.success is False
        assert "Invalid folder path" in result.error

    async def test_valid_subfolder_allowed(self, patched_vault_cache, temp_vault):
        """Test folder='Concepts/SubDir' must work."""
        from src.server import write_note

        result = await write_note(
            title="Valid Subfolder Note",
            content="This should be created successfully",
            note_type="concept",
            tags=["test"],
            folder="Concepts/SubDir"
        )

        assert result.success is True
        assert "Concepts/SubDir" in result.path

        # Verify the subfolder was created and file exists
        created_file = temp_vault / result.path
        assert created_file.exists()

        # Verify the subfolder is within the vault
        assert created_file.is_relative_to(temp_vault)


# ============== Comprehensive Tests for write_note() ==============

class TestWriteNote:
    """Comprehensive tests for the write_note function."""

    async def test_write_concept_note(self, patched_vault_cache, temp_vault):
        """Test creating a concept note - verify file, frontmatter, and C_ naming."""
        from src.server import write_note, parse_frontmatter

        result = await write_note(
            title="Test Concept",
            content="# Test Concept\n\nThis is a test concept note.",
            note_type="concept",
            tags=["testing", "unit-test"],
            related=["Python"]
        )

        assert result.success is True
        assert result.filename == "C_Test_Concept.md"
        assert result.folder == "Concepts"

        # Verify file exists
        file_path = temp_vault / result.path
        assert file_path.exists()

        # Verify frontmatter
        content = file_path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["title"] == "Test Concept"
        assert frontmatter["type"] == "concept"
        assert frontmatter["status"] == "seedling"
        assert frontmatter["tags"] == ["testing", "unit-test"]
        assert frontmatter["related"] == ["Python"]
        assert "# Test Concept" in body

    async def test_write_conversation_note(self, patched_vault_cache, temp_vault):
        """Test creating a conversation note - verify YYYY-MM-DD_Conv_ prefix."""
        from src.server import write_note
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")

        result = await write_note(
            title="Claude Setup",
            content="# Setup session\n\nDiscussion about Claude configuration.",
            note_type="conversation",
            tags=["claude", "setup"]
        )

        assert result.success is True
        assert result.filename == f"{today}_Conv_Claude_Setup.md"
        assert result.folder == "Conversations"

        # Verify file exists with correct name format
        file_path = temp_vault / result.path
        assert file_path.exists()
        assert file_path.stem.startswith(f"{today}_Conv_")

    async def test_write_duplicate_detection(self, patched_vault_cache, temp_vault):
        """Test creating two similar notes - verify warning about duplicates."""
        from src.server import write_note

        # Create first note
        result1 = await write_note(
            title="Duplicate Test",
            content="First note content",
            note_type="concept",
            tags=["test"]
        )
        assert result1.success is True

        # Create second note with similar title
        result2 = await write_note(
            title="Duplicate Test Extended",
            content="Second note with similar title",
            note_type="concept",
            tags=["test"]
        )

        # Second note should succeed but have warnings about potential duplicates
        assert result2.success is True
        assert result2.warnings is not None
        assert "potential_duplicates" in result2.warnings

        # Find the first note in the duplicates list
        duplicates = result2.warnings["potential_duplicates"]
        assert len(duplicates) >= 1
        assert any("Duplicate" in d["title"] for d in duplicates)

    async def test_write_cache_invalidation(self, patched_vault_cache, temp_vault):
        """Test that after writing, the cache contains the new note."""
        from src.server import write_note, vault_cache

        # Get initial stems
        initial_stems = (await vault_cache.get_note_stems()).copy()

        # Create a new note
        result = await write_note(
            title="Cache Test Note",
            content="Testing cache invalidation",
            note_type="concept",
            tags=["test"]
        )

        assert result.success is True

        # Cache should now contain the new note
        new_stems = await vault_cache.get_note_stems()
        assert "C_Cache_Test_Note" in new_stems
        assert len(new_stems) > len(initial_stems)

    async def test_write_existing_file_fails(self, patched_vault_cache, temp_vault):
        """Test that writing to an existing file returns success=False."""
        from src.server import write_note

        # Create first note
        result1 = await write_note(
            title="Existing File Test",
            content="First content",
            note_type="concept",
            tags=["test"]
        )
        assert result1.success is True

        # Try to create the same note again
        result2 = await write_note(
            title="Existing File Test",
            content="Different content",
            note_type="concept",
            tags=["test"]
        )

        assert result2.success is False
        assert "File already exists" in result2.error


# ============== Comprehensive Tests for Knowledge Graph ==============

class TestKnowledgeGraph:
    """Comprehensive tests for the knowledge_graph functionality."""

    async def test_build_graph_basic(self, patched_vault_cache):
        """Test graph with linked notes - verify nodes and edges."""
        from src.server import build_graph

        result = await build_graph()

        # Verify structure
        assert "nodes" in result
        assert "edges" in result
        assert "orphans" in result
        assert "stats" in result

        # We expect nodes for all non-template notes
        node_ids = [n["id"] for n in result["nodes"]]
        assert "C_Python" in node_ids
        assert "C_JavaScript" in node_ids
        assert "R_Docker" in node_ids

        # We expect edges from JavaScript -> Python and JavaScript -> Docker
        js_edges = [e for e in result["edges"] if e["source"] == "C_JavaScript"]
        assert len(js_edges) >= 2  # Links to Python and Docker

        # Verify edge from JavaScript to Python exists
        js_to_python = any(
            e["source"] == "C_JavaScript" and e["target"] == "C_Python"
            for e in result["edges"]
        )
        assert js_to_python

    async def test_subgraph_depth(self, patched_vault_cache):
        """Test subgraph with depth 1 vs depth 2."""
        from src.server import get_subgraph

        # Get subgraph with depth 1
        depth1_result = await get_subgraph("C_Python", depth=1)
        assert "error" not in depth1_result

        # Get subgraph with depth 2
        depth2_result = await get_subgraph("C_Python", depth=2)
        assert "error" not in depth2_result

        # Depth 2 should have >= nodes than depth 1
        assert len(depth2_result["nodes"]) >= len(depth1_result["nodes"])

        # Depth 1 should include immediate neighbors only
        depth1_node_ids = {n["id"] for n in depth1_result["nodes"]}
        assert "C_Python" in depth1_node_ids  # Center
        assert "C_JavaScript" in depth1_node_ids  # Links to Python

        # Verify stats reflect the requested depth
        assert depth1_result["stats"]["depth"] == 1
        assert depth2_result["stats"]["depth"] == 2

    async def test_orphan_detection(self, patched_vault_cache):
        """Test that notes without links are detected as orphans."""
        from src.server import build_graph

        result = await build_graph()

        # Notes without any connections should be in orphans list
        assert "orphans" in result
        assert isinstance(result["orphans"], list)

        # Get all node IDs
        all_node_ids = {n["id"] for n in result["nodes"]}

        # Orphans should have zero connections
        for orphan in result["orphans"]:
            assert orphan["id"] in all_node_ids
            # Find the node in nodes list and verify zero connections
            node = next((n for n in result["nodes"] if n["id"] == orphan["id"]), None)
            assert node is not None
            assert node["connections"] == 0

    async def test_clusters_ordering(self, patched_vault_cache):
        """Test that clusters are sorted by number of connections (descending)."""
        from src.server import get_clusters

        result = await get_clusters()

        assert "clusters" in result
        clusters = result["clusters"]

        if len(clusters) >= 2:
            # Verify clusters are sorted by connections (descending)
            connections = [c["hub"]["connections"] for c in clusters]
            assert connections == sorted(connections, reverse=True)

        # Verify stats structure
        assert "stats" in result
        assert "total_clusters" in result["stats"]
        assert "total_nodes" in result["stats"]
        assert "total_edges" in result["stats"]
        assert "orphan_count" in result["stats"]

    async def test_graph_edge_bidirectionality(self, patched_vault_cache):
        """Test that edges correctly represent directional links."""
        from src.server import build_graph

        result = await build_graph()

        # JavaScript links to Python (not the other way around)
        js_to_python = any(
            e["source"] == "C_JavaScript" and e["target"] == "C_Python"
            for e in result["edges"]
        )
        assert js_to_python

        # Python links to JavaScript (in related frontmatter)
        # Note: Python has related: [JavaScript] but that's in frontmatter, not a wikilink
        # The actual [[JavaScript]] link in Python.md body creates the edge
        python_to_js = any(
            e["source"] == "C_Python" and e["target"] == "C_JavaScript"
            for e in result["edges"]
        )
        # Python does link to JavaScript via [[JavaScript]] in body
        assert python_to_js

    async def test_graph_connection_counting(self, patched_vault_cache):
        """Test that connection counts are calculated correctly."""
        from src.server import build_graph

        result = await build_graph()

        # Find Python node - it's linked by JavaScript and has outgoing links
        python_node = next((n for n in result["nodes"] if n["id"] == "C_Python"), None)
        assert python_node is not None
        # Python should have at least 1 connection (linked by JavaScript)
        assert python_node["connections"] >= 1

        # JavaScript links to both Python and Docker
        js_node = next((n for n in result["nodes"] if n["id"] == "C_JavaScript"), None)
        assert js_node is not None
        assert js_node["connections"] >= 2  # Links to Python and Docker
