"""
Knowledge Assistant MCP Server

Provides tools to search, explore, and query the Obsidian Knowledge vault.
Features in-memory caching for fast repeated queries.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

# Configuration - Cross-platform default paths
def _get_default_vault_path() -> Path:
    """Get default vault path based on platform."""
    if os.name == "nt":  # Windows
        return Path.home() / "Documents" / "Knowledge"
    else:  # Linux/macOS
        return Path.home() / "Documents" / "Knowledge"


def _get_default_index_path() -> Path:
    """Get default index path based on platform."""
    if os.name == "nt":  # Windows
        return Path.home() / ".claude" / "skills" / "knowledge-watcher-skill" / "data" / "notes-index.json"
    else:  # Linux/macOS
        return Path.home() / ".knowledge" / "notes-index.json"


VAULT_PATH = Path(os.environ.get("KNOWLEDGE_VAULT_PATH", str(_get_default_vault_path())))
INDEX_PATH = Path(os.environ.get("KNOWLEDGE_INDEX_PATH", str(_get_default_index_path())))
CACHE_TTL = int(os.environ.get("KNOWLEDGE_CACHE_TTL", "60"))  # seconds

# Initialize server
server = Server("knowledge-assistant")


# ============== In-Memory Cache ==============

class VaultCache:
    """In-memory cache for vault notes. Avoids repeated filesystem scans."""

    def __init__(self, vault_path: Path, ttl: int = 60):
        self.vault_path = vault_path
        self.ttl = ttl
        self._notes: list[dict] = []  # [{path, rel_path, stem, content, content_lower, frontmatter, body, links, tags, mtime}]
        self._loaded_at: float = 0
        self._note_stems: set[str] = set()  # For quick basename lookup

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._loaded_at) > self.ttl

    def refresh(self, force: bool = False) -> None:
        """Reload all notes from disk if cache is stale."""
        if not force and not self.is_stale:
            return

        notes = []
        stems = set()

        for note_file in self.vault_path.rglob("*.md"):
            rel_path = note_file.relative_to(self.vault_path)
            # Skip hidden folders
            if any(part.startswith(".") for part in rel_path.parts):
                continue

            try:
                content = note_file.read_text(encoding="utf-8")
                frontmatter, body = parse_frontmatter(content)
                stem = note_file.stem
                content_lower = content.lower()

                links = WIKILINK_PATTERN.findall(content)

                notes.append({
                    "path": note_file,
                    "rel_path": rel_path,
                    "rel_path_str": str(rel_path),
                    "stem": stem,
                    "stem_lower": stem.lower(),
                    "content": content,
                    "content_lower": content_lower,
                    "frontmatter": frontmatter,
                    "body": body,
                    "body_lower": body.lower(),
                    "links": list(set(links)),
                    "tags": frontmatter.get("tags", []),
                    "type": frontmatter.get("type", "unknown"),
                    "date": frontmatter.get("date", ""),
                    "mtime": note_file.stat().st_mtime,
                    "word_count": len(body.split()),
                    "is_template": any(part.startswith("_Template") for part in rel_path.parts),
                })
                stems.add(stem)
            except Exception as e:
                logger.warning("Failed to read note %s: %s", note_file, e)
                continue

        self._notes = notes
        self._note_stems = stems
        self._loaded_at = time.time()

    def get_notes(self, include_templates: bool = False) -> list[dict]:
        """Get all cached notes."""
        self.refresh()
        if include_templates:
            return self._notes
        return [n for n in self._notes if not n["is_template"]]

    def get_note_by_path(self, note_path: str) -> dict | None:
        """Find a note by relative path or title match.

        Security: Validates path to prevent directory traversal attacks.
        """
        self.refresh()

        # Security validation: reject path traversal attempts
        if not note_path or ".." in note_path:
            return None

        # Reject absolute paths
        if note_path.startswith("/") or (len(note_path) > 1 and note_path[1] == ":"):
            return None

        # Exact path match
        for note in self._notes:
            if note["rel_path_str"] == note_path:
                return note
        # Title/stem match
        path_lower = note_path.lower()
        for note in self._notes:
            if path_lower in note["stem_lower"]:
                return note
        return None

    @property
    def note_count(self) -> int:
        self.refresh()
        return len(self._notes)

    @property
    def note_stems(self) -> set[str]:
        self.refresh()
        return self._note_stems


# Global cache instance
vault_cache = VaultCache(VAULT_PATH, CACHE_TTL)


# ============== Security Validation ==============

# Maximum allowed content size (1MB)
MAX_CONTENT_SIZE = 1 * 1024 * 1024  # 1MB in bytes

# Maximum allowed title length
MAX_TITLE_LENGTH = 200

# Allowed characters in title (alphanumeric, spaces, hyphens, underscores, accented chars)
TITLE_PATTERN = re.compile(r'^[\w\s\-àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]+$', re.UNICODE)

# Pre-compiled regex patterns for performance
WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
SEARCH_SPLIT_PATTERN = re.compile(r'[\s\-_]+')
ALL_LINKS_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
UNSAFE_CHARS_PATTERN = re.compile(r'[^\w\s-]')
WHITESPACE_PATTERN = re.compile(r'[\s]+')
WORD_SPLIT_PATTERN = re.compile(r'[\s_-]+')


class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass


class TitleValidationError(Exception):
    """Raised when title validation fails."""
    pass


class ContentValidationError(Exception):
    """Raised when content validation fails."""
    pass


def validate_path_within_vault(path_str: str, vault_path: Path) -> Path:
    """Validate that a path is safely within the vault directory.

    Args:
        path_str: The path string to validate (relative path or note identifier)
        vault_path: The vault root path

    Returns:
        The validated absolute Path

    Raises:
        PathValidationError: If the path attempts to escape the vault
    """
    # Reject empty paths
    if not path_str or not path_str.strip():
        raise PathValidationError("Path cannot be empty")

    # Reject paths with ".." components (path traversal attempt)
    if ".." in path_str:
        raise PathValidationError("Path traversal detected: '..' is not allowed")

    # Reject absolute paths
    if path_str.startswith("/") or (len(path_str) > 1 and path_str[1] == ":"):
        raise PathValidationError("Absolute paths are not allowed")

    # Build the full path and resolve it
    full_path = (vault_path / path_str).resolve()
    vault_resolved = vault_path.resolve()

    # Verify the resolved path is within the vault
    try:
        full_path.relative_to(vault_resolved)
    except ValueError:
        raise PathValidationError(f"Path escapes vault directory: {path_str}")

    return full_path


def validate_folder_path(folder: str, vault_path: Path) -> Path:
    """Validate a folder path for note creation.

    Args:
        folder: The folder path relative to vault
        vault_path: The vault root path

    Returns:
        The validated folder Path

    Raises:
        PathValidationError: If the folder path is invalid or escapes the vault
    """
    return validate_path_within_vault(folder, vault_path)


def validate_title(title: str) -> str:
    """Validate and sanitize a note title.

    Args:
        title: The title to validate

    Returns:
        The validated title

    Raises:
        TitleValidationError: If the title is invalid
    """
    if not title or not title.strip():
        raise TitleValidationError("Title cannot be empty")

    title = title.strip()

    if len(title) > MAX_TITLE_LENGTH:
        raise TitleValidationError(f"Title exceeds maximum length of {MAX_TITLE_LENGTH} characters")

    # Check for dangerous characters that could cause issues in filenames
    # Allow alphanumeric, spaces, hyphens, underscores, and common accented characters
    if not TITLE_PATTERN.match(title):
        raise TitleValidationError(
            "Title contains invalid characters. Only alphanumeric, spaces, hyphens, "
            "underscores, and common accented characters are allowed."
        )

    return title


def validate_content_size(content: str) -> str:
    """Validate content size.

    Args:
        content: The content to validate

    Returns:
        The validated content

    Raises:
        ContentValidationError: If the content exceeds size limits
    """
    content_bytes = len(content.encode('utf-8'))

    if content_bytes > MAX_CONTENT_SIZE:
        max_mb = MAX_CONTENT_SIZE / (1024 * 1024)
        actual_mb = content_bytes / (1024 * 1024)
        raise ContentValidationError(
            f"Content size ({actual_mb:.2f}MB) exceeds maximum allowed size ({max_mb}MB)"
        )

    return content


# ============== Helper Functions ==============

def load_index() -> dict:
    """Load the notes index."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"notes": [], "terms": {}}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from note content."""
    frontmatter = {}
    body = content

    match = FRONTMATTER_PATTERN.match(content)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            pass
        body = content[match.end():]

    return frontmatter, body


# ============== Search & Query Functions (cached) ==============

def search_notes(query: str, max_results: int = 10) -> list[dict]:
    """Search notes by content or title using cached data.

    Supports multi-word queries: all words must be present (AND logic).
    """
    results = []

    terms = [t.strip().lower() for t in SEARCH_SPLIT_PATTERN.split(query) if t.strip()]
    if not terms:
        return []

    for note in vault_cache.get_notes():
        try:
            # Check if ALL terms are present (AND logic)
            all_terms_found = all(
                term in note["stem_lower"] or term in note["content_lower"]
                for term in terms
            )

            if not all_terms_found:
                continue

            # Calculate score
            score = 0
            for term in terms:
                if term in note["stem_lower"]:
                    score += 10
                score += note["content_lower"].count(term)

            # Extract snippet
            snippet_idx = -1
            for term in terms:
                idx = note["body_lower"].find(term)
                if idx >= 0:
                    snippet_idx = idx
                    break

            if snippet_idx >= 0:
                start = max(0, snippet_idx - 50)
                end = min(len(note["body"]), snippet_idx + 150)
                snippet = "..." + note["body"][start:end].replace("\n", " ") + "..."
            else:
                snippet = note["body"][:200].replace("\n", " ") + "..."

            results.append({
                "title": note["stem"],
                "path": note["rel_path_str"],
                "score": score,
                "snippet": snippet,
                "tags": note["tags"],
                "type": note["type"],
                "date": note["date"],
                "matched_terms": terms,
            })
        except Exception as e:
            logger.warning("Failed to score note %s: %s", note["rel_path_str"], e)
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_note_content(note_path: str) -> dict | None:
    """Get full content of a specific note from cache.

    Security: Validates path before lookup to prevent directory traversal.
    """
    # Early validation - reject obvious traversal attempts
    if not note_path or ".." in note_path:
        return None

    # Additional validation for explicit path lookups
    if "/" in note_path or "\\" in note_path:
        try:
            validate_path_within_vault(note_path, VAULT_PATH)
        except PathValidationError:
            return None

    note = vault_cache.get_note_by_path(note_path)
    if not note:
        return None

    return {
        "title": note["stem"],
        "path": note["rel_path_str"],
        "frontmatter": note["frontmatter"],
        "body": note["body"],
        "links": note["links"],
        "word_count": note["word_count"],
    }


def get_related_notes(concept: str, max_results: int = 10) -> list[dict]:
    """Find notes related to a concept using cached data."""
    results = []
    concept_lower = concept.lower()

    for note in vault_cache.get_notes():
        try:
            score = 0
            reasons = []

            # Check links
            for link in note["links"]:
                if concept_lower in link.lower():
                    score += 5
                    reasons.append(f"links to [[{link}]]")

            # Check tags
            for tag in note["tags"]:
                if concept_lower in str(tag).lower():
                    score += 3
                    reasons.append(f"tagged #{tag}")

            # Check title
            if concept_lower in note["stem_lower"]:
                score += 10
                reasons.append("title match")

            # Check content
            if concept_lower in note["body_lower"]:
                mentions = note["body_lower"].count(concept_lower)
                score += mentions
                reasons.append(f"mentioned {mentions}x")

            if score > 0:
                results.append({
                    "title": note["stem"],
                    "path": note["rel_path_str"],
                    "score": score,
                    "reasons": reasons[:3],
                    "type": note["type"],
                })
        except Exception as e:
            logger.warning("Failed to compute relevance for note %s: %s", note["rel_path_str"], e)
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_vault_stats() -> dict:
    """Get statistics about the vault using cached data."""
    stats = {
        "total_notes": 0,
        "by_type": {},
        "by_folder": {},
        "total_words": 0,
        "total_links": 0,
        "tags": {},
        "recent_notes": [],
    }

    notes_with_dates = []

    for note in vault_cache.get_notes(include_templates=False):
        # Skip hidden
        if any(part.startswith(".") for part in note["rel_path"].parts):
            continue

        stats["total_notes"] += 1
        stats["total_words"] += note["word_count"]

        # Count links (including aliased)
        all_links = ALL_LINKS_PATTERN.findall(note["content"])
        stats["total_links"] += len(all_links)

        # By type
        note_type = note["type"]
        stats["by_type"][note_type] = stats["by_type"].get(note_type, 0) + 1

        # By folder
        folder = note["rel_path"].parts[0] if len(note["rel_path"].parts) > 1 else "root"
        stats["by_folder"][folder] = stats["by_folder"].get(folder, 0) + 1

        # Tags
        for tag in note["tags"]:
            tag_str = str(tag)
            stats["tags"][tag_str] = stats["tags"].get(tag_str, 0) + 1

        notes_with_dates.append((note["stem"], note["rel_path_str"], note["mtime"]))

    # Recent notes
    notes_with_dates.sort(key=lambda x: x[2], reverse=True)
    stats["recent_notes"] = [
        {"title": n[0], "path": n[1]}
        for n in notes_with_dates[:10]
    ]

    # Sort tags by count
    stats["top_tags"] = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)[:20]

    return stats


def explore_by_tag(tag: str) -> list[dict]:
    """Find all notes with a specific tag using cached data."""
    results = []
    tag_lower = tag.lower().replace("#", "")

    for note in vault_cache.get_notes():
        tags_lower = [str(t).lower() for t in note["tags"]]
        if any(tag_lower in t for t in tags_lower):
            results.append({
                "title": note["stem"],
                "path": note["rel_path_str"],
                "tags": note["tags"],
                "type": note["type"],
            })

    return results


def get_backlinks(note_title: str) -> list[dict]:
    """Find all notes that link to a specific note using cached data."""
    results = []
    title_lower = note_title.lower()

    for note in vault_cache.get_notes():
        for link in note["links"]:
            if title_lower in link.lower():
                results.append({
                    "title": note["stem"],
                    "path": note["rel_path_str"],
                    "link_text": link,
                    "type": note["type"],
                })
                break

    return results


# ============== Graph Functions ==============

def _find_note_by_link(link: str, stem_to_note: dict[str, dict]) -> dict | None:
    """Find a note by link text, handling partial matches.

    Link text may be:
    - Exact stem match: "C_Python" -> C_Python
    - Partial match: "Python" -> C_Python (if stem contains "python")
    """
    link_lower = link.lower()

    # Exact match first
    if link_lower in stem_to_note:
        return stem_to_note[link_lower]

    # Partial match: link text is contained in stem
    for stem, note in stem_to_note.items():
        if link_lower in stem:
            return note

    return None


def build_graph() -> dict:
    """Build a complete graph of all notes and their links.

    Returns:
        dict with:
        - nodes: list of {id, title, path, type, connections}
        - edges: list of {source, target}
        - orphans: list of notes with no links
        - stats: global graph statistics
    """
    notes = vault_cache.get_notes()

    # Build a mapping of note stems (lowercase) to note data
    stem_to_note: dict[str, dict] = {}
    for note in notes:
        stem_to_note[note["stem_lower"]] = note

    nodes = []
    edges = []
    orphans = []

    # Track connections per note
    connection_counts: dict[str, int] = {}

    for note in notes:
        note_id = note["stem"]

        # Count outgoing links
        outgoing = 0
        for link in note["links"]:
            # Find the target note (handles partial matches)
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note:
                edges.append({
                    "source": note_id,
                    "target": target_note["stem"],
                })
                outgoing += 1
                # Increment target's incoming count
                target_stem = target_note["stem"]
                connection_counts[target_stem] = connection_counts.get(target_stem, 0) + 1

        connection_counts[note_id] = connection_counts.get(note_id, 0) + outgoing

    # Build nodes with connection counts
    for note in notes:
        note_id = note["stem"]
        conn_count = connection_counts.get(note_id, 0)

        node = {
            "id": note_id,
            "title": note["stem"],
            "path": note["rel_path_str"],
            "type": note["type"],
            "connections": conn_count,
        }
        nodes.append(node)

        # Orphan = no outgoing links AND no incoming links
        if conn_count == 0:
            orphans.append({
                "id": note_id,
                "title": note["stem"],
                "path": note["rel_path_str"],
                "type": note["type"],
            })

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "orphan_count": len(orphans),
        "avg_connections": round(sum(connection_counts.values()) / len(nodes), 2) if nodes else 0,
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "orphans": orphans,
        "stats": stats,
    }


def get_subgraph(center_note: str, depth: int = 2) -> dict:
    """Get a subgraph centered on a specific note.

    Args:
        center_note: Title or path of the center note
        depth: How many levels of connections to include (default: 2)

    Returns:
        dict with nodes[], edges[], center, and stats
    """
    notes = vault_cache.get_notes()

    # Build stem mappings
    stem_to_note: dict[str, dict] = {}
    for note in notes:
        stem_to_note[note["stem_lower"]] = note

    # Find center note
    center_lower = center_note.lower()
    center_data = None

    # Try exact match first
    if center_lower in stem_to_note:
        center_data = stem_to_note[center_lower]
    else:
        # Try partial match
        for stem, note in stem_to_note.items():
            if center_lower in stem:
                center_data = note
                break

    if not center_data:
        return {"error": f"Note not found: {center_note}"}

    center_id = center_data["stem"]

    # BFS to find all nodes within depth
    visited: set[str] = {center_id}
    frontier: set[str] = {center_id}
    subgraph_nodes: dict[str, dict] = {center_id: center_data}

    # Build reverse links (backlinks) using the same matching logic
    backlinks: dict[str, list[str]] = {}
    for note in notes:
        for link in note["links"]:
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note:
                target_stem = target_note["stem"]
                if target_stem not in backlinks:
                    backlinks[target_stem] = []
                backlinks[target_stem].append(note["stem"])

    for _ in range(depth):
        new_frontier: set[str] = set()

        for node_id in frontier:
            node_data = stem_to_note.get(node_id.lower())
            if not node_data:
                continue

            # Outgoing links
            for link in node_data["links"]:
                target_note = _find_note_by_link(link, stem_to_note)
                if target_note:
                    target_stem = target_note["stem"]
                    if target_stem not in visited:
                        visited.add(target_stem)
                        new_frontier.add(target_stem)
                        subgraph_nodes[target_stem] = target_note

            # Incoming links (backlinks)
            if node_id in backlinks:
                for source_id in backlinks[node_id]:
                    if source_id not in visited:
                        visited.add(source_id)
                        new_frontier.add(source_id)
                        subgraph_nodes[source_id] = stem_to_note[source_id.lower()]

        frontier = new_frontier

    # Build edges for the subgraph
    edges = []
    for node_id, node_data in subgraph_nodes.items():
        for link in node_data["links"]:
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note and target_note["stem"] in subgraph_nodes:
                edges.append({
                    "source": node_id,
                    "target": target_note["stem"],
                })

    # Build nodes list with connection counts within subgraph
    nodes = []
    for node_id, node_data in subgraph_nodes.items():
        conn_count = sum(1 for e in edges if e["source"] == node_id or e["target"] == node_id)
        nodes.append({
            "id": node_id,
            "title": node_data["stem"],
            "path": node_data["rel_path_str"],
            "type": node_data["type"],
            "connections": conn_count,
            "is_center": node_id == center_id,
        })

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "depth": depth,
        "center": center_id,
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "center": center_id,
        "stats": stats,
    }


def get_clusters(top_n: int = 10) -> dict:
    """Get the most connected clusters in the graph.

    Returns the top N most connected notes and their immediate neighbors.

    Args:
        top_n: Number of top connected notes to return (default: 10)

    Returns:
        dict with clusters[], stats
    """
    graph = build_graph()

    # Sort nodes by connections
    sorted_nodes = sorted(graph["nodes"], key=lambda x: x["connections"], reverse=True)
    top_nodes = sorted_nodes[:top_n]

    # Build clusters for each top node
    clusters = []
    for node in top_nodes:
        # Find immediate neighbors (connected nodes)
        neighbors = set()
        for edge in graph["edges"]:
            if edge["source"] == node["id"]:
                neighbors.add(edge["target"])
            elif edge["target"] == node["id"]:
                neighbors.add(edge["source"])

        clusters.append({
            "hub": {
                "id": node["id"],
                "title": node["title"],
                "path": node["path"],
                "type": node["type"],
                "connections": node["connections"],
            },
            "neighbors": list(neighbors),
            "neighbor_count": len(neighbors),
        })

    stats = {
        "total_clusters": len(clusters),
        "total_nodes": graph["stats"]["total_nodes"],
        "total_edges": graph["stats"]["total_edges"],
        "orphan_count": graph["stats"]["orphan_count"],
    }

    return {
        "clusters": clusters,
        "orphans": graph["orphans"],
        "stats": stats,
    }


# ============== Write Functions ==============

# Mapping of note types to naming prefixes and default folders
NOTE_TYPE_CONFIG = {
    "concept": {"prefix": "C_", "folder": "Concepts"},
    "conversation": {"prefix": "{date}_Conv_", "folder": "Conversations"},
    "troubleshooting": {"prefix": "{date}_Fix_", "folder": "Troubleshooting"},
    "session": {"prefix": "{date}_Session_", "folder": "Sessions"},
    "reference": {"prefix": "R_", "folder": "References"},
    "project": {"prefix": "P_", "folder": "Projects"},
}


def generate_filename(title: str, note_type: str) -> str:
    """Generate filename based on note type and naming conventions.

    Conventions:
    - concept: C_<title>.md
    - conversation: YYYY-MM-DD_Conv_<title>.md
    - troubleshooting: YYYY-MM-DD_Fix_<title>.md
    - session: YYYY-MM-DD_Session_<title>.md
    - reference: R_<title>.md
    - project: P_<title>.md
    """
    today = datetime.now().strftime("%Y-%m-%d")
    config = NOTE_TYPE_CONFIG.get(note_type, {"prefix": "", "folder": "Notes"})
    prefix = config["prefix"].format(date=today)

    # Sanitize title: remove special chars, replace spaces with underscores
    safe_title = UNSAFE_CHARS_PATTERN.sub('', title)
    safe_title = WHITESPACE_PATTERN.sub('_', safe_title.strip())

    return f"{prefix}{safe_title}.md"


def get_default_folder(note_type: str) -> str:
    """Get default folder for a note type."""
    config = NOTE_TYPE_CONFIG.get(note_type, {"folder": "Notes"})
    return config["folder"]


def generate_frontmatter(title: str, note_type: str, tags: list[str], related: list[str] | None = None) -> str:
    """Generate YAML frontmatter for a note.

    Required fields: title, date, type, status, tags, related
    """
    today = datetime.now().strftime("%Y-%m-%d")

    frontmatter = {
        "title": title,
        "date": today,
        "type": note_type,
        "status": "seedling",
        "tags": tags if tags else [],
        "related": related if related else [],
    }

    yaml_content = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_content}---\n\n"


def check_duplicates(title: str, note_type: str) -> list[dict]:
    """Check for potential duplicate notes.

    Returns a list of notes that might be duplicates based on:
    - Similar title (case-insensitive)
    - Same type with similar content keywords
    """
    duplicates = []
    title_lower = title.lower()
    title_words = set(WORD_SPLIT_PATTERN.split(title_lower))

    for note in vault_cache.get_notes():
        # Check for exact title match
        if note["stem_lower"] == title_lower:
            duplicates.append({
                "title": note["stem"],
                "path": note["rel_path_str"],
                "type": note["type"],
                "match_type": "exact_title",
            })
            continue

        # Check for similar title (word overlap)
        note_words = set(WORD_SPLIT_PATTERN.split(note["stem_lower"]))
        overlap = title_words & note_words
        if len(overlap) >= 2 and len(overlap) / len(title_words) > 0.5:
            duplicates.append({
                "title": note["stem"],
                "path": note["rel_path_str"],
                "type": note["type"],
                "match_type": "similar_title",
                "common_words": list(overlap),
            })

    return duplicates[:5]  # Return at most 5 potential duplicates


def write_note(title: str, content: str, note_type: str, tags: list[str], folder: str | None = None, related: list[str] | None = None) -> dict:
    """Create a new note in the vault.

    Args:
        title: Note title
        content: Note body content (markdown)
        note_type: Type of note (concept, conversation, troubleshooting, etc.)
        tags: List of tags
        folder: Optional folder path (defaults based on type)
        related: Optional list of related note titles

    Returns:
        dict with status, path, and any warnings
    """
    # === Security validations ===

    # Validate title
    try:
        title = validate_title(title)
    except TitleValidationError as e:
        return {
            "success": False,
            "error": f"Invalid title: {str(e)}",
        }

    # Validate content size
    try:
        validate_content_size(content)
    except ContentValidationError as e:
        return {
            "success": False,
            "error": f"Invalid content: {str(e)}",
        }

    # Determine and validate folder path
    target_folder = folder if folder else get_default_folder(note_type)

    try:
        folder_path = validate_folder_path(target_folder, VAULT_PATH)
    except PathValidationError as e:
        return {
            "success": False,
            "error": f"Invalid folder path: {str(e)}",
        }

    # Check for potential duplicates first
    duplicates = check_duplicates(title, note_type)

    # Create folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = generate_filename(title, note_type)
    file_path = folder_path / filename

    # Final validation: ensure the complete file path is within vault
    try:
        validated_file_path = validate_path_within_vault(
            str(file_path.relative_to(VAULT_PATH.resolve())), VAULT_PATH
        )
    except (PathValidationError, ValueError) as e:
        return {
            "success": False,
            "error": f"Invalid file path: {str(e)}",
        }

    # Check if file already exists
    if file_path.exists():
        return {
            "success": False,
            "error": f"File already exists: {file_path.relative_to(VAULT_PATH)}",
            "path": str(file_path.relative_to(VAULT_PATH)),
        }

    # Generate frontmatter and full content
    frontmatter = generate_frontmatter(title, note_type, tags, related)
    full_content = frontmatter + content

    # Write the file
    try:
        file_path.write_text(full_content, encoding="utf-8")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write file: {str(e)}",
        }

    # Invalidate cache to include the new note
    vault_cache.refresh(force=True)

    result = {
        "success": True,
        "path": str(file_path.relative_to(VAULT_PATH)),
        "filename": filename,
        "folder": target_folder,
    }

    if duplicates:
        result["warnings"] = {
            "potential_duplicates": duplicates,
            "message": f"Found {len(duplicates)} potential duplicate(s). Review recommended.",
        }

    return result


# ============== MCP Tools ==============

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="knowledge_search",
            description="Search notes in the Knowledge vault by content or title. Returns matching notes with snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords, phrase, or concept)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="knowledge_read",
            description="Read the full content of a specific note. Use the path from search results or the note title.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Path to the note (e.g., 'Concepts/C_Zettelkasten.md') or note title"
                    }
                },
                "required": ["note_path"]
            }
        ),
        Tool(
            name="knowledge_related",
            description="Find notes related to a concept (by links, tags, or mentions).",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept or topic to find related notes for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["concept"]
            }
        ),
        Tool(
            name="knowledge_stats",
            description="Get statistics about the Knowledge vault (total notes, types, tags, recent notes).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="knowledge_explore_tag",
            description="Find all notes with a specific tag.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Tag to search for (with or without #)"
                    }
                },
                "required": ["tag"]
            }
        ),
        Tool(
            name="knowledge_backlinks",
            description="Find all notes that link to a specific note (backlinks).",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_title": {
                        "type": "string",
                        "description": "Title of the note to find backlinks for"
                    }
                },
                "required": ["note_title"]
            }
        ),
        Tool(
            name="knowledge_recent",
            description="Get recently modified notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent notes to return (default: 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="knowledge_write",
            description="Create a new note in the Knowledge vault with proper frontmatter and naming conventions. "
                       "Naming conventions: C_ for concepts, YYYY-MM-DD_Conv_ for conversations, "
                       "YYYY-MM-DD_Fix_ for troubleshooting notes. Checks for duplicates before creation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the note (will be used in filename and frontmatter)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Body content of the note in Markdown format"
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of note: concept, conversation, troubleshooting, session, reference, project",
                        "enum": ["concept", "conversation", "troubleshooting", "session", "reference", "project"]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags for the note (without # prefix)"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Optional folder path. Defaults based on note type (e.g., 'Concepts' for concept)"
                    },
                    "related": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of related note titles for the 'related' frontmatter field"
                    }
                },
                "required": ["title", "content", "type", "tags"]
            }
        ),
        Tool(
            name="knowledge_graph",
            description="Generate a graph view of links between notes. Without center_note, returns the most connected clusters. "
                       "With center_note, returns the subgraph around that note. Output is JSON with nodes[] and edges[] "
                       "compatible with graph visualizations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "center_note": {
                        "type": "string",
                        "description": "Optional. Title or path of the note to center the graph on. "
                                      "If not provided, returns clusters of most connected notes."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Depth of connections to include when center_note is provided (default: 2)",
                        "default": 2
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'json' for raw JSON, 'summary' for human-readable text (default: json)",
                        "enum": ["json", "summary"],
                        "default": "json"
                    }
                }
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "knowledge_search":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        results = search_notes(query, max_results)

        if not results:
            return [TextContent(type="text", text=f"No notes found for query: '{query}'")]

        output = f"Found {len(results)} notes for '{query}':\n\n"
        for r in results:
            output += f"**{r['title']}** ({r['path']})\n"
            tags_str = ', '.join(str(t) for t in r['tags'][:3]) if r['tags'] else 'none'
            output += f"  Type: {r['type']} | Tags: {tags_str}\n"
            output += f"  {r['snippet']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_read":
        note_path = arguments.get("note_path", "")
        note = get_note_content(note_path)

        if not note:
            return [TextContent(type="text", text=f"Note not found: '{note_path}'")]

        output = f"# {note['title']}\n\n"
        output += f"**Path:** {note['path']}\n"
        output += f"**Type:** {note['frontmatter'].get('type', 'unknown')}\n"
        tags_list = note['frontmatter'].get('tags', [])
        output += f"**Tags:** {', '.join(str(t) for t in tags_list)}\n"
        output += f"**Words:** {note['word_count']}\n"
        output += f"**Links:** {', '.join(note['links'][:10])}\n\n"
        output += "---\n\n"
        output += note['body']

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_related":
        concept = arguments.get("concept", "")
        max_results = arguments.get("max_results", 10)
        results = get_related_notes(concept, max_results)

        if not results:
            return [TextContent(type="text", text=f"No notes related to: '{concept}'")]

        output = f"Found {len(results)} notes related to '{concept}':\n\n"
        for r in results:
            output += f"**{r['title']}** ({r['path']})\n"
            output += f"  Reasons: {', '.join(r['reasons'])}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_stats":
        stats = get_vault_stats()

        output = "# Knowledge Vault Statistics\n\n"
        output += f"**Total Notes:** {stats['total_notes']}\n"
        output += f"**Total Words:** {stats['total_words']:,}\n"
        output += f"**Total Links:** {stats['total_links']:,}\n\n"

        output += "## By Type\n"
        for t, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
            output += f"- {t}: {count}\n"

        output += "\n## By Folder\n"
        for f, count in sorted(stats['by_folder'].items(), key=lambda x: x[1], reverse=True)[:10]:
            output += f"- {f}: {count}\n"

        output += "\n## Top Tags\n"
        for tag, count in stats['top_tags'][:15]:
            output += f"- #{tag}: {count}\n"

        output += "\n## Recent Notes\n"
        for note in stats['recent_notes']:
            output += f"- {note['title']}\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_explore_tag":
        tag = arguments.get("tag", "")
        results = explore_by_tag(tag)

        if not results:
            return [TextContent(type="text", text=f"No notes found with tag: '{tag}'")]

        output = f"Found {len(results)} notes with tag '#{tag}':\n\n"
        for r in results:
            output += f"- **{r['title']}** ({r['type']})\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_backlinks":
        note_title = arguments.get("note_title", "")
        results = get_backlinks(note_title)

        if not results:
            return [TextContent(type="text", text=f"No backlinks found for: '{note_title}'")]

        output = f"Found {len(results)} notes linking to '{note_title}':\n\n"
        for r in results:
            output += f"- **{r['title']}** ({r['type']}) - links via [[{r['link_text']}]]\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_recent":
        count = arguments.get("count", 10)
        stats = get_vault_stats()
        recent = stats['recent_notes'][:count]

        output = f"Last {len(recent)} modified notes:\n\n"
        for note in recent:
            output += f"- **{note['title']}** ({note['path']})\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_write":
        title = arguments.get("title", "")
        content = arguments.get("content", "")
        note_type = arguments.get("type", "")
        tags = arguments.get("tags", [])
        folder = arguments.get("folder")
        related = arguments.get("related")

        if not title or not content or not note_type:
            return [TextContent(type="text", text="Error: title, content, and type are required")]

        if note_type not in NOTE_TYPE_CONFIG:
            valid_types = ", ".join(NOTE_TYPE_CONFIG.keys())
            return [TextContent(type="text", text=f"Error: Invalid type '{note_type}'. Valid types: {valid_types}")]

        result = write_note(title, content, note_type, tags, folder, related)

        if not result["success"]:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"# Note Created Successfully\n\n"
        output += f"**Path:** {result['path']}\n"
        output += f"**Filename:** {result['filename']}\n"
        output += f"**Folder:** {result['folder']}\n"

        if "warnings" in result:
            output += f"\n## Warnings\n"
            output += f"{result['warnings']['message']}\n\n"
            for dup in result['warnings']['potential_duplicates']:
                output += f"- **{dup['title']}** ({dup['path']}) - {dup['match_type']}\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_graph":
        center_note = arguments.get("center_note")
        depth = arguments.get("depth", 2)
        output_format = arguments.get("format", "json")

        if center_note:
            # Return subgraph around a specific note
            result = get_subgraph(center_note, depth)

            if "error" in result:
                return [TextContent(type="text", text=f"Error: {result['error']}")]

            if output_format == "summary":
                output = f"# Graph around '{result['center']}'\n\n"
                output += f"**Depth:** {result['stats']['depth']}\n"
                output += f"**Nodes:** {result['stats']['total_nodes']}\n"
                output += f"**Edges:** {result['stats']['total_edges']}\n\n"

                output += "## Nodes (by connections)\n"
                sorted_nodes = sorted(result["nodes"], key=lambda x: x["connections"], reverse=True)
                for node in sorted_nodes:
                    center_marker = " [CENTER]" if node.get("is_center") else ""
                    output += f"- **{node['title']}** ({node['connections']} connections){center_marker}\n"

                output += "\n## Edges\n"
                for edge in result["edges"][:50]:
                    output += f"- {edge['source']} → {edge['target']}\n"

                if len(result["edges"]) > 50:
                    output += f"\n... and {len(result['edges']) - 50} more edges\n"

                return [TextContent(type="text", text=output)]
            else:
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            # Return clusters of most connected notes
            result = get_clusters()

            if output_format == "summary":
                output = "# Knowledge Graph Clusters\n\n"
                output += f"**Total Nodes:** {result['stats']['total_nodes']}\n"
                output += f"**Total Edges:** {result['stats']['total_edges']}\n"
                output += f"**Orphan Notes:** {result['stats']['orphan_count']}\n\n"

                output += "## Top Connected Hubs\n"
                for cluster in result["clusters"]:
                    hub = cluster["hub"]
                    output += f"\n### {hub['title']} ({hub['connections']} connections)\n"
                    output += f"Type: {hub['type']} | Path: {hub['path']}\n"
                    if cluster["neighbors"]:
                        output += f"Neighbors: {', '.join(cluster['neighbors'][:10])}"
                        if len(cluster["neighbors"]) > 10:
                            output += f" ... (+{len(cluster['neighbors']) - 10} more)"
                        output += "\n"

                if result["orphans"]:
                    output += f"\n## Orphan Notes ({len(result['orphans'])})\n"
                    for orphan in result["orphans"][:20]:
                        output += f"- {orphan['title']} ({orphan['type']})\n"
                    if len(result["orphans"]) > 20:
                        output += f"\n... and {len(result['orphans']) - 20} more orphans\n"

                return [TextContent(type="text", text=output)]
            else:
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============== Resources ==============

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="vault://stats",
            name="Vault Statistics",
            description="Statistics about the Knowledge vault",
            mimeType="application/json"
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "vault://stats":
        stats = get_vault_stats()
        return json.dumps(stats, indent=2)

    return json.dumps({"error": f"Unknown resource: {uri}"})


def main():
    """Main entry point."""
    import sys

    logging.basicConfig(level=logging.WARNING)

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
