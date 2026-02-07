"""
Knowledge Assistant MCP Server

Provides tools to search, explore, and query the Obsidian Knowledge vault.
Features in-memory caching for fast repeated queries.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

# Configuration
VAULT_PATH = Path(os.environ.get("KNOWLEDGE_VAULT_PATH", r"C:\Users\r2d2\Documents\Knowledge"))
INDEX_PATH = Path(os.environ.get("KNOWLEDGE_INDEX_PATH", r"C:\Users\r2d2\.claude\skills\knowledge-watcher-skill\data\notes-index.json"))
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

                links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)

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
            except Exception:
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
        """Find a note by relative path or title match."""
        self.refresh()
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

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
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

    terms = [t.strip().lower() for t in re.split(r'[\s\-_]+', query) if t.strip()]
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
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_note_content(note_path: str) -> dict | None:
    """Get full content of a specific note from cache."""
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
        except Exception:
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
        all_links = re.findall(r'\[\[([^\]]+)\]\]', note["content"])
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

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
