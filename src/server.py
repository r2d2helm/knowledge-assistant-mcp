"""
Knowledge Assistant MCP Server

Provides tools to search, explore, and query the Obsidian Knowledge vault.
"""

import asyncio
import json
import os
import re
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

# Initialize server
server = Server("knowledge-assistant")


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


def search_notes(query: str, max_results: int = 10) -> list[dict]:
    """Search notes by content or title.

    Supports multi-word queries: all words must be present (AND logic).
    Words can be in any order and location in the document.
    """
    results = []

    # Split query into individual terms, filter empty strings
    terms = [t.strip().lower() for t in re.split(r'[\s\-_]+', query) if t.strip()]
    if not terms:
        return []

    for note_file in VAULT_PATH.rglob("*.md"):
        # Skip templates and hidden folders
        rel_path = note_file.relative_to(VAULT_PATH)
        if any(part.startswith("_Template") or part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = note_file.read_text(encoding="utf-8")
            content_lower = content.lower()
            title = note_file.stem
            title_lower = title.lower()

            # Check if ALL terms are present (AND logic)
            all_terms_found = all(
                term in title_lower or term in content_lower
                for term in terms
            )

            if not all_terms_found:
                continue

            # Calculate score based on term occurrences
            score = 0
            best_term_idx = -1

            for term in terms:
                # Title matches worth more
                if term in title_lower:
                    score += 10

                # Count occurrences in content
                term_count = content_lower.count(term)
                score += term_count

                # Track position of first term for snippet
                idx = content_lower.find(term)
                if idx >= 0 and (best_term_idx < 0 or idx < best_term_idx):
                    best_term_idx = idx

            frontmatter, body = parse_frontmatter(content)
            body_lower = body.lower()

            # Extract snippet around first found term
            snippet_idx = -1
            for term in terms:
                idx = body_lower.find(term)
                if idx >= 0:
                    snippet_idx = idx
                    break

            if snippet_idx >= 0:
                start = max(0, snippet_idx - 50)
                end = min(len(body), snippet_idx + 150)
                snippet = "..." + body[start:end].replace("\n", " ") + "..."
            else:
                snippet = body[:200].replace("\n", " ") + "..."

            results.append({
                "title": title,
                "path": str(rel_path),
                "score": score,
                "snippet": snippet,
                "tags": frontmatter.get("tags", []),
                "type": frontmatter.get("type", "note"),
                "date": frontmatter.get("date", ""),
                "matched_terms": terms,
            })
        except Exception:
            continue

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_note_content(note_path: str) -> dict | None:
    """Get full content of a specific note."""
    full_path = VAULT_PATH / note_path

    if not full_path.exists():
        # Try to find by title
        for note_file in VAULT_PATH.rglob("*.md"):
            if note_path.lower() in note_file.stem.lower():
                full_path = note_file
                break

    if not full_path.exists():
        return None

    content = full_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Extract links
    links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)

    return {
        "title": full_path.stem,
        "path": str(full_path.relative_to(VAULT_PATH)),
        "frontmatter": frontmatter,
        "body": body,
        "links": list(set(links)),
        "word_count": len(body.split()),
    }


def get_related_notes(concept: str, max_results: int = 10) -> list[dict]:
    """Find notes related to a concept (by links, tags, or content)."""
    results = []
    concept_lower = concept.lower()

    for note_file in VAULT_PATH.rglob("*.md"):
        rel_path = note_file.relative_to(VAULT_PATH)
        if any(part.startswith("_Template") or part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = note_file.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)

            score = 0
            reasons = []

            # Check links
            links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
            for link in links:
                if concept_lower in link.lower():
                    score += 5
                    reasons.append(f"links to [[{link}]]")

            # Check tags
            tags = frontmatter.get("tags", [])
            for tag in tags:
                if concept_lower in str(tag).lower():
                    score += 3
                    reasons.append(f"tagged #{tag}")

            # Check title
            if concept_lower in note_file.stem.lower():
                score += 10
                reasons.append("title match")

            # Check content
            if concept_lower in body.lower():
                mentions = body.lower().count(concept_lower)
                score += mentions
                reasons.append(f"mentioned {mentions}x")

            if score > 0:
                results.append({
                    "title": note_file.stem,
                    "path": str(rel_path),
                    "score": score,
                    "reasons": reasons[:3],
                    "type": frontmatter.get("type", "note"),
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


def get_vault_stats() -> dict:
    """Get statistics about the vault."""
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

    for note_file in VAULT_PATH.rglob("*.md"):
        rel_path = note_file.relative_to(VAULT_PATH)
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = note_file.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)

            stats["total_notes"] += 1
            stats["total_words"] += len(body.split())

            # Count links
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            stats["total_links"] += len(links)

            # By type
            note_type = frontmatter.get("type", "unknown")
            stats["by_type"][note_type] = stats["by_type"].get(note_type, 0) + 1

            # By folder
            folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            stats["by_folder"][folder] = stats["by_folder"].get(folder, 0) + 1

            # Tags
            for tag in frontmatter.get("tags", []):
                tag_str = str(tag)
                stats["tags"][tag_str] = stats["tags"].get(tag_str, 0) + 1

            # Track for recent
            mtime = note_file.stat().st_mtime
            notes_with_dates.append((note_file.stem, str(rel_path), mtime))

        except Exception:
            continue

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
    """Find all notes with a specific tag."""
    results = []
    tag_lower = tag.lower().replace("#", "")

    for note_file in VAULT_PATH.rglob("*.md"):
        rel_path = note_file.relative_to(VAULT_PATH)
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = note_file.read_text(encoding="utf-8")
            frontmatter, _ = parse_frontmatter(content)

            tags = [str(t).lower() for t in frontmatter.get("tags", [])]

            if any(tag_lower in t for t in tags):
                results.append({
                    "title": note_file.stem,
                    "path": str(rel_path),
                    "tags": frontmatter.get("tags", []),
                    "type": frontmatter.get("type", "note"),
                })
        except Exception:
            continue

    return results


def get_backlinks(note_title: str) -> list[dict]:
    """Find all notes that link to a specific note."""
    results = []
    title_lower = note_title.lower()

    for note_file in VAULT_PATH.rglob("*.md"):
        rel_path = note_file.relative_to(VAULT_PATH)
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        try:
            content = note_file.read_text(encoding="utf-8")

            # Find links to the target note
            links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)

            for link in links:
                if title_lower in link.lower():
                    frontmatter, _ = parse_frontmatter(content)
                    results.append({
                        "title": note_file.stem,
                        "path": str(rel_path),
                        "link_text": link,
                        "type": frontmatter.get("type", "note"),
                    })
                    break
        except Exception:
            continue

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
        output += f"**Tags:** {', '.join(note['frontmatter'].get('tags', []))}\n"
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
