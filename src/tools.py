"""
MCP Tools module for Knowledge Assistant MCP Server.

Contains the MCP tool handlers (list_tools and call_tool).
"""

import json
from typing import Any

from mcp.server import Server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)

from .config import NOTE_TYPE_CONFIG
from .graph import build_graph, get_clusters, get_subgraph
from .search import (
    explore_by_tag,
    get_backlinks,
    get_note_content,
    get_related_notes,
    get_vault_stats,
    search_notes,
)
from .writer import write_note

# Initialize server
server = Server("knowledge-assistant")


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
        results = await search_notes(query, max_results)

        if not results:
            return [TextContent(type="text", text=f"No notes found for query: '{query}'")]

        output = f"Found {len(results)} notes for '{query}':\n\n"
        for r in results:
            output += f"**{r.title}** ({r.path})\n"
            tags_str = ', '.join(str(t) for t in r.tags[:3]) if r.tags else 'none'
            output += f"  Type: {r.type} | Tags: {tags_str}\n"
            output += f"  {r.snippet}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_read":
        note_path = arguments.get("note_path", "")
        note = await get_note_content(note_path)

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
        results = await get_related_notes(concept, max_results)

        if not results:
            return [TextContent(type="text", text=f"No notes related to: '{concept}'")]

        output = f"Found {len(results)} notes related to '{concept}':\n\n"
        for r in results:
            output += f"**{r.title}** ({r.path})\n"
            output += f"  Reasons: {', '.join(r.reasons)}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_stats":
        stats = await get_vault_stats()

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
        results = await explore_by_tag(tag)

        if not results:
            return [TextContent(type="text", text=f"No notes found with tag: '{tag}'")]

        output = f"Found {len(results)} notes with tag '#{tag}':\n\n"
        for r in results:
            output += f"- **{r['title']}** ({r['type']})\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_backlinks":
        note_title = arguments.get("note_title", "")
        results = await get_backlinks(note_title)

        if not results:
            return [TextContent(type="text", text=f"No backlinks found for: '{note_title}'")]

        output = f"Found {len(results)} notes linking to '{note_title}':\n\n"
        for r in results:
            output += f"- **{r['title']}** ({r['type']}) - links via [[{r['link_text']}]]\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_recent":
        count = arguments.get("count", 10)
        stats = await get_vault_stats()
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

        result = await write_note(title, content, note_type, tags, folder, related)

        if not result.success:
            return [TextContent(type="text", text=f"Error: {result.error}")]

        output = f"# Note Created Successfully\n\n"
        output += f"**Path:** {result.path}\n"
        output += f"**Filename:** {result.filename}\n"
        output += f"**Folder:** {result.folder}\n"

        if result.warnings:
            output += f"\n## Warnings\n"
            output += f"{result.warnings['message']}\n\n"
            for dup in result.warnings['potential_duplicates']:
                output += f"- **{dup['title']}** ({dup['path']}) - {dup['match_type']}\n"

        return [TextContent(type="text", text=output)]

    elif name == "knowledge_graph":
        center_note = arguments.get("center_note")
        depth = arguments.get("depth", 2)
        output_format = arguments.get("format", "json")

        if center_note:
            # Return subgraph around a specific note
            result = await get_subgraph(center_note, depth)

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
            result = await get_clusters()

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
        stats = await get_vault_stats()
        return json.dumps(stats, indent=2)

    return json.dumps({"error": f"Unknown resource: {uri}"})
