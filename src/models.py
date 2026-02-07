"""
Pydantic models for Knowledge Assistant MCP Server.

Contains data models for cached notes, search results, graph elements, and write results.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class CachedNote(BaseModel):
    """Model for a cached note from the vault."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    rel_path: Path
    rel_path_str: str
    stem: str
    stem_lower: str
    content: str
    content_lower: str
    frontmatter: dict[str, Any]
    body: str
    body_lower: str
    links: list[str]
    tags: list[str]
    type: str
    date: str
    mtime: float
    word_count: int
    is_template: bool


class SearchResult(BaseModel):
    """Model for a search result."""

    title: str
    path: str
    score: float
    snippet: str
    tags: list[str]
    type: str
    date: str
    matched_terms: list[str]


class RelatedResult(BaseModel):
    """Model for a related note result."""

    title: str
    path: str
    score: float
    reasons: list[str]
    type: str


class GraphNode(BaseModel):
    """Model for a node in the knowledge graph."""

    id: str
    title: str
    path: str
    type: str
    connections: int
    is_center: bool = False


class GraphEdge(BaseModel):
    """Model for an edge in the knowledge graph."""

    source: str
    target: str


class WriteResult(BaseModel):
    """Model for the result of a write operation."""

    success: bool
    path: str = ""
    filename: str = ""
    folder: str = ""
    error: str = ""
    warnings: dict[str, Any] | None = None
