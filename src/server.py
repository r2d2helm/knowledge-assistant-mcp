"""
Knowledge Assistant MCP Server

Backward-compatible module that re-exports all public APIs from the modular structure.
This allows existing code (tests, imports) to continue working with `from src.server import ...`.

The actual implementation is split across:
- config.py: Configuration constants
- utils.py: Utility functions and regex patterns
- cache.py: VaultCache class
- search.py: Search and query functions
- writer.py: Note writing functions
- graph.py: Knowledge graph functions
- tools.py: MCP tool handlers
- main.py: Entry point
"""

# Re-export configuration
from .config import (
    CACHE_TTL,
    INDEX_PATH,
    MAX_CONTENT_SIZE,
    MAX_TITLE_LENGTH,
    NOTE_TYPE_CONFIG,
    Settings,
    VAULT_PATH,
    _get_default_index_path,
    _get_default_vault_path,
    settings,
)

# Re-export utility functions and patterns
from .utils import (
    ALL_LINKS_PATTERN,
    ContentValidationError,
    FRONTMATTER_PATTERN,
    PathValidationError,
    SEARCH_SPLIT_PATTERN,
    TITLE_PATTERN,
    TitleValidationError,
    UNSAFE_CHARS_PATTERN,
    WHITESPACE_PATTERN,
    WIKILINK_PATTERN,
    WORD_SPLIT_PATTERN,
    load_index,
    parse_frontmatter,
    validate_content_size,
    validate_folder_path,
    validate_path_within_vault,
    validate_title,
)

# Re-export cache
from .cache import VaultCache, vault_cache

# Re-export models
from .models import (
    CachedNote,
    GraphEdge,
    GraphNode,
    RelatedResult,
    SearchResult,
    WriteResult,
)

# Re-export search functions
from .search import (
    explore_by_tag,
    get_backlinks,
    get_note_content,
    get_related_notes,
    get_vault_stats,
    search_notes,
)

# Re-export writer functions
from .writer import (
    check_duplicates,
    generate_filename,
    generate_frontmatter,
    get_default_folder,
    write_note,
)

# Re-export graph functions
from .graph import (
    _find_note_by_link,
    build_graph,
    get_clusters,
    get_subgraph,
)

# Re-export MCP components
from .tools import (
    call_tool,
    list_resources,
    list_tools,
    read_resource,
    server,
)

# Re-export main entry point
from .main import main

# Re-export logging configuration
from .logging import configure_logging, get_logger

# Ensure all public symbols are available
__all__ = [
    # Config
    "Settings",
    "settings",
    "VAULT_PATH",
    "INDEX_PATH",
    "CACHE_TTL",
    "MAX_CONTENT_SIZE",
    "MAX_TITLE_LENGTH",
    "NOTE_TYPE_CONFIG",
    "_get_default_vault_path",
    "_get_default_index_path",
    # Utils
    "WIKILINK_PATTERN",
    "FRONTMATTER_PATTERN",
    "SEARCH_SPLIT_PATTERN",
    "ALL_LINKS_PATTERN",
    "UNSAFE_CHARS_PATTERN",
    "WHITESPACE_PATTERN",
    "WORD_SPLIT_PATTERN",
    "TITLE_PATTERN",
    "PathValidationError",
    "TitleValidationError",
    "ContentValidationError",
    "load_index",
    "parse_frontmatter",
    "validate_path_within_vault",
    "validate_folder_path",
    "validate_title",
    "validate_content_size",
    # Cache
    "VaultCache",
    "vault_cache",
    # Models
    "CachedNote",
    "SearchResult",
    "RelatedResult",
    "GraphNode",
    "GraphEdge",
    "WriteResult",
    # Search
    "search_notes",
    "get_note_content",
    "get_related_notes",
    "get_vault_stats",
    "explore_by_tag",
    "get_backlinks",
    # Writer
    "generate_filename",
    "get_default_folder",
    "generate_frontmatter",
    "check_duplicates",
    "write_note",
    # Graph
    "_find_note_by_link",
    "build_graph",
    "get_subgraph",
    "get_clusters",
    # MCP
    "server",
    "list_tools",
    "call_tool",
    "list_resources",
    "read_resource",
    # Main
    "main",
    # Logging
    "configure_logging",
    "get_logger",
]

# Entry point for `python -m src.server`
if __name__ == "__main__":
    main()
