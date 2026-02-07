"""
Configuration module for Knowledge Assistant MCP Server.

Uses pydantic-settings for configuration management with environment variable support.
Environment variables use KNOWLEDGE_ prefix (e.g., KNOWLEDGE_VAULT_PATH).
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Environment variables:
    - KNOWLEDGE_VAULT_PATH: Path to the Obsidian vault
    - KNOWLEDGE_INDEX_PATH: Path to the notes index file
    - KNOWLEDGE_CACHE_TTL: Cache TTL in seconds
    - KNOWLEDGE_MAX_SEARCH_RESULTS: Maximum search results
    - KNOWLEDGE_MAX_CONTENT_SIZE: Maximum content size in bytes
    - KNOWLEDGE_MAX_TITLE_LENGTH: Maximum title length
    """

    vault_path: Path = Field(default_factory=_get_default_vault_path)
    index_path: Path = Field(default_factory=_get_default_index_path)
    cache_ttl: int = 60
    max_search_results: int = 50
    max_content_size: int = 1 * 1024 * 1024  # 1MB in bytes
    max_title_length: int = 200

    model_config = SettingsConfigDict(env_prefix="KNOWLEDGE_")


# Global settings instance
settings = Settings()

# Backward-compatible aliases for existing code
VAULT_PATH = settings.vault_path
INDEX_PATH = settings.index_path
CACHE_TTL = settings.cache_ttl
MAX_CONTENT_SIZE = settings.max_content_size
MAX_TITLE_LENGTH = settings.max_title_length

# Mapping of note types to naming prefixes and default folders
NOTE_TYPE_CONFIG = {
    "concept": {"prefix": "C_", "folder": "Concepts"},
    "conversation": {"prefix": "{date}_Conv_", "folder": "Conversations"},
    "troubleshooting": {"prefix": "{date}_Fix_", "folder": "Troubleshooting"},
    "session": {"prefix": "{date}_Session_", "folder": "Sessions"},
    "reference": {"prefix": "R_", "folder": "References"},
    "project": {"prefix": "P_", "folder": "Projects"},
}
