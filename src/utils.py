"""
Utility functions and compiled regex patterns for Knowledge Assistant MCP Server.

Contains parsing functions, validation utilities, and pre-compiled patterns.
"""

import json
import re
from pathlib import Path

import yaml

from .config import settings

# Pre-compiled regex patterns for performance
WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
SEARCH_SPLIT_PATTERN = re.compile(r'[\s\-_]+')
ALL_LINKS_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
UNSAFE_CHARS_PATTERN = re.compile(r'[^\w\s-]')
WHITESPACE_PATTERN = re.compile(r'[\s]+')
WORD_SPLIT_PATTERN = re.compile(r'[\s_-]+')

# Allowed characters in title (alphanumeric, spaces, hyphens, underscores, accented chars)
TITLE_PATTERN = re.compile(r'^[\w\s\-àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]+$', re.UNICODE)


# ============== Exceptions ==============

class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass


class TitleValidationError(Exception):
    """Raised when title validation fails."""
    pass


class ContentValidationError(Exception):
    """Raised when content validation fails."""
    pass


# ============== Helper Functions ==============

def load_index() -> dict:
    """Load the notes index."""
    if settings.index_path.exists():
        with open(settings.index_path, "r", encoding="utf-8") as f:
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


# ============== Security Validation ==============

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

    if len(title) > settings.max_title_length:
        raise TitleValidationError(f"Title exceeds maximum length of {settings.max_title_length} characters")

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

    if content_bytes > settings.max_content_size:
        max_mb = settings.max_content_size / (1024 * 1024)
        actual_mb = content_bytes / (1024 * 1024)
        raise ContentValidationError(
            f"Content size ({actual_mb:.2f}MB) exceeds maximum allowed size ({max_mb}MB)"
        )

    return content
