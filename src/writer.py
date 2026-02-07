"""
Note writing functions for Knowledge Assistant MCP Server.

Contains functions for creating notes with proper frontmatter and naming conventions.
"""

from datetime import datetime

import aiofiles
import structlog
import yaml

from .cache import vault_cache
from .config import NOTE_TYPE_CONFIG, settings
from .models import WriteResult
from .utils import (
    ContentValidationError,
    PathValidationError,
    TitleValidationError,
    UNSAFE_CHARS_PATTERN,
    WHITESPACE_PATTERN,
    WORD_SPLIT_PATTERN,
    validate_content_size,
    validate_folder_path,
    validate_path_within_vault,
    validate_title,
)

logger = structlog.get_logger(__name__)


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


async def check_duplicates(title: str, note_type: str) -> list[dict]:
    """Check for potential duplicate notes.

    Returns a list of notes that might be duplicates based on:
    - Similar title (case-insensitive)
    - Same type with similar content keywords
    """
    duplicates: list[dict] = []
    title_lower = title.lower()
    title_words = set(WORD_SPLIT_PATTERN.split(title_lower))

    for note in await vault_cache.get_notes():
        # Check for exact title match
        if note.stem_lower == title_lower:
            duplicates.append({
                "title": note.stem,
                "path": note.rel_path_str,
                "type": note.type,
                "match_type": "exact_title",
            })
            continue

        # Check for similar title (word overlap)
        note_words = set(WORD_SPLIT_PATTERN.split(note.stem_lower))
        overlap = title_words & note_words
        if len(overlap) >= 2 and len(overlap) / len(title_words) > 0.5:
            duplicates.append({
                "title": note.stem,
                "path": note.rel_path_str,
                "type": note.type,
                "match_type": "similar_title",
                "common_words": list(overlap),
            })

    return duplicates[:5]  # Return at most 5 potential duplicates


async def write_note(title: str, content: str, note_type: str, tags: list[str], folder: str | None = None, related: list[str] | None = None) -> WriteResult:
    """Create a new note in the vault.

    Args:
        title: Note title
        content: Note body content (markdown)
        note_type: Type of note (concept, conversation, troubleshooting, etc.)
        tags: List of tags
        folder: Optional folder path (defaults based on type)
        related: Optional list of related note titles

    Returns:
        WriteResult with status, path, and any warnings
    """
    # === Security validations ===

    # Validate title
    try:
        title = validate_title(title)
    except TitleValidationError as e:
        return WriteResult(
            success=False,
            error=f"Invalid title: {str(e)}",
        )

    # Validate content size
    try:
        validate_content_size(content)
    except ContentValidationError as e:
        return WriteResult(
            success=False,
            error=f"Invalid content: {str(e)}",
        )

    # Determine and validate folder path
    target_folder = folder if folder else get_default_folder(note_type)

    try:
        folder_path = validate_folder_path(target_folder, settings.vault_path)
    except PathValidationError as e:
        return WriteResult(
            success=False,
            error=f"Invalid folder path: {str(e)}",
        )

    # Check for potential duplicates first
    duplicates = await check_duplicates(title, note_type)

    # Create folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = generate_filename(title, note_type)
    file_path = folder_path / filename

    # Final validation: ensure the complete file path is within vault
    try:
        validated_file_path = validate_path_within_vault(
            str(file_path.relative_to(settings.vault_path.resolve())), settings.vault_path
        )
    except (PathValidationError, ValueError) as e:
        return WriteResult(
            success=False,
            error=f"Invalid file path: {str(e)}",
        )

    # Check if file already exists
    if file_path.exists():
        return WriteResult(
            success=False,
            error=f"File already exists: {file_path.relative_to(settings.vault_path)}",
            path=str(file_path.relative_to(settings.vault_path)),
        )

    # Generate frontmatter and full content
    frontmatter = generate_frontmatter(title, note_type, tags, related)
    full_content = frontmatter + content

    # Write the file asynchronously
    try:
        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(full_content)
        logger.info("note_created", path=str(file_path.relative_to(settings.vault_path)), note_type=note_type)
    except Exception as e:
        logger.error("note_write_failed", path=str(file_path), error=str(e))
        return WriteResult(
            success=False,
            error=f"Failed to write file: {str(e)}",
        )

    # Invalidate cache to include the new note
    await vault_cache.refresh(force=True)

    warnings = None
    if duplicates:
        warnings = {
            "potential_duplicates": duplicates,
            "message": f"Found {len(duplicates)} potential duplicate(s). Review recommended.",
        }

    return WriteResult(
        success=True,
        path=str(file_path.relative_to(settings.vault_path)),
        filename=filename,
        folder=target_folder,
        warnings=warnings,
    )
