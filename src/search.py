"""
Search and query functions for Knowledge Assistant MCP Server.

Contains functions for searching notes, finding related notes, and exploring tags.
"""

import structlog

from .cache import vault_cache
from .config import settings
from .models import RelatedResult, SearchResult
from .utils import (
    ALL_LINKS_PATTERN,
    SEARCH_SPLIT_PATTERN,
    PathValidationError,
    validate_path_within_vault,
)

logger = structlog.get_logger(__name__)


async def search_notes(query: str, max_results: int = 10) -> list[SearchResult]:
    """Search notes by content or title using cached data.

    Supports multi-word queries: all words must be present (AND logic).
    """
    results: list[SearchResult] = []

    terms = [t.strip().lower() for t in SEARCH_SPLIT_PATTERN.split(query) if t.strip()]
    if not terms:
        return []

    for note in await vault_cache.get_notes():
        try:
            # Check if ALL terms are present (AND logic)
            all_terms_found = all(
                term in note.stem_lower or term in note.content_lower
                for term in terms
            )

            if not all_terms_found:
                continue

            # Calculate score
            score = 0
            for term in terms:
                if term in note.stem_lower:
                    score += 10
                score += note.content_lower.count(term)

            # Extract snippet
            snippet_idx = -1
            for term in terms:
                idx = note.body_lower.find(term)
                if idx >= 0:
                    snippet_idx = idx
                    break

            if snippet_idx >= 0:
                start = max(0, snippet_idx - 50)
                end = min(len(note.body), snippet_idx + 150)
                snippet = "..." + note.body[start:end].replace("\n", " ") + "..."
            else:
                snippet = note.body[:200].replace("\n", " ") + "..."

            results.append(SearchResult(
                title=note.stem,
                path=note.rel_path_str,
                score=float(score),
                snippet=snippet,
                tags=note.tags,
                type=note.type,
                date=note.date,
                matched_terms=terms,
            ))
        except Exception as e:
            logger.warning("note_score_failed", path=note.rel_path_str, error=str(e))
            continue

    results.sort(key=lambda x: x.score, reverse=True)
    final_results = results[:max_results]
    logger.debug("search_completed", query=query, results=len(final_results))
    return final_results


async def get_note_content(note_path: str) -> dict | None:
    """Get full content of a specific note from cache.

    Security: Validates path before lookup to prevent directory traversal.

    Note: Returns a dict for backward compatibility with tools.py formatting.
    """
    # Early validation - reject obvious traversal attempts
    if not note_path or ".." in note_path:
        return None

    # Additional validation for explicit path lookups
    if "/" in note_path or "\\" in note_path:
        try:
            validate_path_within_vault(note_path, settings.vault_path)
        except PathValidationError:
            return None

    note = await vault_cache.get_note_by_path(note_path)
    if not note:
        return None

    return {
        "title": note.stem,
        "path": note.rel_path_str,
        "frontmatter": note.frontmatter,
        "body": note.body,
        "links": note.links,
        "word_count": note.word_count,
    }


async def get_related_notes(concept: str, max_results: int = 10) -> list[RelatedResult]:
    """Find notes related to a concept using cached data."""
    results: list[RelatedResult] = []
    concept_lower = concept.lower()

    for note in await vault_cache.get_notes():
        try:
            score = 0
            reasons: list[str] = []

            # Check links
            for link in note.links:
                if concept_lower in link.lower():
                    score += 5
                    reasons.append(f"links to [[{link}]]")

            # Check tags
            for tag in note.tags:
                if concept_lower in tag.lower():
                    score += 3
                    reasons.append(f"tagged #{tag}")

            # Check title
            if concept_lower in note.stem_lower:
                score += 10
                reasons.append("title match")

            # Check content
            if concept_lower in note.body_lower:
                mentions = note.body_lower.count(concept_lower)
                score += mentions
                reasons.append(f"mentioned {mentions}x")

            if score > 0:
                results.append(RelatedResult(
                    title=note.stem,
                    path=note.rel_path_str,
                    score=float(score),
                    reasons=reasons[:3],
                    type=note.type,
                ))
        except Exception as e:
            logger.warning("relevance_compute_failed", path=note.rel_path_str, error=str(e))
            continue

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_results]


async def get_vault_stats() -> dict:
    """Get statistics about the vault using cached data."""
    stats: dict = {
        "total_notes": 0,
        "by_type": {},
        "by_folder": {},
        "total_words": 0,
        "total_links": 0,
        "tags": {},
        "recent_notes": [],
    }

    notes_with_dates: list[tuple[str, str, float]] = []

    for note in await vault_cache.get_notes(include_templates=False):
        # Skip hidden
        if any(part.startswith(".") for part in note.rel_path.parts):
            continue

        stats["total_notes"] += 1
        stats["total_words"] += note.word_count

        # Count links (including aliased)
        all_links = ALL_LINKS_PATTERN.findall(note.content)
        stats["total_links"] += len(all_links)

        # By type
        note_type = note.type
        stats["by_type"][note_type] = stats["by_type"].get(note_type, 0) + 1

        # By folder
        folder = note.rel_path.parts[0] if len(note.rel_path.parts) > 1 else "root"
        stats["by_folder"][folder] = stats["by_folder"].get(folder, 0) + 1

        # Tags
        for tag in note.tags:
            stats["tags"][tag] = stats["tags"].get(tag, 0) + 1

        notes_with_dates.append((note.stem, note.rel_path_str, note.mtime))

    # Recent notes
    notes_with_dates.sort(key=lambda x: x[2], reverse=True)
    stats["recent_notes"] = [
        {"title": n[0], "path": n[1]}
        for n in notes_with_dates[:10]
    ]

    # Sort tags by count
    stats["top_tags"] = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)[:20]

    return stats


async def explore_by_tag(tag: str) -> list[dict]:
    """Find all notes with a specific tag using cached data.

    Note: Returns dicts for backward compatibility with tools.py formatting.
    """
    results: list[dict] = []
    tag_lower = tag.lower().replace("#", "")

    for note in await vault_cache.get_notes():
        tags_lower = [t.lower() for t in note.tags]
        if any(tag_lower in t for t in tags_lower):
            results.append({
                "title": note.stem,
                "path": note.rel_path_str,
                "tags": note.tags,
                "type": note.type,
            })

    return results


async def get_backlinks(note_title: str) -> list[dict]:
    """Find all notes that link to a specific note using cached data.

    Note: Returns dicts for backward compatibility with tools.py formatting.
    """
    results: list[dict] = []
    title_lower = note_title.lower()

    for note in await vault_cache.get_notes():
        for link in note.links:
            if title_lower in link.lower():
                results.append({
                    "title": note.stem,
                    "path": note.rel_path_str,
                    "link_text": link,
                    "type": note.type,
                })
                break

    return results
