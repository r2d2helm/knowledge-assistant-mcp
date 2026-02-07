"""
In-memory cache module for Knowledge Assistant MCP Server.

Contains the VaultCache class for caching vault notes.
"""

import asyncio
import time
from pathlib import Path

import aiofiles
import structlog

from .config import settings
from .models import CachedNote
from .utils import WIKILINK_PATTERN, parse_frontmatter

logger = structlog.get_logger(__name__)


class VaultCache:
    """In-memory cache for vault notes. Avoids repeated filesystem scans.

    Uses incremental refresh: only reloads notes whose mtime has changed,
    removes deleted files, and adds new files.
    """

    def __init__(self, vault_path: Path, ttl: int = 60):
        self.vault_path = vault_path
        self.ttl = ttl
        self._notes: dict[Path, CachedNote] = {}
        self._mtimes: dict[Path, float] = {}
        self._loaded_at: float = 0
        self._note_stems: set[str] = set()  # For quick basename lookup

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._loaded_at) > self.ttl

    async def _load_note(self, note_file: Path) -> CachedNote | None:
        """Load a single note from disk and return a CachedNote or None on error."""
        rel_path = note_file.relative_to(self.vault_path)
        # Skip hidden folders
        if any(part.startswith(".") for part in rel_path.parts):
            return None

        try:
            async with aiofiles.open(note_file, encoding="utf-8") as f:
                content = await f.read()
            frontmatter, body = parse_frontmatter(content)
            stem = note_file.stem
            content_lower = content.lower()

            links = WIKILINK_PATTERN.findall(content)

            # Convert tags to list of strings
            raw_tags = frontmatter.get("tags", [])
            tags = [str(t) for t in raw_tags] if raw_tags else []

            # Convert date to string
            raw_date = frontmatter.get("date", "")
            date_str = str(raw_date) if raw_date else ""

            return CachedNote(
                path=note_file,
                rel_path=rel_path,
                rel_path_str=str(rel_path),
                stem=stem,
                stem_lower=stem.lower(),
                content=content,
                content_lower=content_lower,
                frontmatter=frontmatter,
                body=body,
                body_lower=body.lower(),
                links=list(set(links)),
                tags=tags,
                type=frontmatter.get("type", "unknown"),
                date=date_str,
                mtime=note_file.stat().st_mtime,
                word_count=len(body.split()),
                is_template=any(part.startswith("_Template") for part in rel_path.parts),
            )
        except Exception as e:
            logger.warning("note_read_failed", path=str(note_file), error=str(e))
            return None

    async def _full_reload(self) -> tuple[int, int, int]:
        """Perform a full reload of all notes. Returns (added, updated, removed)."""
        self._notes.clear()
        self._mtimes.clear()
        self._note_stems.clear()

        # Collect all files first (rglob is sync)
        note_files = list(self.vault_path.rglob("*.md"))

        # Load all notes in parallel
        tasks = [self._load_note(note_file) for note_file in note_files]
        results = await asyncio.gather(*tasks)

        added = 0
        for note_file, note in zip(note_files, results):
            if note is not None:
                self._notes[note_file] = note
                self._mtimes[note_file] = note.mtime
                self._note_stems.add(note.stem)
                added += 1

        return added, 0, 0

    async def _incremental_refresh(self) -> tuple[int, int, int]:
        """Perform an incremental refresh. Returns (added, updated, removed)."""
        added = 0
        updated = 0
        removed = 0

        # Scan current files on disk (rglob is sync)
        current_files: set[Path] = set()
        for note_file in self.vault_path.rglob("*.md"):
            rel_path = note_file.relative_to(self.vault_path)
            # Skip hidden folders
            if any(part.startswith(".") for part in rel_path.parts):
                continue
            current_files.add(note_file)

        # Identify files to load (new or modified)
        files_to_load: list[tuple[Path, str]] = []  # (path, reason: 'new' or 'modified')
        old_stems: dict[Path, str | None] = {}

        for note_file in current_files:
            try:
                current_mtime = note_file.stat().st_mtime
            except OSError:
                # File was deleted between rglob and stat
                continue

            cached_mtime = self._mtimes.get(note_file)

            if cached_mtime is None:
                # New file
                files_to_load.append((note_file, "new"))
            elif current_mtime > cached_mtime:
                # Modified file
                old_stems[note_file] = self._notes[note_file].stem if note_file in self._notes else None
                files_to_load.append((note_file, "modified"))

        # Load all new/modified files in parallel
        if files_to_load:
            tasks = [self._load_note(note_file) for note_file, _ in files_to_load]
            results = await asyncio.gather(*tasks)

            for (note_file, reason), note in zip(files_to_load, results):
                if note is not None:
                    if reason == "new":
                        self._notes[note_file] = note
                        self._mtimes[note_file] = note.mtime
                        self._note_stems.add(note.stem)
                        added += 1
                    else:  # modified
                        old_stem = old_stems.get(note_file)
                        if old_stem and old_stem != note.stem:
                            self._note_stems.discard(old_stem)
                        self._notes[note_file] = note
                        self._mtimes[note_file] = note.mtime
                        self._note_stems.add(note.stem)
                        updated += 1

        # Check for deleted files
        cached_files = set(self._notes.keys())
        deleted_files = cached_files - current_files
        for note_file in deleted_files:
            old_note = self._notes.pop(note_file, None)
            self._mtimes.pop(note_file, None)
            if old_note:
                self._note_stems.discard(old_note.stem)
            removed += 1

        return added, updated, removed

    async def refresh(self, force: bool = False) -> None:
        """Reload notes from disk if cache is stale.

        Uses incremental refresh by default: only reloads notes whose mtime
        has changed, removes deleted files, and adds new files.

        Args:
            force: If True, performs a full reload of all notes.
        """
        if not force and not self.is_stale:
            return

        start_time = time.time()

        if force or not self._notes:
            # Full reload if forced or cache is empty
            added, updated, removed = await self._full_reload()
            refresh_type = "full"
        else:
            # Incremental refresh
            added, updated, removed = await self._incremental_refresh()
            refresh_type = "incremental"

        self._loaded_at = time.time()
        duration_ms = round((time.time() - start_time) * 1000, 2)

        reloaded_count = added + updated
        logger.info(
            "cache_refreshed",
            refresh_type=refresh_type,
            note_count=len(self._notes),
            added=added,
            updated=updated,
            removed=removed,
            reloaded=reloaded_count,
            duration_ms=duration_ms,
        )

    async def get_notes(self, include_templates: bool = False) -> list[CachedNote]:
        """Get all cached notes."""
        await self.refresh()
        notes = list(self._notes.values())
        if include_templates:
            return notes
        return [n for n in notes if not n.is_template]

    async def get_note_by_path(self, note_path: str) -> CachedNote | None:
        """Find a note by relative path or title match.

        Security: Validates path to prevent directory traversal attacks.
        """
        await self.refresh()

        # Security validation: reject path traversal attempts
        if not note_path or ".." in note_path:
            return None

        # Reject absolute paths
        if note_path.startswith("/") or (len(note_path) > 1 and note_path[1] == ":"):
            return None

        # Exact path match
        for note in self._notes.values():
            if note.rel_path_str == note_path:
                return note
        # Title/stem match
        path_lower = note_path.lower()
        for note in self._notes.values():
            if path_lower in note.stem_lower:
                return note
        return None

    async def get_note_count(self) -> int:
        """Return the number of cached notes."""
        await self.refresh()
        return len(self._notes)

    async def get_note_stems(self) -> set[str]:
        """Return set of note stems."""
        await self.refresh()
        return self._note_stems


# Global cache instance
vault_cache = VaultCache(settings.vault_path, settings.cache_ttl)
