"""
Pytest configuration and fixtures for knowledge-assistant tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_vault(tmp_path: Path):
    """Create a temporary vault with test notes."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    # Create folder structure
    (vault_path / "Concepts").mkdir()
    (vault_path / "Sessions").mkdir()
    (vault_path / "References").mkdir()
    (vault_path / "_Templates").mkdir()

    # Note 1: Concept with full frontmatter
    (vault_path / "Concepts" / "C_Python.md").write_text("""---
title: Python
date: 2024-01-15
type: concept
status: evergreen
tags:
  - programming
  - language
related:
  - JavaScript
---

# Python

Python is a programming language.

It has many features like:
- Dynamic typing
- Indentation-based syntax
- Rich standard library

See also [[JavaScript]] for comparison.
""", encoding="utf-8")

    # Note 2: Another concept with links
    (vault_path / "Concepts" / "C_JavaScript.md").write_text("""---
title: JavaScript
date: 2024-01-16
type: concept
status: seedling
tags:
  - programming
  - web
related:
  - Python
---

# JavaScript

JavaScript is a web programming language.

It links to [[Python]] and [[Docker]].
""", encoding="utf-8")

    # Note 3: Session note with date prefix
    (vault_path / "Sessions" / "2024-01-20_Session_DevSetup.md").write_text("""---
title: Dev Setup Session
date: 2024-01-20
type: session
status: complete
tags:
  - devops
  - setup
related: []
---

# Development Setup

Today we configured Python and Docker for development.
The setup includes [[Python]] configuration.
""", encoding="utf-8")

    # Note 4: Reference note
    (vault_path / "References" / "R_Docker.md").write_text("""---
title: Docker Reference
date: 2024-01-10
type: reference
status: evergreen
tags:
  - devops
  - containers
related:
  - Kubernetes
---

# Docker

Docker is a containerization platform.

Related: [[Python]], [[Kubernetes]]
""", encoding="utf-8")

    # Note 5: Template (should be filtered out by default)
    (vault_path / "_Templates" / "T_Concept.md").write_text("""---
title: Concept Template
type: template
tags:
  - template
---

# {{title}}

Content here.
""", encoding="utf-8")

    # Note 6: Note without frontmatter
    (vault_path / "no_frontmatter.md").write_text("""# Simple Note

This note has no YAML frontmatter.
Just plain markdown content.
""", encoding="utf-8")

    # Note 7: Note with invalid frontmatter
    (vault_path / "invalid_frontmatter.md").write_text("""---
title: [invalid yaml
date: not-a-date
---

This note has invalid YAML frontmatter.
""", encoding="utf-8")

    # Note 8: Note with aliased link
    (vault_path / "Concepts" / "C_Aliases.md").write_text("""---
title: Aliases Test
date: 2024-01-25
type: concept
status: seedling
tags:
  - testing
related: []
---

# Aliases

This note uses [[Python|the Python language]] with an alias.
""", encoding="utf-8")

    yield vault_path


@pytest.fixture
def vault_cache(temp_vault):
    """Create a VaultCache instance with the temp vault."""
    from src.server import VaultCache
    cache = VaultCache(temp_vault, ttl=60)
    cache.refresh(force=True)
    return cache


@pytest.fixture
def patched_vault_cache(temp_vault, monkeypatch):
    """Patch the global vault_cache to use the temp vault."""
    from src import server
    from src.server import VaultCache

    cache = VaultCache(temp_vault, ttl=60)
    cache.refresh(force=True)
    monkeypatch.setattr(server, "vault_cache", cache)
    monkeypatch.setattr(server, "VAULT_PATH", temp_vault)
    return cache
