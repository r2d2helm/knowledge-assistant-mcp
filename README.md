# Knowledge Assistant MCP Server

Serveur MCP (Model Context Protocol) pour interroger un vault Obsidian depuis Claude Code.

## Fonctionnalités

| Outil | Description |
|-------|-------------|
| `knowledge_search` | Recherche multi-termes avec logique AND |
| `knowledge_read` | Lire le contenu complet d'une note |
| `knowledge_related` | Trouver les notes liées à un concept |
| `knowledge_stats` | Statistiques du vault (notes, tags, types) |
| `knowledge_explore_tag` | Lister les notes par tag |
| `knowledge_backlinks` | Trouver les backlinks d'une note |
| `knowledge_recent` | Notes récemment modifiées |

## Installation

### Prérequis

- Python 3.10+
- uv (gestionnaire de packages)

### Configuration Claude Code

#### Linux/macOS

Ajouter dans `~/.claude/settings.json` :

```json
{
  "mcpServers": {
    "knowledge-assistant": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/knowledge-assistant-mcp",
        "python",
        "-m",
        "src.server"
      ],
      "env": {
        "KNOWLEDGE_VAULT_PATH": "/home/username/Documents/Knowledge",
        "KNOWLEDGE_INDEX_PATH": "/home/username/.knowledge/notes-index.json"
      }
    }
  }
}
```

#### Windows

```json
{
  "mcpServers": {
    "knowledge-assistant": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\knowledge-assistant-mcp",
        "python",
        "-m",
        "src.server"
      ],
      "env": {
        "KNOWLEDGE_VAULT_PATH": "C:\\Users\\username\\Documents\\Knowledge",
        "KNOWLEDGE_INDEX_PATH": "C:\\Users\\username\\.knowledge\\notes-index.json"
      }
    }
  }
}
```

## Utilisation

### Recherche multi-termes

```
knowledge_search("PowerShell UTF-8")
```
→ Trouve les notes contenant **tous** les termes (AND)

### Lire une note

```
knowledge_read("Concepts/C_Zettelkasten.md")
```

### Notes liées

```
knowledge_related("PowerShell")
```

### Explorer un tag

```
knowledge_explore_tag("dev/powershell")
```

## Configuration

Les chemins sont configurables via variables d'environnement :

| Variable | Description | Défaut Linux/macOS | Défaut Windows |
|----------|-------------|---------------------|----------------|
| `KNOWLEDGE_VAULT_PATH` | Chemin du vault Obsidian | `~/Documents/Knowledge` | `%USERPROFILE%\Documents\Knowledge` |
| `KNOWLEDGE_INDEX_PATH` | Chemin du fichier d'index | `~/.knowledge/notes-index.json` | `%USERPROFILE%\.knowledge\notes-index.json` |
| `KNOWLEDGE_CACHE_TTL` | Durée du cache en secondes | `60` | `60` |

## Structure du Vault

```
Knowledge/
├── _Inbox/        # Nouvelles captures
├── Concepts/      # Notes atomiques (C_*)
├── Conversations/ # Sessions Claude
├── Projets/       # Notes projet
├── Références/    # Documentation
└── ...
```

## Licence

MIT
