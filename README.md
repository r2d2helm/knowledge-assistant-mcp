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
        "C:\\Users\\r2d2\\.claude\\mcp-servers\\knowledge-assistant",
        "python",
        "-m",
        "src.server"
      ]
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

Le chemin du vault est défini dans `src/server.py` :

```python
VAULT_PATH = Path(r"C:\Users\r2d2\Documents\Knowledge")
```

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
