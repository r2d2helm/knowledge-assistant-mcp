"""
Graph functions for Knowledge Assistant MCP Server.

Contains functions for building and querying the knowledge graph.
"""

from .cache import vault_cache
from .models import CachedNote, GraphEdge, GraphNode


def _find_note_by_link(link: str, stem_to_note: dict[str, CachedNote]) -> CachedNote | None:
    """Find a note by link text, handling partial matches.

    Link text may be:
    - Exact stem match: "C_Python" -> C_Python
    - Partial match: "Python" -> C_Python (if stem contains "python")
    """
    link_lower = link.lower()

    # Exact match first
    if link_lower in stem_to_note:
        return stem_to_note[link_lower]

    # Partial match: link text is contained in stem
    for stem, note in stem_to_note.items():
        if link_lower in stem:
            return note

    return None


async def build_graph() -> dict:
    """Build a complete graph of all notes and their links.

    Returns:
        dict with:
        - nodes: list of GraphNode as dicts
        - edges: list of GraphEdge as dicts
        - orphans: list of GraphNode as dicts (notes with no links)
        - stats: global graph statistics
    """
    notes = await vault_cache.get_notes()

    # Build a mapping of note stems (lowercase) to note data
    stem_to_note: dict[str, CachedNote] = {}
    for note in notes:
        stem_to_note[note.stem_lower] = note

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    orphans: list[GraphNode] = []

    # Track connections per note
    connection_counts: dict[str, int] = {}

    for note in notes:
        note_id = note.stem

        # Count outgoing links
        outgoing = 0
        for link in note.links:
            # Find the target note (handles partial matches)
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note:
                edges.append(GraphEdge(
                    source=note_id,
                    target=target_note.stem,
                ))
                outgoing += 1
                # Increment target's incoming count
                target_stem = target_note.stem
                connection_counts[target_stem] = connection_counts.get(target_stem, 0) + 1

        connection_counts[note_id] = connection_counts.get(note_id, 0) + outgoing

    # Build nodes with connection counts
    for note in notes:
        note_id = note.stem
        conn_count = connection_counts.get(note_id, 0)

        node = GraphNode(
            id=note_id,
            title=note.stem,
            path=note.rel_path_str,
            type=note.type,
            connections=conn_count,
        )
        nodes.append(node)

        # Orphan = no outgoing links AND no incoming links
        if conn_count == 0:
            orphans.append(GraphNode(
                id=note_id,
                title=note.stem,
                path=note.rel_path_str,
                type=note.type,
                connections=0,
            ))

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "orphan_count": len(orphans),
        "avg_connections": round(sum(connection_counts.values()) / len(nodes), 2) if nodes else 0,
    }

    return {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
        "orphans": [o.model_dump() for o in orphans],
        "stats": stats,
    }


async def get_subgraph(center_note: str, depth: int = 2) -> dict:
    """Get a subgraph centered on a specific note.

    Args:
        center_note: Title or path of the center note
        depth: How many levels of connections to include (default: 2)

    Returns:
        dict with nodes[], edges[], center, and stats
    """
    notes = await vault_cache.get_notes()

    # Build stem mappings
    stem_to_note: dict[str, CachedNote] = {}
    for note in notes:
        stem_to_note[note.stem_lower] = note

    # Find center note
    center_lower = center_note.lower()
    center_data: CachedNote | None = None

    # Try exact match first
    if center_lower in stem_to_note:
        center_data = stem_to_note[center_lower]
    else:
        # Try partial match
        for stem, note in stem_to_note.items():
            if center_lower in stem:
                center_data = note
                break

    if not center_data:
        return {"error": f"Note not found: {center_note}"}

    center_id = center_data.stem

    # BFS to find all nodes within depth
    visited: set[str] = {center_id}
    frontier: set[str] = {center_id}
    subgraph_notes: dict[str, CachedNote] = {center_id: center_data}

    # Build reverse links (backlinks) using the same matching logic
    backlinks: dict[str, list[str]] = {}
    for note in notes:
        for link in note.links:
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note:
                target_stem = target_note.stem
                if target_stem not in backlinks:
                    backlinks[target_stem] = []
                backlinks[target_stem].append(note.stem)

    for _ in range(depth):
        new_frontier: set[str] = set()

        for node_id in frontier:
            node_data = stem_to_note.get(node_id.lower())
            if not node_data:
                continue

            # Outgoing links
            for link in node_data.links:
                target_note = _find_note_by_link(link, stem_to_note)
                if target_note:
                    target_stem = target_note.stem
                    if target_stem not in visited:
                        visited.add(target_stem)
                        new_frontier.add(target_stem)
                        subgraph_notes[target_stem] = target_note

            # Incoming links (backlinks)
            if node_id in backlinks:
                for source_id in backlinks[node_id]:
                    if source_id not in visited:
                        visited.add(source_id)
                        new_frontier.add(source_id)
                        subgraph_notes[source_id] = stem_to_note[source_id.lower()]

        frontier = new_frontier

    # Build edges for the subgraph
    edges: list[GraphEdge] = []
    for node_id, node_data in subgraph_notes.items():
        for link in node_data.links:
            target_note = _find_note_by_link(link, stem_to_note)
            if target_note and target_note.stem in subgraph_notes:
                edges.append(GraphEdge(
                    source=node_id,
                    target=target_note.stem,
                ))

    # Build nodes list with connection counts within subgraph
    edge_dicts = [e.model_dump() for e in edges]
    nodes: list[GraphNode] = []
    for node_id, node_data in subgraph_notes.items():
        conn_count = sum(1 for e in edge_dicts if e["source"] == node_id or e["target"] == node_id)
        nodes.append(GraphNode(
            id=node_id,
            title=node_data.stem,
            path=node_data.rel_path_str,
            type=node_data.type,
            connections=conn_count,
            is_center=node_id == center_id,
        ))

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "depth": depth,
        "center": center_id,
    }

    return {
        "nodes": [n.model_dump() for n in nodes],
        "edges": edge_dicts,
        "center": center_id,
        "stats": stats,
    }


async def get_clusters(top_n: int = 10) -> dict:
    """Get the most connected clusters in the graph.

    Returns the top N most connected notes and their immediate neighbors.

    Args:
        top_n: Number of top connected notes to return (default: 10)

    Returns:
        dict with clusters[], stats
    """
    graph = await build_graph()

    # Sort nodes by connections (nodes are already dicts from model_dump())
    sorted_nodes = sorted(graph["nodes"], key=lambda x: x["connections"], reverse=True)
    top_nodes = sorted_nodes[:top_n]

    # Build clusters for each top node
    clusters: list[dict] = []
    for node in top_nodes:
        # Find immediate neighbors (connected nodes)
        neighbors: set[str] = set()
        for edge in graph["edges"]:
            if edge["source"] == node["id"]:
                neighbors.add(edge["target"])
            elif edge["target"] == node["id"]:
                neighbors.add(edge["source"])

        clusters.append({
            "hub": {
                "id": node["id"],
                "title": node["title"],
                "path": node["path"],
                "type": node["type"],
                "connections": node["connections"],
            },
            "neighbors": list(neighbors),
            "neighbor_count": len(neighbors),
        })

    stats = {
        "total_clusters": len(clusters),
        "total_nodes": graph["stats"]["total_nodes"],
        "total_edges": graph["stats"]["total_edges"],
        "orphan_count": graph["stats"]["orphan_count"],
    }

    return {
        "clusters": clusters,
        "orphans": graph["orphans"],
        "stats": stats,
    }
