from __future__ import annotations

import logging
from typing import Any, Dict, List

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


ENTITY_LABEL = "RagEntity"
CHUNK_LABEL = "RagChunk"
RELATION_TYPE = "GRAPH_REL"
MENTION_TYPE = "MENTIONED_IN"
_NEO4J_AVAILABLE: bool | None = None
_WARNED_UNAVAILABLE = False


def is_neo4j_enabled() -> bool:
    return str(getattr(SETTINGS, "graph_store_backend", "")).lower() == "neo4j"


def _get_driver():
    if _NEO4J_AVAILABLE is False:
        raise RuntimeError("Neo4j was marked unavailable in this process")

    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        raise RuntimeError("neo4j Python driver is not installed") from exc

    return GraphDatabase.driver(
        SETTINGS.neo4j_uri,
        auth=(SETTINGS.neo4j_user, SETTINGS.neo4j_password),
        connection_timeout=1.0,
        max_connection_lifetime=30,
        max_transaction_retry_time=0,
    )


def _execute_write(session, query: str, **params):
    return list(session.run(query, **params))


def _execute_read(session, query: str, **params):
    return list(session.run(query, **params))


def write_semantic_graph(file_hash: str, graph_data: Dict[str, Any], chunks: List[dict]) -> None:
    """Persist the semantic graph to Neo4j.

    The JSON graph shape stays as the interchange format in code:
    nodes + edges + entity_chunks. Neo4j stores that shape as entity nodes,
    chunk nodes, GRAPH_REL edges, and MENTIONED_IN edges.
    """
    if not is_neo4j_enabled():
        return

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    entity_chunks = graph_data.get("entity_chunks", {})
    chunk_rows = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "parent_id": c.get("parent_id", ""),
            "text": c.get("text", ""),
            "pos": int(c.get("pos", idx)),
        }
        for idx, c in enumerate(chunks)
        if c.get("chunk_id")
    ]

    with _get_driver() as driver:
        with driver.session(database=SETTINGS.neo4j_database) as session:
            _execute_write(
                session,
                """
                MATCH (n)
                WHERE n.file_hash = $file_hash
                  AND ($entity_label IN labels(n) OR $chunk_label IN labels(n))
                DETACH DELETE n
                """,
                file_hash=file_hash,
                entity_label=ENTITY_LABEL,
                chunk_label=CHUNK_LABEL,
            )
            _execute_write(
                session,
                f"""
                UNWIND $nodes AS row
                CREATE (e:{ENTITY_LABEL} {{
                    file_hash: $file_hash,
                    node_id: row.id,
                    label: row.label,
                    type: row.type,
                    attrs_json: row.attrs_json
                }})
                """,
                file_hash=file_hash,
                nodes=[
                    {
                        "id": n.get("id", ""),
                        "label": n.get("label", ""),
                        "type": n.get("type", "CONCEPT"),
                        "attrs_json": str(n.get("attrs", {})),
                    }
                    for n in nodes
                    if n.get("id") and n.get("label")
                ],
            )
            _execute_write(
                session,
                f"""
                UNWIND $chunks AS row
                CREATE (c:{CHUNK_LABEL} {{
                    file_hash: $file_hash,
                    chunk_id: row.chunk_id,
                    parent_id: row.parent_id,
                    text: row.text,
                    pos: row.pos
                }})
                """,
                file_hash=file_hash,
                chunks=chunk_rows,
            )
            _execute_write(
                session,
                f"""
                UNWIND $edges AS row
                MATCH (a:{ENTITY_LABEL} {{file_hash: $file_hash, node_id: row.from}})
                MATCH (b:{ENTITY_LABEL} {{file_hash: $file_hash, node_id: row.to}})
                CREATE (a)-[:{RELATION_TYPE} {{
                    file_hash: $file_hash,
                    relation: row.relation,
                    weight: row.weight
                }}]->(b)
                """,
                file_hash=file_hash,
                edges=edges,
            )
            mentions = [
                {"node_id": node_id, "chunk_id": chunk_id}
                for node_id, chunk_ids in entity_chunks.items()
                for chunk_id in chunk_ids
            ]
            _execute_write(
                session,
                f"""
                UNWIND $mentions AS row
                MATCH (e:{ENTITY_LABEL} {{file_hash: $file_hash, node_id: row.node_id}})
                MATCH (c:{CHUNK_LABEL} {{file_hash: $file_hash, chunk_id: row.chunk_id}})
                CREATE (e)-[:{MENTION_TYPE} {{file_hash: $file_hash}}]->(c)
                """,
                file_hash=file_hash,
                mentions=mentions,
            )


def load_semantic_graph(file_hash: str) -> Dict[str, Any]:
    if not is_neo4j_enabled():
        return {}

    with _get_driver() as driver:
        with driver.session(database=SETTINGS.neo4j_database) as session:
            node_records = _execute_read(
                session,
                f"""
                MATCH (e:{ENTITY_LABEL} {{file_hash: $file_hash}})
                RETURN e.node_id AS id, e.label AS label, e.type AS type
                ORDER BY e.node_id
                """,
                file_hash=file_hash,
            )
            edge_records = _execute_read(
                session,
                f"""
                MATCH (a:{ENTITY_LABEL} {{file_hash: $file_hash}})
                    -[r:{RELATION_TYPE}]->(b:{ENTITY_LABEL} {{file_hash: $file_hash}})
                RETURN a.node_id AS from_id, b.node_id AS to_id,
                       r.relation AS relation, r.weight AS weight
                ORDER BY r.weight DESC
                """,
                file_hash=file_hash,
            )
            mention_records = _execute_read(
                session,
                f"""
                MATCH (e:{ENTITY_LABEL} {{file_hash: $file_hash}})
                    -[:{MENTION_TYPE}]->(c:{CHUNK_LABEL} {{file_hash: $file_hash}})
                RETURN e.node_id AS node_id, collect(c.chunk_id) AS chunk_ids
                """,
                file_hash=file_hash,
            )

    nodes = [
        {
            "id": record["id"],
            "label": record["label"],
            "type": record["type"] or "CONCEPT",
            "attrs": {},
        }
        for record in node_records
    ]
    edges = [
        {
            "from": record["from_id"],
            "to": record["to_id"],
            "relation": record["relation"] or "related_to",
            "weight": int(record["weight"] or 1),
        }
        for record in edge_records
    ]
    entity_chunks = {
        record["node_id"]: sorted(record["chunk_ids"] or [])
        for record in mention_records
    }
    return {"nodes": nodes, "edges": edges, "entity_chunks": entity_chunks}


def try_write_semantic_graph(file_hash: str, graph_data: Dict[str, Any], chunks: List[dict]) -> bool:
    global _NEO4J_AVAILABLE, _WARNED_UNAVAILABLE
    if _NEO4J_AVAILABLE is False:
        return False
    try:
        write_semantic_graph(file_hash, graph_data, chunks)
        _NEO4J_AVAILABLE = True
        return is_neo4j_enabled()
    except Exception as exc:
        _NEO4J_AVAILABLE = False
        if getattr(SETTINGS, "neo4j_enabled_fallback", True):
            if not _WARNED_UNAVAILABLE:
                logger.warning("Neo4j graph write failed, JSON export kept as fallback: %s", exc)
                _WARNED_UNAVAILABLE = True
            return False
        raise


def try_load_semantic_graph(file_hash: str) -> Dict[str, Any]:
    global _NEO4J_AVAILABLE, _WARNED_UNAVAILABLE
    if _NEO4J_AVAILABLE is False:
        return {}
    try:
        graph_data = load_semantic_graph(file_hash)
        _NEO4J_AVAILABLE = True
        return graph_data
    except Exception as exc:
        _NEO4J_AVAILABLE = False
        if getattr(SETTINGS, "neo4j_enabled_fallback", True):
            if not _WARNED_UNAVAILABLE:
                logger.warning("Neo4j graph load failed, falling back to JSON graph: %s", exc)
                _WARNED_UNAVAILABLE = True
            return {}
        raise
