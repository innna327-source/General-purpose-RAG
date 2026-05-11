from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from graph.neo4j_store import write_semantic_graph
from retrieval.bm25 import load_chunks_jsonl


def migrate(file_hash: str) -> None:
    graph_path = SETTINGS.graph_dir / f"{file_hash}_semantic_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {graph_path}")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f) or {}
    chunks = load_chunks_jsonl(file_hash)
    write_semantic_graph(file_hash=file_hash, graph_data=graph_data, chunks=chunks)

    print(
        "migrated graph to Neo4j: "
        f"file_hash={file_hash} nodes={len(graph_data.get('nodes', []))} "
        f"edges={len(graph_data.get('edges', []))} chunks={len(chunks)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate exported semantic graph JSON into Neo4j.")
    parser.add_argument("file_hash", help="Index/file hash, without _semantic_graph.json suffix.")
    args = parser.parse_args()
    migrate(args.file_hash)


if __name__ == "__main__":
    main()
