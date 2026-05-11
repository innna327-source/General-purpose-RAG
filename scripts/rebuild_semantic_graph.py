from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SETTINGS
from graph.semantic_graph import build_semantic_graph


def _load_chunks(file_hash: str) -> list[dict]:
    chunk_path = SETTINGS.index_dir / f"{file_hash}.chunks.jsonl"
    if not chunk_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunk_path}")

    chunks: list[dict] = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild semantic graph from chunks.")
    parser.add_argument("--file-hash", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    chunks = _load_chunks(args.file_hash)
    print(f"Loaded chunks: {len(chunks)}")
    out_path = build_semantic_graph(
        chunks,
        file_hash=args.file_hash,
        graph_dir=SETTINGS.graph_dir,
        max_workers=args.workers,
        progress_every=args.progress_every,
        batch_size=args.batch_size,
    )

    with open(out_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    print(f"Graph path: {Path(out_path)}")
    print(f"Nodes: {len(graph_data.get('nodes', []))}")
    print(f"Edges: {len(graph_data.get('edges', []))}")
    print(f"Entity chunks: {len(graph_data.get('entity_chunks', {}))}")


if __name__ == "__main__":
    main()
