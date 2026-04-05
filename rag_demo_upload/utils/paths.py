from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # rag_demo/utils/paths.py -> rag_demo/
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs() -> None:
    root = project_root()
    for rel in ["index", "graph", "logs", "output", "data/raw_pdf", "data/processed"]:
        ensure_dir(root / rel)

