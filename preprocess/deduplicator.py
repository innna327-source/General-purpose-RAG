from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def deduplicate_paragraphs(
    paragraphs: List[str],
    model_name: str,
    similarity_threshold: float,
) -> List[str]:
    if not paragraphs:
        return []

    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(
        paragraphs,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    kept: list[str] = []
    kept_embs: list[np.ndarray] = []
    for p, e in zip(paragraphs, embs):
        if not kept_embs:
            kept.append(p)
            kept_embs.append(e)
            continue
        sims = np.dot(np.stack(kept_embs, axis=0), e)
        if float(np.max(sims)) > similarity_threshold:
            continue
        kept.append(p)
        kept_embs.append(e)
    return kept

