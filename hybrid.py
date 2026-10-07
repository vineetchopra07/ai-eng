# hybrid.py
import numpy as np
from rank_bm25 import BM25Okapi
from search import corpus, vectors, model

tokenized = [c["text"].lower().split() for c in corpus]
bm25 = BM25Okapi(tokenized)


def dense_indices(query: str, k: int = 20) -> list[int]:
    q = model.encode(query)
    q = q / np.linalg.norm(q)
    return list(np.argsort(-(vectors @ q))[:k])


def bm25_indices(query: str, k: int = 20) -> list[int]:
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(-scores)[:k])


def rrf(lists: list[list[int]], c: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for indices in lists:
        for rank, idx in enumerate(indices):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def hybrid_search(query: str, k: int = 5):
    fused = rrf([dense_indices(query), bm25_indices(query)])
    return [(score, corpus[i]) for i, score in fused[:k]]

