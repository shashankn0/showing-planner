"""Choose which photos the user should label next.

Early on (cold start) there is no model to ask, so pick a *diverse* spread: cluster the photo
embeddings and take the photo nearest each cluster center, balanced across room types. Later, ask
the model where it is least sure and pick a diverse subset of those, so labels aren't spent on
near-identical photos. Every few picks a random photo from the permanent holdout is mixed in so
the final evaluation stays unbiased.
"""

from __future__ import annotations

import numpy as np

from vision.config import ROOMS
from vision.heads import RidgeThresholdHead, make_features

HOLDOUT_EVERY = 5  # every 5th photo shown comes from the holdout pool
COLD_START_LABELS = 100  # below this many labels, use diverse sampling


def _farthest_first(X: np.ndarray, k: int, seed_idx: int = 0) -> list[int]:
    """Greedy k-center: repeatedly take the point farthest (by cosine) from those already chosen."""
    k = min(k, len(X))
    if k == 0:
        return []
    chosen = [seed_idx]
    dist = 1.0 - X @ X[seed_idx]
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 1e-9:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, 1.0 - X @ X[nxt])
    return chosen


def pick_diverse(X: np.ndarray, rooms: list[str], k: int, seed: int = 0) -> list[int]:
    """Cold start: cluster per room type and take the photo nearest each center."""
    from sklearn.cluster import KMeans

    if len(X) <= k:
        return list(range(len(X)))
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rooms):
        groups.setdefault(r if r in ROOMS else "other", []).append(i)
    quota = {r: max(1, k // len(groups)) for r in groups}
    out: list[int] = []
    for r, idx in groups.items():
        q = min(quota[r], len(idx))
        if q >= len(idx):
            out += idx
            continue
        km = KMeans(n_clusters=q, n_init=3, random_state=seed).fit(X[idx])
        for c in range(q):
            members = np.flatnonzero(km.labels_ == c)
            if len(members):
                d = np.linalg.norm(X[idx][members] - km.cluster_centers_[c], axis=1)
                out.append(idx[int(members[d.argmin()])])
    rng = np.random.default_rng(seed)
    rest = [i for i in range(len(X)) if i not in set(out)]
    if len(out) < k and rest:  # top up with random photos
        out += list(rng.choice(rest, size=min(k - len(out), len(rest)), replace=False))
    return out[:k]


def pick_informative(
    X_pool: np.ndarray,
    rooms_pool: list[str],
    X_lab: np.ndarray,
    rooms_lab: list[str],
    y_lab: np.ndarray,
    k: int,
    seed: int = 0,
    candidate_factor: int = 5,
) -> list[int]:
    """Train a quick model, find the photos it is least sure about, then thin them for diversity."""
    if len(X_pool) <= k:
        return list(range(len(X_pool)))
    head = RidgeThresholdHead().fit(make_features(X_lab, rooms_lab), y_lab, seed=seed)
    unsure = 1.0 - head.confidence(make_features(X_pool, rooms_pool))
    top = np.argsort(-unsure)[: k * candidate_factor]
    sub = X_pool[top]
    picked = _farthest_first(sub, k, seed_idx=0)  # index 0 = the single most uncertain photo
    return [int(top[i]) for i in picked]


def interleave_holdout(train_picks: list[int], holdout_picks: list[int]) -> list[tuple[str, int]]:
    """Merge into one queue: [("train", i) ...] with a holdout item every HOLDOUT_EVERY-th slot."""
    out: list[tuple[str, int]] = []
    t, h = list(train_picks), list(holdout_picks)
    while t or h:
        slot = len(out) + 1
        if slot % HOLDOUT_EVERY == 0 and h:
            out.append(("holdout", h.pop(0)))
        elif t:
            out.append(("train", t.pop(0)))
        elif h:
            out.append(("holdout", h.pop(0)))
    return out
