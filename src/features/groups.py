"""Group-aware splitting: MOFs are linked when they share a refcode family or a DOI; each connected component is a group that
is placed wholly in train, calibration or test.

MOFid is not available in the released data (see notes/stability_inventory.md), so these are proxy groups."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def family_key(name: str, k: int = 4) -> str:
    """CSD-refcode family proxy: the first k letters."""
    return str(name)[:k]


def build_groups(names: Iterable[str], doi_by_name: Optional[Dict[str, str]] = None, k: int = 4) -> pd.DataFrame:
    """Connected components over (shared refcode family) OR (shared DOI). Returns one row per MOF: name, group_id, group_size.
    group_id is the rank of the group by (size desc, smallest name) so it is deterministic."""
    names = sorted(set(map(str, names)))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def link(keys: Dict[str, list]):
        for members in keys.values():
            for m in members[1:]:
                union(members[0], m)

    by_family: Dict[str, list] = {}
    for n in names:
        by_family.setdefault(family_key(n, k), []).append(n)
    link(by_family)
    if doi_by_name:
        by_doi: Dict[str, list] = {}
        for n in names:
            d = doi_by_name.get(n)
            if isinstance(d, str) and d:
                by_doi.setdefault(d, []).append(n)
        link(by_doi)
    comp: Dict[str, list] = {}
    for n in names:
        comp.setdefault(find(n), []).append(n)
    ordered = sorted(comp.values(), key=lambda m: (-len(m), min(m)))
    rows = [(n, gid, len(m)) for gid, m in enumerate(ordered) for n in m]
    return pd.DataFrame(rows, columns=["name", "group_id", "group_size"]).sort_values("name").reset_index(drop=True)


def assign_groups(groups: pd.DataFrame, fractions: Dict[str, float], seed: int = 0) -> pd.Series:
    """Assign whole groups to splits so the MOF-count fractions are met as closely as possible: groups are visited from largest to
    smallest (ties shuffled with `seed`) and each goes to the split with the largest remaining relative deficit."""
    assert abs(sum(fractions.values()) - 1) < 1e-9
    rng = np.random.RandomState(seed)
    sizes = groups.drop_duplicates("group_id").set_index("group_id")["group_size"]
    order = sorted(sizes.index, key=lambda g: (-sizes[g], rng.rand()))
    total = int(sizes.sum())
    target = {s: f * total for s, f in fractions.items()}
    filled = {s: 0 for s in fractions}
    where: Dict[int, str] = {}
    for g in order:
        s = max(fractions, key=lambda s: (target[s] - filled[s]) / target[s])
        where[g] = s
        filled[s] += int(sizes[g])
    return groups["group_id"].map(where).rename("split")


def random_assign(names: Sequence[str], fractions: Dict[str, float], seed: int = 0) -> pd.Series:
    """MOF-level random split with the same fractions (the released scripts' style), for comparison."""
    rng = np.random.RandomState(seed)
    names = list(names)
    perm = rng.permutation(len(names))
    cuts = np.cumsum([f for f in fractions.values()])[:-1] * len(names)
    parts = np.split(perm, np.round(cuts).astype(int))
    out = pd.Series(index=names, dtype=object)
    for (s, _), p in zip(fractions.items(), parts):
        out.iloc[p] = s
    return out


def size_summary(groups: pd.DataFrame) -> pd.DataFrame:
    sz = groups.drop_duplicates("group_id")["group_size"]
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 10 ** 9)]
    rows = []
    for lo, hi in bins:
        m = (sz >= lo) & (sz <= hi)
        rows.append(dict(size=f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}"), groups=int(m.sum()), mofs=int(sz[m].sum())))
    return pd.DataFrame(rows)


def random_group_assign(groups: pd.DataFrame, fractions: Dict[str, float], seed: int = 0) -> pd.Series:
    """Random GROUP-level split: groups are shuffled (all of them, including the largest) and cut where the cumulative MOF count reaches each split's share.
    Unlike `assign_groups` (deterministic apart from ties, designed to hit the fractions closely), this explores the split-to-split variability; the realised
    fractions are only approximate when a large group straddles a cut."""
    assert abs(sum(fractions.values()) - 1) < 1e-9
    rng = np.random.RandomState(seed)
    sizes = groups.drop_duplicates("group_id").set_index("group_id")["group_size"]
    perm = rng.permutation(sizes.index.values)
    cum = np.cumsum(sizes.loc[perm].values) / sizes.sum()
    edges = np.cumsum(list(fractions.values()))
    labels = list(fractions)
    where = {g: labels[min(int(np.searchsorted(edges, c - 1e-12)), len(labels) - 1)] for g, c in zip(perm, cum)}
    return groups["group_id"].map(where).rename("split")
