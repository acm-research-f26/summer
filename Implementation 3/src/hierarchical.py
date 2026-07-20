
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

GED_KW = dict(
    node_subst_cost=lambda a, b: 0.0 if a.get("label") == b.get("label") else 1.0,
    node_del_cost=lambda a: 1.0,
    node_ins_cost=lambda a: 1.0,
    edge_subst_cost=lambda a, b: 0.0,
    edge_del_cost=lambda a: 1.0,
    edge_ins_cost=lambda a: 1.0,
)


# ------------------------------------------------------------------ loading --
def load(derived: Path):
    comps = defaultdict(list)
    with (derived / "components.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            comps[(int(row["circuit_id"]), int(row["drawing"]))].append(row)
    nets = defaultdict(list)
    with (derived / "nets.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            members = [int(x) for x in row["components"].split(";") if x]
            nets[(int(row["circuit_id"]), int(row["drawing"]))].append(members)
    return comps, nets


def attachment(rows, nets_for_key) -> float:
    devices = {i for i, r in enumerate(rows) if r["role"] == "device"}
    if not devices:
        return 0.0
    attached = {m for members in nets_for_key for m in members} & devices
    return len(attached) / len(devices)


# ------------------------------------------------------------ graph building --
def build(rows, nets, rep: str, star: bool) -> nx.Graph:
    G = nx.Graph()
    for idx, r in enumerate(rows):
        if r["role"] == "device":
            G.add_node(idx, label=r["label"])
    for j, members in enumerate(nets):
        members = [m for m in members if m in G]
        if rep == "class3":
            if members:
                hub = f"n{j}"
                G.add_node(hub, label="__net__")
                for m in members:
                    G.add_edge(hub, m)
            continue
        if len(members) < 2:
            continue
        if star and len(members) > 2:
            hub = f"n{j}"
            G.add_node(hub, label="__net__")
            for m in members:
                G.add_edge(hub, m)
        else:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    G.add_edge(members[a], members[b])
    return G


# -------------------------------------------------------------- distances ----
def ged_approx(A, B) -> float:
    return float(next(nx.optimize_graph_edit_distance(A, B, **GED_KW)))


def similarity(d, A, B) -> float:
    denom = (A.number_of_nodes() + B.number_of_nodes()) / 2.0
    return math.exp(-d / denom) if denom else 0.0


def sim_matrix(queries, database, tag: str, cache_dir: Path, verbose=True):
    """Full |Q| x |D| similarity matrix, cached to disk."""
    sig = hashlib.md5(
        (tag + "|" + ",".join(f"{g.number_of_nodes()}:{g.number_of_edges()}"
                              for g in list(queries) + list(database))).encode()
    ).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"sim_{tag}_{sig}.npz"

    if path.exists():
        if verbose:
            print(f"  [cache hit] {path.name}")
        z = np.load(path)
        return z["S"], float(z["seconds"])

    S = np.zeros((len(queries), len(database)))
    t0 = time.perf_counter()
    for i, A in enumerate(queries):
        for j, B in enumerate(database):
            S[i, j] = similarity(ged_approx(A, B), A, B)
        if verbose and (i + 1) % 10 == 0:
            done = (i + 1) / len(queries)
            el = time.perf_counter() - t0
            print(f"    {i+1}/{len(queries)}  {el:.0f}s elapsed, "
                  f"~{el/done - el:.0f}s left", flush=True)
    elapsed = time.perf_counter() - t0
    np.savez_compressed(path, S=S, seconds=elapsed)
    return S, elapsed


# ------------------------------------------------------------------ ranking --
def topk_from_row(row: np.ndarray, k: int, rng: random.Random):
    order = sorted(range(len(row)), key=lambda j: (-row[j], rng.random()))
    return order[:k]


def metrics(rankings, k: int):
    ranks = []
    for i, order in enumerate(rankings):
        ranks.append(order.index(i) + 1 if i in order else len(order) + 1)
    n = len(ranks)
    return {
        "top1": sum(r == 1 for r in ranks) / n,
        "recall": sum(r <= k for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("--topk", type=int, default=20, help="stage-1 shortlist size")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--star", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-attachment", type=float, default=0.5,
                    help="drop drawings where topology extraction largely failed")
    args = ap.parse_args()

    comps, nets = load(args.derived)

    circuits = sorted({c for (c, d) in comps if (c, 1) in comps and (c, 2) in comps})
    kept, dropped = [], 0
    for c in circuits:
        a1 = attachment(comps[(c, 1)], nets.get((c, 1), []))
        a2 = attachment(comps[(c, 2)], nets.get((c, 2), []))
        if min(a1, a2) >= args.min_attachment:
            kept.append(c)
        else:
            dropped += 1
    if args.limit:
        kept = kept[: args.limit]

    print(f"Circuits: {len(kept)} kept, {dropped} dropped "
          f"(attachment < {args.min_attachment})")

    g1q = [build(comps[(c, 1)], nets.get((c, 1), []), "class1", args.star) for c in kept]
    g1d = [build(comps[(c, 2)], nets.get((c, 2), []), "class1", args.star) for c in kept]
    g3q = [build(comps[(c, 1)], nets.get((c, 1), []), "class3", False) for c in kept]
    g3d = [build(comps[(c, 2)], nets.get((c, 2), []), "class3", False) for c in kept]

    for name, gs in (("class1", g1q + g1d), ("class3", g3q + g3d)):
        nn = sorted(g.number_of_nodes() for g in gs)
        ne = sorted(g.number_of_edges() for g in gs)
        m = len(nn) // 2
        print(f"  {name}: Nn mean {sum(nn)/len(nn):.2f} median {nn[m]} max {nn[-1]}"
              f" | Ne mean {sum(ne)/len(ne):.2f} median {ne[m]} max {ne[-1]}")

    cache = args.derived / "cache"
    print("\nStage 1 (class1) similarity matrix:")
    S1, t1 = sim_matrix(g1q, g1d, "class1" + ("_star" if args.star else ""), cache)
    print(f"  {t1:.1f}s total, {t1/len(kept)*1000:.1f} ms/query")

    print("Stage 2 (class3) similarity matrix:")
    S3, t3 = sim_matrix(g3q, g3d, "class3", cache)
    print(f"  {t3:.1f}s total, {t3/len(kept)*1000:.1f} ms/query")

    rng = random.Random(0)
    pure1 = [topk_from_row(S1[i], args.k, random.Random(i)) for i in range(len(kept))]
    pure3 = [topk_from_row(S3[i], args.k, random.Random(i)) for i in range(len(kept))]

    hier = []
    for i in range(len(kept)):
        shortlist = topk_from_row(S1[i], args.topk, random.Random(i))
        rescored = sorted(shortlist,
                          key=lambda j: (-S3[i, j], random.Random(i * 1000 + j).random()))
        hier.append(rescored[: args.k])

    # Hierarchical cost: full stage 1, then stage 2 on topk only.
    t_hier = t1 + t3 * (args.topk / len(kept))

    print(f"\n{'':<16}{'top-1':>8}{'recall@'+str(args.k):>10}{'MRR':>8}{'sec/query':>12}")
    for name, rk, tt in (("class1 only", pure1, t1),
                         ("class3 only", pure3, t3),
                         (f"hierarchical", hier, t_hier)):
        m = metrics(rk, args.k)
        print(f"  {name:<14}{m['top1']:>8.3f}{m['recall']:>10.3f}"
              f"{m['mrr']:>8.3f}{tt/len(kept):>12.3f}")

    # ---- self-consistency: does the hierarchy reproduce pure class3? --------
    overlap = [len(set(h) & set(p)) / args.k for h, p in zip(hier, pure3)]
    exact_same = sum(1 for h, p in zip(hier, pure3) if h == p)
    recovered = sum(1 for h, p in zip(hier, pure3) if p and p[0] in h)

    print(f"\n--- self-consistency vs pure class3 (needs no ground truth) ---")
    print(f"  mean top-{args.k} overlap        {sum(overlap)/len(overlap):.3f}")
    print(f"  identical top-{args.k} ordering  {exact_same}/{len(kept)}"
          f"  ({exact_same/len(kept):.1%})")
    print(f"  class3 top-1 retained      {recovered}/{len(kept)}"
          f"  ({recovered/len(kept):.1%})")
    print(f"\n  Chen asserts the hierarchy matches the detailed representation.")
    print(f"  These three numbers are that assertion, measured.")


if __name__ == "__main__":
    main()