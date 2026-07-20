
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


# ------------------------------------------------------------------ loading --
def load(derived: Path):
    comps: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with (derived / "components.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            comps[(int(row["circuit_id"]), int(row["drawing"]))].append(row)

    nets: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    with (derived / "nets.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            members = [int(x) for x in row["components"].split(";") if x != ""]
            nets[(int(row["circuit_id"]), int(row["drawing"]))].append(members)
    return comps, nets


# ------------------------------------------------------------ graph building --
def build_class1(rows: list[dict], nets: list[list[int]], star: bool) -> nx.Graph:
    G = nx.Graph()
    for idx, r in enumerate(rows):
        if r["role"] == "device":
            G.add_node(idx, label=r["label"])
    for j, members in enumerate(nets):
        members = [m for m in members if m in G]
        if len(members) < 2:
            continue
        if star and len(members) > 2:
            hub = f"net{j}"                       # explicit hub, labelled as a net
            G.add_node(hub, label="__net__")
            for m in members:
                G.add_edge(hub, m)
        else:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    G.add_edge(members[a], members[b])
    return G


def build_class3(rows: list[dict], nets: list[list[int]]) -> nx.Graph:
    """Bipartite: device nodes + net nodes, edges are memberships."""
    G = nx.Graph()
    for idx, r in enumerate(rows):
        if r["role"] == "device":
            G.add_node(idx, label=r["label"])
    for j, members in enumerate(nets):
        members = [m for m in members if m in G]
        if not members:
            continue
        node = f"net{j}"
        G.add_node(node, label="__net__")
        for m in members:
            G.add_edge(node, m)
    return G


def build_all(comps, nets, rep: str, star: bool):
    graphs = {}
    for key, rows in comps.items():
        n = nets.get(key, [])
        graphs[key] = build_class1(rows, n, star) if rep == "class1" else build_class3(rows, n)
    return graphs


# -------------------------------------------------------------- distances ----
GED_KW = dict(
    node_subst_cost=lambda a, b: 0.0 if a.get("label") == b.get("label") else 1.0,
    node_del_cost=lambda a: 1.0,
    node_ins_cost=lambda a: 1.0,
    edge_subst_cost=lambda a, b: 0.0,
    edge_del_cost=lambda a: 1.0,
    edge_ins_cost=lambda a: 1.0,
)


def d_approx(A: nx.Graph, B: nx.Graph) -> float:
    return float(next(nx.optimize_graph_edit_distance(A, B, **GED_KW)))


def d_exact(A: nx.Graph, B: nx.Graph) -> float:
    return float(nx.graph_edit_distance(A, B, **GED_KW))


def d_histogram(A: nx.Graph, B: nx.Graph) -> float:
    a = Counter(d["label"] for _, d in A.nodes(data=True))
    b = Counter(d["label"] for _, d in B.nodes(data=True))
    keys = set(a) | set(b)
    return float(max(sum(max(0, a[k] - b[k]) for k in keys),
                     sum(max(0, b[k] - a[k]) for k in keys)))


DISTANCES = {"approx": d_approx, "exact": d_exact, "histogram": d_histogram}


def similarity(d: float, A: nx.Graph, B: nx.Graph) -> float:
    """Paper eq. (6) and (7)."""
    denom = (A.number_of_nodes() + B.number_of_nodes()) / 2.0
    return math.exp(-d / denom) if denom else 0.0


# ------------------------------------------------------------- evaluation ----
def evaluate(graphs, method: str, k: int, max_nodes: int | None, seed: int = 0):
    circuits = sorted({c for (c, d) in graphs if (c, 1) in graphs and (c, 2) in graphs})
    if max_nodes:
        circuits = [c for c in circuits
                    if graphs[(c, 1)].number_of_nodes() <= max_nodes
                    and graphs[(c, 2)].number_of_nodes() <= max_nodes]
    if len(circuits) < 2:
        raise SystemExit(f"Only {len(circuits)} circuits qualify. Relax --max-nodes.")

    dist = DISTANCES[method]
    ranks, t0 = [], time.perf_counter()

    for q in circuits:
        A = graphs[(q, 1)]
        scored = []
        for g in circuits:
            B = graphs[(g, 2)]
            scored.append((similarity(dist(A, B), A, B), g))
        rng = random.Random(seed + q)          # random tie-break, never by id
        scored.sort(key=lambda t: (-t[0], rng.random()))
        ranks.append(next(i for i, (_, g) in enumerate(scored, 1) if g == q))

    elapsed = time.perf_counter() - t0
    n = len(ranks)
    return {
        "circuits": n,
        "top1": sum(r == 1 for r in ranks) / n,
        "recall": sum(r <= k for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "median_rank": sorted(ranks)[n // 2],
        "random_recall": min(1.0, k / n),
        "sec_per_query": elapsed / n,
        "total_sec": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("--rep", choices=["class1", "class3"], default="class1")
    ap.add_argument("--method", choices=list(DISTANCES), default="approx")
    ap.add_argument("--star", action="store_true",
                    help="expand high fan-out nets as stars, not cliques")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="only circuits with graphs at or below this size")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    comps, nets = load(args.derived)
    graphs = build_all(comps, nets, args.rep, args.star)

    sizes = sorted(g.number_of_nodes() for g in graphs.values())
    edges = sorted(g.number_of_edges() for g in graphs.values())
    m = len(sizes) // 2
    print(f"{args.rep}: Nn mean {sum(sizes)/len(sizes):.2f} median {sizes[m]} max {sizes[-1]}"
          f" | Ne mean {sum(edges)/len(edges):.2f} median {edges[m]}")

    r = evaluate(graphs, args.method, args.k, args.max_nodes)

    print(f"\n{args.rep} / {args.method}   ({r['circuits']} circuits)")
    print(f"  top-1        {r['top1']:.3f}")
    print(f"  recall@{args.k}     {r['recall']:.3f}   (random {r['random_recall']:.3f})")
    print(f"  MRR          {r['mrr']:.3f}")
    print(f"  median rank  {r['median_rank']}")
    print(f"  AT           {r['sec_per_query']:.3f} s/query   "
          f"({r['total_sec']:.1f} s total)")


if __name__ == "__main__":
    main()