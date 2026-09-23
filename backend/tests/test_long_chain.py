"""Acceptance test for the 24-edge / 27-node long chain.

The chain admits C(24, 12) = 2,704,156 stage-1&2 co-optimal vectors, so any
algorithm that enumerates co-optima (or forks one MILP per partial vector)
appears to hang even though the batch is far below the 55-node business
limit. The solver must instead derive the canonical vector and the per-edge
ranges with a number of MILP calls linear in the edge count.
"""

from __future__ import annotations

import time

from app.solver import Edge, Window, build_model, solve


def _long_chain_batch():
    # R -c00-> n01 -c01-> ... -c23-> L1 ; and R -z-> L2
    nodes = ["R"] + [f"n{i:02d}" for i in range(1, 24)] + ["L1", "L2"]
    edges = []
    prev = "R"
    for i in range(24):
        nxt = "L1" if i == 23 else f"n{i + 1:02d}"
        edges.append(Edge(f"c{i:02d}", prev, nxt, 0, 1))
        prev = nxt
    edges.append(Edge("z", "R", "L2", 0, 1))
    windows = [Window("L1", 12, 12), Window("L2", 0, 0)]
    return nodes, edges, windows


def test_long_chain_returns_full_result_in_time():
    nodes, edges, windows = _long_chain_batch()
    # 24 chain edges + z = 25 edges -> the tree has 26 nodes (the audit brief
    # counts the 24 chain edges as if nodes and says 27); either way far below
    # the 55-node business limit.
    assert len(nodes) == 26
    assert len(edges) == 25

    t0 = time.time()
    r = solve(build_model(nodes, edges, windows))
    dt = time.time() - t0

    assert r["status"] == "feasible", r.get("conflict")
    assert dt < 30, f"solve took {dt:.1f}s"

    obj = r["objectives"]
    assert obj["positive_edges"] == 12
    assert obj["total_compensation"] == 12

    # canonical vector ordered by edge id ASCII: c00..c23, z
    assert obj["vector_order"] == [f"c{i:02d}" for i in range(24)] + ["z"]
    assert obj["vector"] == [0] * 12 + [1] * 12 + [0]

    by_id = {row["id"]: row for row in r["edges"]}
    for i in range(24):
        row = by_id[f"c{i:02d}"]
        assert row["chosen"] == (0 if i < 12 else 1)
        assert (row["min"], row["max"]) == (0, 1)
    assert (by_id["z"]["chosen"], by_id["z"]["min"], by_id["z"]["max"]) == (0, 0, 0)

    arrivals = {x["node"]: x for x in r["leaves"]}
    assert arrivals["L1"]["arrival"] == 12
    assert arrivals["L2"]["arrival"] == 0


def test_two_interchangeable_edges_on_one_path():
    # R -a-> A -b-> L, both cap 1, L requires arrival 1: exactly one of
    # {a, b} carries the unit -> 2 co-optima; lexicographically smallest in
    # edge-id order minimizes a first: a=0,b=1; both edges range over 0..1.
    # A fixed (cap 0) second leaf keeps the batch valid.
    nodes = ["R", "A", "L", "L2"]
    edges = [
        Edge("a", "R", "A", 0, 1),
        Edge("b", "A", "L", 0, 1),
        Edge("z", "R", "L2", 0, 0),
    ]
    r = solve(build_model(nodes, edges, [Window("L", 1, 1), Window("L2", 0, 0)]))
    assert r["status"] == "feasible"
    assert r["objectives"]["positive_edges"] == 1
    assert r["objectives"]["total_compensation"] == 1
    d = dict(zip(r["objectives"]["vector_order"], r["objectives"]["vector"]))
    assert d == {"a": 0, "b": 1, "z": 0}
    by_id = {x["id"]: x for x in r["edges"]}
    assert (by_id["a"]["min"], by_id["a"]["max"]) == (0, 1)
    assert (by_id["b"]["min"], by_id["b"]["max"]) == (0, 1)
    assert (by_id["z"]["min"], by_id["z"]["max"]) == (0, 0)


def test_unique_solution_is_pinned_everywhere():
    # independent leaf edges with closed-point windows admit exactly one
    # vector: chosen = min = max on every edge.
    nodes = ["R", "L1", "L2"]
    edges = [
        Edge("a", "R", "L1", 0, 16),
        Edge("b", "R", "L2", 0, 16),
    ]
    r = solve(build_model(nodes, edges, [Window("L1", 3, 3), Window("L2", 5, 5)]))
    assert r["objectives"]["positive_edges"] == 2
    assert r["objectives"]["total_compensation"] == 8
    assert r["objectives"]["vector"] == [3, 5]
    by_id = {x["id"]: x for x in r["edges"]}
    assert (by_id["a"]["chosen"], by_id["a"]["min"], by_id["a"]["max"]) == (3, 3, 3)
    assert (by_id["b"]["chosen"], by_id["b"]["min"], by_id["b"]["max"]) == (5, 5, 5)
