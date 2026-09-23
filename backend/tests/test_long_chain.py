"""Regression: a long single chain admits exponentially many co-optima.

R -c00-> n01 -c01-> ... -c23-> L1 (24 chain edges, delay 0, cap 1)
R -z (cap 0, fixed)-> L2
L1 window [12, 12], L2 window [0, 0].

Optima: exactly 12 positive edges carrying exactly 12 units of compensation;
every 12-subset of the 24 chain edges is co-optimal for stages 1 & 2, i.e.
C(24, 12) = 2,704,156 solutions. The previous solver enumerated them one MILP
per solution and never returned; the response time must not grow with that
count. The canonical (lexicographically smallest) vector pushes every unit to
the tail of the id ordering: c00..c11 = 0, c12..c23 = 1, z = 0.
"""

from __future__ import annotations

import math
import time

from app.solver import Edge, Window, build_model, solve

CHAIN_LEN = 24
CO_OPTIMAL_COUNT = math.comb(CHAIN_LEN, 12)


def _long_chain_batch():
    # 24 chain edges => 25 vertices R, n01..n23 (23 internal), L1; plus L2.
    chain = ["R"] + [f"n{k:02d}" for k in range(1, CHAIN_LEN)] + ["L1"]
    assert len(chain) == CHAIN_LEN + 1
    nodes = chain + ["L2"]
    edges = [
        Edge(
            id=f"c{k:02d}",
            source=chain[k],
            target=chain[k + 1],
            delay=0,
            cap=1,
        )
        for k in range(CHAIN_LEN)
    ]
    # fixed edge (cap 0): L2 is pinned to arrival 0 no matter what
    edges.append(Edge(id="z", source="R", target="L2", delay=0, cap=0))
    windows = [Window("L1", 12, 12), Window("L2", 0, 0)]
    return nodes, edges, windows


def test_long_chain_co_optima_complete_result_in_time():
    nodes, edges, windows = _long_chain_batch()
    # 25 edges (24 chain edges + z) force 26 vertices in any tree; well below
    # the 55-node business limit.
    assert len(nodes) == 26
    assert len(edges) == 25
    m = build_model(nodes, edges, windows)

    t0 = time.time()
    r = solve(m)
    dt = time.time() - t0

    assert r["status"] == "feasible", r.get("conflict")
    # must return promptly; the old enumeration issued one MILP per each of
    # the 2,704,156 co-optima and never finished.
    assert dt < 30, f"solve took {dt:.1f}s"
    print(f"\n24-edge chain ({CO_OPTIMAL_COUNT} co-optima): {dt:.2f}s")

    assert r["objectives"]["positive_edges"] == 12
    assert r["objectives"]["total_compensation"] == 12

    order = r["objectives"]["vector_order"]
    vector = r["objectives"]["vector"]
    assert order == [f"c{k:02d}" for k in range(CHAIN_LEN)] + ["z"]
    # lexicographic tie-break pushes compensation to the largest ids
    assert vector == [0] * 12 + [1] * 12 + [0], vector

    by_id = {row["id"]: row for row in r["edges"]}
    for k in range(CHAIN_LEN):
        row = by_id[f"c{k:02d}"]
        assert (row["min"], row["max"]) == (0, 1), (row["id"], row["min"], row["max"])
    assert (by_id["z"]["min"], by_id["z"]["max"]) == (0, 0)
    assert by_id["z"]["chosen"] == 0
    # chosen column matches the canonical vector
    for k in range(CHAIN_LEN):
        assert by_id[f"c{k:02d}"]["chosen"] == vector[k]

    arrivals = {x["node"]: x for x in r["leaves"]}
    assert arrivals["L1"]["arrival"] == 12
    assert arrivals["L2"]["arrival"] == 0
    assert arrivals["L1"]["margin"] == 0
    assert arrivals["L2"]["margin"] == 0
    assert len(r["tree"]["rows"]) == 26
