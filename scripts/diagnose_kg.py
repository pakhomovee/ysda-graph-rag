"""Can the knowledge graph even support the multi-hop questions we ask of it?

QAFD retrieves at 0.4935 recall@10 on our corpus while plain dense retrieval over
the same passages with the same encoder gets 0.5733. A graph retriever losing to
its own flat baseline has three candidate causes -- the KG, the hyperparameters,
or the diffusion -- and this script rules on the first one, offline, with no LLM
and no GPU.

The question a multi-hop retriever has to answer is: starting from one gold
passage, can flow reach the others? In a passage-entity graph, two passages are
at distance 2 when they share an extracted entity, 4 when they are joined through
one intermediate passage, and so on. If a question's gold passages are far apart
-- or in different components -- then no weighting of any edge can retrieve them
together, and the ceiling is a property of the extraction, not of the method.

Reads the dump from scripts/export_qafd_nodes.sh.

    python scripts/diagnose_kg.py musique
    python scripts/diagnose_kg.py musique --sample 1000
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):
    sys.exit(f"needs the mbuzai env (3.10+), got {sys.version.split()[0]}")

from mbuzai import dataio  # noqa: E402

OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--nodes", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=300,
                    help="questions to run shortest paths for (0 = all)")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    npz = args.nodes or OUT / f"qafd_nodes_{args.dataset}.npz"
    if not npz.exists():
        sys.exit(f"missing {npz}\n  build it first (QAFD env): bash scripts/export_qafd_nodes.sh")
    z = np.load(npz, allow_pickle=False)

    ds = dataio.load(args.dataset)
    n_nodes = int(max(z["entity_vertex"].max(), z["passage_vertex"].max())) + 1

    # pid -> passage vertex, and pid -> entities extracted from it
    pid_to_vertex = {int(p): int(v) for p, v in zip(z["passage_pid"], z["passage_vertex"])
                     if p >= 0 and v >= 0}
    pid_to_ents = collections.defaultdict(set)
    for e, p in zip(z["ent_gold_entity"], z["ent_gold_pid"]):
        pid_to_ents[int(p)].add(int(e))

    print(f"graph: {n_nodes} nodes, {len(z['edges'])} edges, "
          f"{len(pid_to_vertex)} passages mapped to vertices")

    # ---- 1. extraction coverage -------------------------------------------
    gold_pids = sorted({int(p) for q in ds.queries for p in q.gold_pids})
    empty = [p for p in gold_pids if not pid_to_ents.get(p)]
    sizes = [len(pid_to_ents.get(p, ())) for p in gold_pids]
    print(f"\n1. extraction coverage over {len(gold_pids)} distinct gold passages")
    print(f"   with no extracted entity : {len(empty)} ({len(empty)/len(gold_pids):.1%})")
    print(f"   entities per gold passage: mean {np.mean(sizes):.1f}, "
          f"median {int(np.median(sizes))}, min {min(sizes)}, max {max(sizes)}")

    # ---- 2. do a question's gold passages share an entity? -----------------
    # Distance 2 in the passage-entity graph. This is the cheapest possible
    # bridge: without it the diffusion has to route through an intermediate
    # passage, and multi-hop traversal is exactly what is supposed to be hard.
    print("\n2. gold passages sharing at least one extracted entity")
    by_hops = collections.defaultdict(lambda: [0, 0])
    for q in ds.queries:
        pids = sorted(q.gold_pids)
        if len(pids) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(pids) for b in pids[i + 1:]]
        linked = sum(1 for a, b in pairs if pid_to_ents.get(a, set()) & pid_to_ents.get(b, set()))
        rec = by_hops[q.n_hops]
        rec[0] += linked
        rec[1] += len(pairs)
    tot_l = sum(v[0] for v in by_hops.values())
    tot_p = sum(v[1] for v in by_hops.values())
    for h in sorted(by_hops):
        l, t = by_hops[h]
        print(f"   {h}hop: {l}/{t} gold pairs share an entity ({l/max(t,1):.1%})")
    print(f"   all  : {tot_l}/{tot_p} ({tot_l/max(tot_p,1):.1%})")

    # ---- 3. graph distance between a question's gold passages -------------
    # CSR adjacency, built with numpy so the diagnostic needs no scipy.
    e = z["edges"]
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.argsort(src, kind="stable")
    indices = dst[order].astype(np.int64)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n_nodes), out=indptr[1:])

    def bfs_to(start, targets, max_dist):
        """Distance from `start` to each target, stopping as soon as all are
        found. Frontier expansion is vectorised via the ragged-gather idiom, so
        this is a few ms per source even on ~1M edges."""
        want = set(targets)
        found, seen = {}, np.zeros(n_nodes, dtype=bool)
        seen[start] = True
        frontier = np.array([start], dtype=np.int64)
        for d in range(1, max_dist + 1):
            if frontier.size == 0 or not want:
                break
            starts, ends = indptr[frontier], indptr[frontier + 1]
            counts = ends - starts
            if counts.sum() == 0:
                break
            offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            nbrs = indices[np.repeat(starts, counts) + offs]
            nbrs = np.unique(nbrs[~seen[nbrs]])
            if nbrs.size == 0:
                break
            seen[nbrs] = True
            for t in list(want):
                if seen[t]:
                    found[t] = d
                    want.discard(t)
            frontier = nbrs
        return found

    qs = [q for q in ds.queries if len(q.gold_pids) >= 2
          and all(int(p) in pid_to_vertex for p in q.gold_pids)]
    rng = np.random.default_rng(args.seed)
    if args.sample and len(qs) > args.sample:
        qs = [qs[i] for i in rng.choice(len(qs), args.sample, replace=False)]
    print(f"\n3. shortest path between gold passages, {len(qs)} questions sampled")

    dists = collections.Counter()
    worst_by_hops = collections.defaultdict(list)
    for q in qs:
        vs = [pid_to_vertex[int(p)] for p in sorted(q.gold_pids)]
        got = bfs_to(vs[0], vs[1:], args.max_dist)
        # -1 = not reached within max_dist (different component, or simply far)
        far = -1 if len(got) < len(vs) - 1 else max(got.values())
        dists[far] += 1
        worst_by_hops[q.n_hops].append(far)

    print("   max distance from the first gold passage to the others:")
    for k in sorted(dists, key=lambda x: (x < 0, x)):
        label = f"UNREACHABLE within {args.max_dist}" if k < 0 else f"distance {k}"
        print(f"     {label:<36} {dists[k]:4d}  ({dists[k]/len(qs):.1%})")
    reach4 = sum(v for k, v in dists.items() if 0 <= k <= 4)
    print(f"   within distance 4 (one shared entity, or one intermediate passage): "
          f"{reach4}/{len(qs)} ({reach4/len(qs):.1%})")
    print("   by question depth:")
    for h in sorted(worst_by_hops):
        v = np.array(worst_by_hops[h])
        ok = int((v >= 0).sum())
        print(f"     {h}hop  n={len(v):4d}  reachable {ok/len(v):.1%}  "
              f"median distance {int(np.median(v[v >= 0])) if ok else '-'}")

    print("""
how to read this:
  gold passages mostly UNREACHABLE or far apart -> the extraction did not build
      the bridges the questions need. No edge weighting can fix that, and the
      ceiling is a property of the KG, not of the retriever.
  gold passages mostly at distance 2 -> the graph supports the questions, and a
      retrieval shortfall is the seeding or the diffusion, not the KG.""")


if __name__ == "__main__":
    main()
