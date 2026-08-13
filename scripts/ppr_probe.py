"""Does Personalized PageRank respond to query-aware edge weights at all?

HippoRAG-style retrieval runs PPR over an OpenIE graph and calls it with
``weights="weight"`` -- a STATIC graph attribute. The query enters only through
``reset``, the personalisation vector, i.e. seeding. So there is no query-aware
edge weight in that pipeline to ablate; introducing one is a new intervention.

Before training anything to produce such a weight, this asks the cheap question
first: hand the edge the answer and see whether PPR's output moves at all.

    oracle<mult>   multiply every edge incident to a gold passage node by <mult>.
                   Upper-bounds every possible scorer at this site, learned
                   included. Flat at 1000x => nothing to train, stop here.
    hybrid/product/exp   the bounded and unbounded query-aware forms, for when
                   the oracle does respond and the question becomes whether a
                   heuristic already captures it.

Two things make this a different question from a flow-diffusion push loop, and
both are swept rather than assumed:

  DAMPING   PPR reaches a stationary distribution, so per-hop weight contrast
            compounds along paths instead of being truncated. But at the 0.5
            both shipped implementations use, the walk teleports half the time
            and the expected path is ~2 hops -- barely any path to compound
            over. If edge weights matter here at all, they should matter more at
            0.85-0.95. Sweeping this IS the mechanism test.
  SCHEME    PPR normalises transitions as w_ij / sum_k w_ik, so only the SPREAD
            of weights within a neighbourhood can steer, exactly as in a push
            loop. routing_cv below measures that spread directly, so a null arm
            can be told apart from an inert one.

Seeding is held FIXED across every arm: the screen is about edge weights, and a
seeding difference would confound it.

Self-contained on purpose -- igraph, numpy, pandas and mbuzai.dataio only. No
retrieval module from any third_party system is imported, so nothing here
inherits another pipeline's behaviour.

    # the screen: does the site respond, and does damping change the answer
    python scripts/ppr_probe.py musique --kg_dir <kg> --query_emb out/qemb.npz \\
        --arms none oracle10 oracle100 oracle1000 --damping 0.5 0.85 0.95

    # then score the dumps it writes, with the usual paired bootstrap
    python scripts/score_qafd.py musique --runs out/ppr_musique_*.json \\
        --baseline none-d0.5

Runs in the QAFD env (igraph + pandas are there); the scorer runs in the mbuzai
env, as everywhere else in this repo.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Weight functions. Pure numpy over edge arrays, so --selftest can exercise them
# without igraph, a graph, or a corpus.
# ---------------------------------------------------------------------------

def edge_weights(w_base, s_u, s_v, scheme, hybrid_a=1.0, hybrid_b=0.25, beta=8.0):
    """Query-aware edge weights, vectorised over the whole edge list.

    PPR requires non-negative weights, and cosine similarities are signed, so the
    node scores are clipped at 0 first. That is a real modelling choice: a node
    anti-correlated with the query and one merely unrelated both become 0.
    """
    su = np.maximum(s_u, 0.0)
    sv = np.maximum(s_v, 0.0)
    if scheme == "none":
        return w_base.copy()
    if scheme == "hybrid":
        return w_base * (hybrid_a + hybrid_b * (su + sv) / 2.0)
    if scheme == "product":
        return w_base * su * sv
    if scheme == "exp":
        return w_base * np.exp(beta * (su + sv))
    raise ValueError(f"unknown scheme: {scheme}")


def apply_oracle(w, eu, ev, gold_vertices, mult, eligible=None):
    """Multiply every eligible edge incident to a gold vertex.

    The PASSAGE form leaks, badly, and worse here than in a push loop: a node's
    stationary probability under PPR is essentially the weighted flow into it, so
    scaling every edge touching a gold passage lifts that passage's own score
    almost mechanically. It ranks nodes you marked rather than steering traversal
    toward them, which is why it saturates.

    `eligible` is what makes the ENTITY form mean something: restricted to edges
    that touch no passage node at all, the oracle can only raise the probability
    of REACHING the gold neighbourhood, and ordinary entity->passage edges then
    have to carry mass to the target at their unmodified weight. That is steering
    with the leak removed, and it is the arm the decision actually rests on.
    """
    if mult == 1.0 or len(gold_vertices) == 0:
        return w
    gold = np.zeros(int(max(eu.max(), ev.max())) + 1, dtype=bool)
    gold[np.fromiter(gold_vertices, dtype=np.int64)] = True
    hit = gold[eu] | gold[ev]
    if eligible is not None:
        hit &= eligible
    out = w.copy()
    out[hit] *= mult
    return out


def routing_cv(w, eu, ev, n, node_mass=None):
    """Mean dispersion of the transition distributions PPR actually uses.

    PPR routes as w_ij / sum_k w_ik, so the absolute level of a weight divides
    out and only its spread within a neighbourhood can move anything. This is the
    load-bearing diagnostic: a null recall delta whose routing_cv matches the
    baseline means the weight never reached the routing distribution (the arm is
    vacuous), while a null at a raised routing_cv is evidence the site is inert.

    Weighted by node_mass when given, since dispersion at a node carrying no
    probability cannot affect the ranking.
    """
    cnt = np.bincount(eu, minlength=n) + np.bincount(ev, minlength=n)
    tot = np.bincount(eu, weights=w, minlength=n) + np.bincount(ev, weights=w, minlength=n)
    sq = np.bincount(eu, weights=w * w, minlength=n) + np.bincount(ev, weights=w * w, minlength=n)
    ok = cnt > 1
    mean = np.zeros(n)
    mean[ok] = tot[ok] / cnt[ok]
    var = np.zeros(n)
    var[ok] = np.maximum(sq[ok] / cnt[ok] - mean[ok] ** 2, 0.0)
    cv = np.zeros(n)
    nz = ok & (mean > 0)
    cv[nz] = np.sqrt(var[nz]) / mean[nz]
    if node_mass is None:
        return float(cv[nz].mean()) if nz.any() else 0.0
    wt = np.asarray(node_mass, dtype=np.float64)
    wt = np.where(nz, wt, 0.0)
    return float((cv * wt).sum() / wt.sum()) if wt.sum() > 0 else 0.0


def arm_spec(arm):
    """Arm name -> (scheme, oracle_mult, beta). One definition, used for both the
    run and the filename, so the two cannot drift."""
    if arm == "none":
        return "none", 1.0, 0.0
    if arm.startswith("oracle"):
        return "none", float(arm[len("oracle"):]), 0.0
    if arm.startswith("hybrid"):
        return "hybrid", 1.0, 0.0
    if arm == "product":
        return "product", 1.0, 0.0
    if arm.startswith("exp"):
        return "exp", 1.0, float(arm[len("exp"):])
    raise ValueError(f"unknown arm: {arm}")


# ---------------------------------------------------------------------------

def selftest():
    """Exercise the weight math without igraph, a graph, or a corpus."""
    rng = np.random.default_rng(0)
    n, m = 50, 300
    eu = rng.integers(0, n, m)
    ev = rng.integers(0, n, m)
    w_base = rng.random(m) + 0.5
    s = rng.random(n) * 2 - 1          # signed, like real cosines
    su, sv = s[eu], s[ev]

    assert np.array_equal(edge_weights(w_base, su, sv, "none"), w_base)
    # hybrid with b=0 is the identity, so any difference is the query term alone
    assert np.allclose(edge_weights(w_base, su, sv, "hybrid", 1.0, 0.0), w_base)
    hy = edge_weights(w_base, su, sv, "hybrid", 1.0, 0.25)
    assert (hy >= w_base - 1e-12).all(), "hybrid with a=1,b>0 cannot reduce a weight"
    assert (hy <= w_base * 1.25 + 1e-12).all(), "hybrid must stay within [1, 1+b]"
    assert (edge_weights(w_base, su, sv, "product") >= 0).all()
    assert (edge_weights(w_base, su, sv, "exp", beta=8.0) > 0).all()

    # oracle at 1x must be bit-identical, or the arm is not a no-op baseline
    assert np.array_equal(apply_oracle(w_base, eu, ev, {3, 7}, 1.0), w_base)
    boosted = apply_oracle(w_base, eu, ev, {3, 7}, 100.0)
    touched = (eu == 3) | (ev == 3) | (eu == 7) | (ev == 7)
    assert np.allclose(boosted[touched], w_base[touched] * 100)
    assert np.array_equal(boosted[~touched], w_base[~touched])

    # uniform weights have zero dispersion; that is the floor routing_cv reports
    assert routing_cv(np.ones(m), eu, ev, n) == 0.0
    assert routing_cv(w_base, eu, ev, n) > 0.0
    # Scale invariance is the whole reason routing_cv is the right diagnostic:
    # PPR divides the level out, so a diagnostic that moved with it would report
    # steering where there is none.
    assert abs(routing_cv(w_base * 1000, eu, ev, n) - routing_cv(w_base, eu, ev, n)) < 1e-9
    # The eligibility mask must confine the boost, or the entity arm silently
    # becomes the passage arm and measures the leak it exists to remove.
    elig = (eu != 0) & (ev != 0)
    gated = apply_oracle(w_base, eu, ev, {3, 7}, 100.0, elig)
    assert np.array_equal(gated[~elig], w_base[~elig]), "ineligible edges must be untouched"
    both = ((eu == 3) | (ev == 3) | (eu == 7) | (ev == 7)) & elig
    assert np.allclose(gated[both], w_base[both] * 100)

    # An oracle that fires must raise dispersion, or the arm cannot steer at all.
    assert routing_cv(apply_oracle(w_base, eu, ev, {3, 7}, 100.0), eu, ev, n) > \
        routing_cv(w_base, eu, ev, n)

    for a, expected in [("none", ("none", 1.0, 0.0)), ("oracle100", ("none", 100.0, 0.0)),
                        ("hybrid", ("hybrid", 1.0, 0.0)), ("exp8", ("exp", 1.0, 8.0))]:
        assert arm_spec(a) == expected, (a, arm_spec(a))
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--kg_dir", help="directory with graph.pickle and the vdb_*.parquet stores")
    ap.add_argument("--query_emb", help="npz from --cache_query_emb (questions/fact/passage)")
    ap.add_argument("--query_emb_kind", default="passage", choices=["passage", "fact"],
                    help="which cached query vector to score nodes with")
    ap.add_argument("--arms", nargs="+", default=["none", "oracle10", "oracle100", "oracle1000"])
    ap.add_argument("--damping", nargs="+", type=float, default=[0.5],
                    help="0.5 is what both shipped implementations use (~2-hop walk)")
    ap.add_argument("--hybrid_a", type=float, default=1.0)
    ap.add_argument("--hybrid_b", type=float, default=0.25)
    # passages: boost edges incident to the gold passage node. Leaks -- the node
    #           is also the ranking target. Saturates, and bounds nothing useful.
    # entities: boost edges among the gold passage's own entities, excluding every
    #           edge that touches ANY passage node. Mass still has to travel to the
    #           target over unmodified entity->passage edges, so this measures
    #           steering. The gold entities are read off the graph (entity
    #           neighbours of the gold passage), not from an OpenIE sidecar.
    ap.add_argument("--oracle_nodes", default="passages", choices=["passages", "entities"])
    ap.add_argument("--seed_top_k", type=int, default=5, help="entity seeds kept per query")
    ap.add_argument("--passage_node_weight", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--num_queries", type=int, default=0)
    ap.add_argument("--out_dir", default=str(ROOT / "out"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    for req in ("dataset", "kg_dir", "query_emb"):
        if not getattr(args, req):
            ap.error(f"--{req} is required (or pass --selftest)")

    import pickle
    import pandas as pd
    from mbuzai import dataio

    print(f"==> corpus + gold ({args.dataset})")
    ds = dataio.load(args.dataset)
    docs = ds.docs
    doc_to_idx = {d: i for i, d in enumerate(docs)}
    queries = ds.queries[: args.num_queries] if args.num_queries else ds.queries
    print(f"    {len(docs)} passages, {len(queries)} questions")

    print(f"==> graph {args.kg_dir}")
    with open(os.path.join(args.kg_dir, "graph.pickle"), "rb") as f:
        graph = pickle.load(f)
    n = graph.vcount()
    name_to_idx = {v["name"]: i for i, v in enumerate(graph.vs)}
    eu, ev = (np.asarray(a, dtype=np.int64) for a in zip(*graph.get_edgelist()))
    if "weight" in graph.es.attributes():
        w_base = np.asarray(graph.es["weight"], dtype=np.float64)
    else:
        print("    no edge 'weight' attribute — falling back to uniform weights")
        w_base = np.ones(graph.ecount(), dtype=np.float64)
    print(f"    {n} nodes, {len(eu)} edges")

    def store(ns):
        # The bundle nests these under <ns>_embeddings/; a self-built index may not.
        for cand in (os.path.join(args.kg_dir, f"{ns}_embeddings", f"vdb_{ns}.parquet"),
                     os.path.join(args.kg_dir, f"vdb_{ns}.parquet")):
            if os.path.exists(cand):
                return pd.read_parquet(cand)
        raise SystemExit(f"no vdb_{ns}.parquet under {args.kg_dir}")

    print("==> stores")
    node_sim = np.zeros(n, dtype=np.float64)   # filled per query, reused
    ent_df, chunk_df = store("entity"), store("chunk")
    ent_idx = np.array([name_to_idx[h] for h in ent_df["hash_id"]], dtype=np.int64)
    pas_idx = np.array([name_to_idx[h] for h in chunk_df["hash_id"]], dtype=np.int64)
    E_ent = np.vstack(ent_df["embedding"].to_numpy()).astype(np.float32)
    E_pas = np.vstack(chunk_df["embedding"].to_numpy()).astype(np.float32)
    E_ent /= np.maximum(np.linalg.norm(E_ent, axis=1, keepdims=True), 1e-12)
    E_pas /= np.maximum(np.linalg.norm(E_pas, axis=1, keepdims=True), 1e-12)
    # Passage node -> our corpus index. Exact string identity, the same mapping
    # the rest of the repo scores through; -1 means the KG indexed a different
    # corpus and every number below would be meaningless.
    pas_pid = np.array([doc_to_idx.get(c, -1) for c in chunk_df["content"]], dtype=np.int64)
    unmapped = int((pas_pid < 0).sum())
    if unmapped:
        print(f"    WARNING {unmapped}/{len(pas_pid)} passage nodes are not in our corpus")
    print(f"    {len(ent_idx)} entities, {len(pas_idx)} passages, dim {E_ent.shape[1]}")

    qz = np.load(args.query_emb, allow_pickle=False)
    qmap = {q: i for i, q in enumerate(qz["questions"])}
    Q = np.asarray(qz[args.query_emb_kind], dtype=np.float32)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    missing = [q for q in queries if q.question not in qmap]
    if missing:
        raise SystemExit(f"\nFATAL: {len(missing)} questions absent from {args.query_emb}, "
                         f"e.g. {missing[0].question[:60]!r}")

    pid_to_vertex = {int(p): int(v) for p, v in zip(pas_pid, pas_idx) if p >= 0}
    is_passage = np.zeros(n, dtype=bool)
    is_passage[pas_idx] = True
    is_entity = np.zeros(n, dtype=bool)
    is_entity[ent_idx] = True
    # Edges touching no passage node. The entity oracle is confined to these so it
    # cannot pump mass straight into a ranking target; see apply_oracle.
    entity_only_edges = ~(is_passage[eu] | is_passage[ev])
    if args.oracle_nodes == "entities":
        print(f"    entity-only oracle: {int(entity_only_edges.sum())} of {len(eu)} "
              f"edges are eligible (touch no passage node)")
    os.makedirs(args.out_dir, exist_ok=True)

    for damping in args.damping:
        for arm in args.arms:
            scheme, mult, beta = arm_spec(arm)
            _scope = "ent" if (args.oracle_nodes == "entities" and mult != 1.0) else ""
            tag = f"{arm}{_scope}-d{damping:g}"
            t0 = time.time()
            dump, cvs = {}, []
            for q in queries:
                qv = Q[qmap[q.question]]
                s_ent = E_ent @ qv
                s_pas = E_pas @ qv
                node_sim[:] = 0.0
                node_sim[ent_idx] = s_ent
                node_sim[pas_idx] = s_pas

                # --- reset vector: fixed across arms, by construction ---
                reset = np.zeros(n, dtype=np.float64)
                top = np.argpartition(-s_ent, min(args.seed_top_k, len(s_ent) - 1))[:args.seed_top_k]
                reset[ent_idx[top]] = np.maximum(s_ent[top], 0.0)
                pv = np.maximum(s_pas, 0.0)
                if pv.max() > 0:
                    reset[pas_idx] += args.passage_node_weight * pv / pv.max()
                if reset.sum() <= 0:
                    reset[:] = 1.0

                # --- edge weights ---
                w = edge_weights(w_base, node_sim[eu], node_sim[ev], scheme,
                                 args.hybrid_a, args.hybrid_b, beta)
                if mult != 1.0:
                    gold_v, eligible = set(), None
                    for pid in q.gold_pids:
                        v = pid_to_vertex.get(int(pid))
                        if v is None:
                            continue
                        if args.oracle_nodes == "passages":
                            gold_v.add(v)
                        else:
                            # The gold passage's own entities, straight off the
                            # graph: passage->entity edges are how the extraction
                            # is recorded, so no OpenIE sidecar is needed.
                            gold_v.update(nb for nb in graph.neighbors(v) if is_entity[nb])
                    if args.oracle_nodes == "entities":
                        eligible = entity_only_edges
                    w = apply_oracle(w, eu, ev, gold_v, mult, eligible)
                # prpack rejects non-positive weights; keep the graph connected
                w = np.maximum(w, 1e-12)

                scores = np.asarray(graph.personalized_pagerank(
                    vertices=range(n), damping=damping, directed=False,
                    weights=w.tolist(), reset=reset.tolist(), implementation="prpack"))
                cvs.append(routing_cv(w, eu, ev, n, node_mass=scores))

                doc_scores = scores[pas_idx]
                order = np.argsort(-doc_scores)[: args.topk]
                dump[q.qid] = [int(pas_pid[i]) for i in order]

            path = os.path.join(args.out_dir, f"ppr_{args.dataset}_{tag}.json")
            with open(path, "w") as f:
                json.dump(dump, f)
            stats = {"arm": tag, "scheme": scheme, "oracle_mult": mult, "beta": beta,
                     "damping": damping, "routing_cv": float(np.mean(cvs)),
                     "queries": len(dump), "seconds": round(time.time() - t0, 1)}
            with open(os.path.join(args.out_dir, f"pprstats_{args.dataset}_{tag}.json"), "w") as f:
                json.dump(stats, f, indent=2)
            print(f"    {tag:<22} routing_cv={stats['routing_cv']:.4f}  "
                  f"{stats['seconds']}s  -> {os.path.basename(path)}")

    print("\nRead routing_cv BEFORE any recall delta: an arm whose routing_cv matches\n"
          "the none arm never reached the transition distribution and proves nothing.\n"
          f"Score with:\n  python scripts/score_qafd.py {args.dataset} "
          f"--runs {args.out_dir}/ppr_{args.dataset}_*.json --baseline none-d{args.damping[0]:g}")


if __name__ == "__main__":
    main()
