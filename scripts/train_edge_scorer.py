"""Train a light w(u, v, q) for QAFD's diffusion, and see whether it helps.

The oracle arms measured one policy — multiply by a constant when both endpoints
are gold — and found it null to negative. That is weaker evidence than it looked:
information dominance is not performance dominance, the oracle never touched
edges *leading toward* gold, and recall is not monotone in weight quality. So
this trains the thing directly instead of arguing about its ceiling.

Positives are edges that lie on a shortest path between two of a question's gold
passages: `d(s,u) + 1 + d(v,t) == d(s,t)`. Those are the bridges the question
actually needs traversed, which is precisely the case the both-endpoints-gold
oracle could not express.

Negatives come in two kinds, and both matter:
  * incident — edges touching a node on a gold path but leading off it. Without
    these the model learns "is this near gold", which is a question-independent
    prior and shows up immediately in the shuffled control.
  * random — edges from elsewhere in the graph.

Split is by QUESTION. The model is evaluated on held-out questions, and
--shuffle-control permutes the gold sets to check that nothing query-independent
is being learned.

    python scripts/train_edge_scorer.py musique --shuffle-control   # gate: AUC ~ 0.5
    python scripts/train_edge_scorer.py musique

--jobs parallelises the label pass, which is the expensive half: it runs BFS from
every gold passage and tests the path condition against all ~850k edges, per
question. Training itself is numpy matmuls, so it is already BLAS-threaded --
control that with OMP_NUM_THREADS rather than --jobs.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):
    sys.exit(f"needs the mbuzai env (3.10+), got {sys.version.split()[0]}")

from mbuzai import dataio, metrics  # noqa: E402
from mbuzai.edgemodel import EdgeScorer, build_features  # noqa: E402

OUT = ROOT / "out"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"


def csr(edges, n_nodes):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    indices = dst[order].astype(np.int64)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n_nodes), out=indptr[1:])
    return indptr, indices


def bfs_levels(indptr, indices, start, n_nodes, max_dist):
    """Distance from `start` to every node it reaches within max_dist, -1 else."""
    dist = np.full(n_nodes, -1, dtype=np.int32)
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    for d in range(1, max_dist + 1):
        if frontier.size == 0:
            break
        starts, ends = indptr[frontier], indptr[frontier + 1]
        counts = ends - starts
        if counts.sum() == 0:
            break
        offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        nbrs = indices[np.repeat(starts, counts) + offs]
        nbrs = np.unique(nbrs[dist[nbrs] < 0])
        if nbrs.size == 0:
            break
        dist[nbrs] = d
        frontier = nbrs
    return dist


# Read-only state for the label pass, inherited copy-on-write by forked workers.
_G = {}


def _label_question(qi):
    """Edges on a shortest path between this question's gold passages, plus
    negatives. Returns (edge_ids, labels, n_positive) or None.

    Fully vectorised over the edge array. The obvious loop -- iterate the edges,
    test the path condition, look the id up in a dict -- runs 850k times per gold
    pair and dominated everything else in the script.
    """
    indptr, indices = _G["indptr"], _G["indices"]
    edges, n_nodes = _G["edges"], _G["n_nodes"]
    u_all, v_all = edges[:, 0], edges[:, 1]

    vs = [_G["pid_to_vertex"][int(p)] for p in _G["gold_sets"][qi]]
    pos = np.zeros(len(edges), dtype=bool)
    for a in range(len(vs)):
        da = bfs_levels(indptr, indices, vs[a], n_nodes, _G["max_dist"])
        for b in range(a + 1, len(vs)):
            t = vs[b]
            if da[t] < 0:
                continue
            db = bfs_levels(indptr, indices, t, n_nodes, _G["max_dist"])
            # (u,v) is on a shortest path vs[a] -> t iff traversing it advances
            # the distance exactly, in either orientation.
            du, dv = da[u_all], da[v_all]
            bu, bv = db[u_all], db[v_all]
            pos |= (du >= 0) & (bv >= 0) & (du + 1 + bv == da[t])
            pos |= (dv >= 0) & (bu >= 0) & (dv + 1 + bu == da[t])
    pos_ids = np.flatnonzero(pos)
    if pos_ids.size == 0:
        return None

    rng = np.random.default_rng(_G["seed"] * 1000003 + qi)
    n_pos_all = pos_ids.size
    if pos_ids.size > _G["max_pos"]:
        pos_ids = rng.choice(pos_ids, _G["max_pos"], replace=False)

    # hard negatives: incident to a gold-path node but not themselves on a path
    touched = np.unique(edges[pos_ids].ravel())
    inc = np.flatnonzero((np.isin(u_all, touched) | np.isin(v_all, touched)) & ~pos)
    n_hard = min(len(inc), _G["neg_per_pos"] * len(pos_ids) // 2)
    hard = rng.choice(inc, n_hard, replace=False) if n_hard else np.empty(0, np.int64)
    n_rand = _G["neg_per_pos"] * len(pos_ids) - n_hard
    rand = rng.choice(len(edges), max(n_rand, 0), replace=False)
    rand = rand[~pos[rand]]

    sel = np.concatenate([pos_ids, hard, rand]).astype(np.int64)
    lab = np.concatenate([np.ones(len(pos_ids)),
                          np.zeros(len(hard) + len(rand))])
    return sel, lab, n_pos_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--nodes", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--train-questions", type=int, default=700)
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--max-pos", type=int, default=400,
                    help="cap on-path edges per question; bounds the feature "
                         "block, which is |rows| x (4d+6) floats")
    ap.add_argument("--jobs", type=int, default=8,
                    help="processes for the label pass. Training itself is "
                         "BLAS-threaded; set OMP_NUM_THREADS for that")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shuffle-control", action="store_true")
    args = ap.parse_args()

    npz = args.nodes or OUT / f"qafd_nodes_{args.dataset}.npz"
    if not npz.exists():
        sys.exit(f"missing {npz}\n  rebuild it (QAFD env): bash scripts/export_qafd_nodes.sh")
    z = np.load(npz, allow_pickle=False)
    if "passage_emb" not in z.files:
        sys.exit(f"{npz} predates passage embeddings — re-run scripts/export_qafd_nodes.sh")

    edges = z["edges"]
    ew = z["edge_weight"].astype(np.float32)
    ent_v, pas_v = z["entity_vertex"], z["passage_vertex"]
    n_nodes = int(max(ent_v.max(), pas_v.max())) + 1

    # One embedding table indexed by graph vertex, covering both node kinds.
    d = z["entity_emb"].shape[1]
    H = np.zeros((n_nodes, d), dtype=np.float32)
    for vs, em in ((ent_v, z["entity_emb"]), (pas_v, z["passage_emb"])):
        ok = vs >= 0
        H[vs[ok]] = np.asarray(em, dtype=np.float32)[ok]
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12

    deg = np.bincount(np.concatenate([edges[:, 0], edges[:, 1]]),
                      minlength=n_nodes).astype(np.float32)
    indptr, indices = csr(edges, n_nodes)
    pid_to_vertex = {int(p): int(v) for p, v in zip(z["passage_pid"], pas_v)
                     if p >= 0 and v >= 0}

    ds = dataio.load(args.dataset)
    qs = [q for q in ds.queries if len(q.gold_pids) >= 2
          and all(int(p) in pid_to_vertex for p in q.gold_pids)]
    print(f"graph {n_nodes} nodes / {len(edges)} edges; {len(qs)} usable questions")

    rng = np.random.default_rng(args.seed)
    gold_sets = [sorted(q.gold_pids) for q in qs]
    if args.shuffle_control:
        gold_sets = [gold_sets[i] for i in rng.permutation(len(gold_sets))]
        print("SHUFFLE CONTROL: gold sets permuted across questions")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device=args.device)
    Q = np.asarray(model.encode([q.question for q in qs], normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False),
                   dtype=np.float32)

    # ---- label the edges -------------------------------------------------
    # Published to forked workers rather than pickled: the arrays are large and
    # every worker only reads them.
    _G.update(indptr=indptr, indices=indices, edges=edges, n_nodes=n_nodes,
              pid_to_vertex=pid_to_vertex, gold_sets=gold_sets,
              max_dist=args.max_dist, neg_per_pos=args.neg_per_pos,
              max_pos=args.max_pos, seed=args.seed)

    t0 = time.time()
    if args.jobs > 1:
        import concurrent.futures as cf, multiprocessing as mp
        with cf.ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("fork")) as ex:
            labelled = list(ex.map(_label_question, range(len(qs)), chunksize=8))
    else:
        labelled = [_label_question(i) for i in range(len(qs))]

    rows_e, rows_q, rows_y, on_path_total, skipped = [], [], [], 0, 0
    for qi, out in enumerate(labelled):
        if out is None:
            skipped += 1
            continue
        sel, lab, n_pos = out
        on_path_total += n_pos
        rows_e.append(sel); rows_q.append(np.full(len(sel), qi)); rows_y.append(lab)

    E = np.concatenate(rows_e); QI = np.concatenate(rows_q)
    Y = np.concatenate(rows_y).astype(np.float32)
    kept = len(qs) - skipped
    print(f"labels: {len(Y)} edges over {kept} questions ({int(Y.sum())} positive, "
          f"{on_path_total / max(kept, 1):.1f} on-path edges per question; "
          f"{skipped} questions had no connected gold pair) in {time.time()-t0:.0f}s")

    order = rng.permutation(len(qs))
    tr_q = set(order[:args.train_questions].tolist())
    tr = np.isin(QI, list(tr_q))
    print(f"split: {len(tr_q)} train questions, {len(qs) - len(tr_q)} held out; "
          f"{tr.sum()} / {(~tr).sum()} edge rows")

    def feats(mask):
        e = E[mask]
        u, v = edges[e, 0], edges[e, 1]
        return build_features(H[u], H[v], Q[QI[mask]], deg[u], deg[v], ew[e])

    Xtr, ytr = feats(tr), Y[tr]
    Xte, yte = feats(~tr), Y[~tr]

    net = EdgeScorer(d, hidden=args.hidden, seed=args.seed)
    state, best = {}, (-np.inf, None)
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(idx), args.batch):
            net.step(Xtr[idx[i:i + args.batch]], ytr[idx[i:i + args.batch]], state, args.lr)
        if ep % 10 == 0 or ep == args.epochs:
            auc = roc_auc(net.score(Xte), yte)
            if auc > best[0]:
                best = (auc, [p.copy() for p in net.params()])
            print(f"  epoch {ep:3d}  held-out AUC {auc:.4f}")
    for p, b in zip(net.params(), best[1]):
        p[...] = b

    dest = args.out or OUT / (f"edge_scorer_{args.dataset}"
                              + ("_shuffled" if args.shuffle_control else "") + ".npz")
    net.save(dest, held_out_auc=np.float32(best[0]))
    print(f"\nheld-out AUC {best[0]:.4f}   wrote {dest}")
    print("""
An AUC near 0.5 means the model cannot tell path edges from their neighbours, and
the in-pipeline arm will be a null by construction. Well above 0.5 means it has
learned something; whether that converts to retrieval is what the arm measures.
Run --shuffle-control: it must collapse to ~0.5.""")


def roc_auc(scores, labels):
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = (starts + (counts - 1) / 2.0 + 1)[inv]
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


if __name__ == "__main__":
    main()
