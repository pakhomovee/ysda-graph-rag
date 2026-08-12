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

    python scripts/train_edge_scorer.py musique
    python scripts/train_edge_scorer.py musique --shuffle-control
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--nodes", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--train-questions", type=int, default=700)
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--neg-per-pos", type=int, default=3)
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
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(edges)}
    eid.update({(b, a): i for (a, b), i in list(eid.items())})

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
    rows_u, rows_v, rows_q, rows_y = [], [], [], []
    on_path_total, skipped = 0, 0
    for qi, pids in enumerate(gold_sets):
        vs = [pid_to_vertex[int(p)] for p in pids]
        pos = set()
        for a in range(len(vs)):
            da = bfs_levels(indptr, indices, vs[a], n_nodes, args.max_dist)
            for b in range(a + 1, len(vs)):
                t = vs[b]
                if da[t] < 0:
                    continue
                db = bfs_levels(indptr, indices, t, n_nodes, args.max_dist)
                # edge (u,v) is on a shortest path a->t iff it advances both ways
                for u, v in edges[(da[edges[:, 0]] >= 0) & (db[edges[:, 1]] >= 0)]:
                    u, v = int(u), int(v)
                    if da[u] + 1 + db[v] == da[t]:
                        pos.add(eid[(u, v)])
                for u, v in edges[(da[edges[:, 1]] >= 0) & (db[edges[:, 0]] >= 0)]:
                    u, v = int(u), int(v)
                    if da[v] + 1 + db[u] == da[t]:
                        pos.add(eid[(u, v)])
        if not pos:
            skipped += 1
            continue
        on_path_total += len(pos)
        pos = np.array(sorted(pos), dtype=np.int64)

        # hard negatives: edges touching a gold-path node but not on a path
        touched = np.unique(edges[pos].ravel())
        cand = np.unique(np.concatenate([
            indices[indptr[t]:indptr[t + 1]] for t in touched[:200]]) ) if len(touched) else np.array([], np.int64)
        inc = [eid[(int(a), int(b))] for a in touched[:200]
               for b in indices[indptr[a]:indptr[a + 1]] if eid.get((int(a), int(b))) is not None]
        inc = np.setdiff1d(np.array(inc, dtype=np.int64), pos)
        n_hard = min(len(inc), args.neg_per_pos * len(pos) // 2)
        hard = rng.choice(inc, n_hard, replace=False) if n_hard else np.array([], np.int64)
        n_rand = args.neg_per_pos * len(pos) - len(hard)
        rand = rng.choice(len(edges), max(n_rand, 0), replace=False)
        rand = np.setdiff1d(rand, pos)

        sel = np.concatenate([pos, hard, rand])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(hard) + len(rand))])
        rows_u.append(edges[sel, 0]); rows_v.append(edges[sel, 1])
        rows_q.append(np.full(len(sel), qi)); rows_y.append(lab)

    U = np.concatenate(rows_u); V = np.concatenate(rows_v)
    QI = np.concatenate(rows_q); Y = np.concatenate(rows_y).astype(np.float32)
    print(f"labels: {len(Y)} edges over {len(qs) - skipped} questions "
          f"({int(Y.sum())} positive, {on_path_total / max(len(qs) - skipped, 1):.1f} "
          f"on-path edges per question; {skipped} questions had no connected gold pair)")

    order = rng.permutation(len(qs))
    tr_q = set(order[:args.train_questions].tolist())
    tr = np.isin(QI, list(tr_q))
    print(f"split: {len(tr_q)} train questions, {len(qs) - len(tr_q)} held out; "
          f"{tr.sum()} / {(~tr).sum()} edge rows")

    def feats(mask):
        return build_features(H[U[mask]], H[V[mask]], Q[QI[mask]],
                              deg[U[mask]], deg[V[mask]],
                              ew[[eid[(int(a), int(b))] for a, b in
                                  zip(U[mask], V[mask])]])

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
