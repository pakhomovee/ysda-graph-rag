"""Is the authors' prebuilt KG scorable against OUR corpus? Check before running it.

``benchmark_runner`` writes its retrieval dump as indices into ``docs`` — the
list it builds from ``data/<dataset>_corpus.json``, i.e. OUR corpus — by looking
up each retrieved chunk's *content string* in that list. The chunks in a
downloaded bundle came from whatever corpus the authors indexed. Nothing forces
the two to agree, and every disagreement is silent in the same way:

    their chunk not in our corpus  -> written as -1, scores as a miss
    our gold passage not in their  -> unreachable, caps recall below 1

Both look exactly like "the method underperforms". A run that is really a corpus
mismatch would be indistinguishable from a finding, which is why this is a gate
rather than a postmortem — it needs no GPU, no LLM and no retrieval.

    python scripts/check_qafd_bundle.py musique \\
        third_party/QAFD-RAG/kg/multihop/<llm>_<emb>_<dataset>

Runs in the QAFD env (pandas + pyarrow, for the parquet stores). Reads gold from
``out/qafd_oracle_<dataset>.json`` rather than importing ``mbuzai.dataio``, so it
does not need the 3.10+ env: that file is already {question: [corpus_idx, ...]}
and is written by ``make_qafd_oracle.py`` in the probe's own preflight.
"""

import argparse
import json
import os
import sys


def _norm(s):
    """Whitespace-insensitive form, for telling 'different corpus' from
    'same corpus, different serialisation'. Only ever used to diagnose a
    failure — the runner itself matches byte for byte."""
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("kg_dir", help="kg/multihop/<llm>_<emb>_<dataset>")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--oracle", default=None,
                    help="default out/qafd_oracle_<dataset>.json")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kg = args.kg_dir
    if not os.path.isdir(kg):
        sys.exit(f"no such directory: {kg}")

    print(f"==> bundle {kg}")
    for f in ("graph.pickle", "vdb_chunk.parquet", "vdb_entity.parquet", "vdb_fact.parquet"):
        p = os.path.join(kg, f)
        print(f"    {'ok ' if os.path.exists(p) else 'MISSING'} {f}"
              + (f"  ({os.path.getsize(p) / 1e6:.1f} MB)" if os.path.exists(p) else ""))
    if not os.path.exists(os.path.join(kg, "graph.pickle")):
        # working_dir only accepts a bundle directory that has this file; without
        # it the config silently falls back to outputs/ and the run quietly uses
        # a different KG than the one under test.
        sys.exit("\nFATAL: no graph.pickle — config.working_dir will NOT select this "
                 "directory,\n  and the run will fall back to outputs/ without saying so.")

    import pandas as pd

    print("\n==> stores")
    dims = {}
    counts = {}
    for ns in ("chunk", "entity", "fact"):
        p = os.path.join(kg, f"vdb_{ns}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        counts[ns] = len(df)
        dims[ns] = len(df["embedding"].iloc[0]) if len(df) else 0
        print(f"    {ns:<7} {len(df):>7} rows, dim {dims[ns]}")
        if ns == "chunk":
            their_chunks = df["content"].tolist()
    # The encoder is not recorded in the bundle; the dimension is the only
    # evidence of it, and EMB has to match or the query vectors live in a
    # different space than the index.
    _by_dim = {4096: "nvidia-nv-embed-v2 / gritlm (7B)", 1024: "jina-v3",
               768: "mpnet", 3072: "openai-large", 1536: "openai-small"}
    if dims:
        d = dims.get("chunk") or next(iter(dims.values()))
        print(f"    -> {d}-dim: {_by_dim.get(d, 'unrecognised')}")

    print("\n==> graph")
    try:
        import pickle
        with open(os.path.join(kg, "graph.pickle"), "rb") as f:
            g = pickle.load(f)
        print(f"    {g.vcount()} nodes, {g.ecount()} edges")
    except Exception as exc:                      # igraph missing, or a pickle
        print(f"    (could not read: {exc})")     # from another igraph version

    # ---------------------------------------------------------------
    # The gate: does their chunk text match ours byte for byte?
    # ---------------------------------------------------------------
    corpus_path = os.path.join(args.data_dir, f"{args.dataset}_corpus.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    our_docs = [f"{d['title']}\n{d['text']}" for d in corpus]
    our_set = set(our_docs)

    theirs = set(their_chunks)
    both = theirs & our_set
    print(f"\n==> corpus agreement ({corpus_path})")
    print(f"    ours   {len(our_set):>7} passages")
    print(f"    theirs {len(theirs):>7} chunks")
    print(f"    exact match      {len(both):>7}  "
          f"({100 * len(both) / max(1, len(theirs)):.1f}% of theirs, "
          f"{100 * len(both) / max(1, len(our_set)):.1f}% of ours)")

    if len(both) < len(theirs):
        # Distinguish a different corpus from the same one serialised differently:
        # the first is fatal, the second is a normalisation we could apply.
        n_both = {_norm(t) for t in theirs} & {_norm(d) for d in our_set}
        print(f"    match ignoring whitespace {len(n_both):>7}"
              + ("   <- serialisation differs, not the corpus"
                 if len(n_both) > len(both) else ""))

    # ---------------------------------------------------------------
    # The ceiling: can their graph even reach our gold?
    # ---------------------------------------------------------------
    oracle_path = args.oracle or os.path.join(root, "out", f"qafd_oracle_{args.dataset}.json")
    if not os.path.exists(oracle_path):
        print(f"\n(no {oracle_path} — run scripts/make_qafd_oracle.py {args.dataset} "
              "for the gold-reachability check)")
        return

    gold_by_q = json.load(open(oracle_path))
    reachable = [i for i, d in enumerate(our_docs) if d in theirs]
    reachable = set(reachable)
    full, partial, none_ = 0, 0, 0
    covered_frac = 0.0
    for _q, pids in gold_by_q.items():
        hit = sum(1 for p in pids if p in reachable)
        covered_frac += hit / len(pids) if pids else 0
        if not pids or hit == len(pids):
            full += 1
        elif hit:
            partial += 1
        else:
            none_ += 1
    n = len(gold_by_q)
    print(f"\n==> gold reachability ({n} questions, from {os.path.basename(oracle_path)})")
    print(f"    all gold present  {full:>5}  ({100 * full / n:.1f}%)")
    print(f"    some              {partial:>5}")
    print(f"    none              {none_:>5}")
    print(f"    mean gold covered {covered_frac / n:.4f}   <- the recall@inf ceiling")

    print()
    if covered_frac / n > 0.999:
        print("VERDICT: the bundle indexes our corpus. Recall is comparable to "
              "every other row in RESULTS.md.")
    elif covered_frac / n > 0.5:
        print("VERDICT: partial overlap. Recall on this KG is NOT comparable to our "
              "other runs —\n  the ceiling above is the number it would be measured "
              "against, not 1.0.")
    else:
        print("VERDICT: different corpus. Retrieval against this KG cannot be scored "
              "with score_qafd.py;\n  the dump would be mostly -1 and would read as a "
              "catastrophic regression.")


if __name__ == "__main__":
    main()
