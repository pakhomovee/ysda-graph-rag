#!/usr/bin/env bash
# Bring up the patched HippoRAG 2 working copy.
#
# Same shape as setup_qafd.sh / setup_linearrag.sh: the submodule is pinned to the
# commit the patch was written against, referenced rather than vendored, and the
# patch application is idempotent.
#
#     bash scripts/setup_hipporag.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/HippoRAG"
PATCH="$ROOT/patches/hipporag_rerank.patch"

echo "==> submodule"
git -C "$ROOT" submodule update --init --recursive third_party/HippoRAG
echo "    at $(git -C "$SUB" rev-parse --short HEAD)"

echo "==> patch"
if git -C "$SUB" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "    already applied — nothing to do"
elif git -C "$SUB" apply --check "$PATCH" 2>/dev/null; then
    git -C "$SUB" apply "$PATCH"
    echo "    applied $(basename "$PATCH")"
else
    FILES=$(git -C "$SUB" apply --numstat "$PATCH" | awk '{print $3}')
    echo "    patch is out of date in the working tree; resetting and reapplying:"
    echo "$FILES" | sed 's/^/      /'
    # A patch that ADDS a file lists a path git has never tracked, and
    # 'git checkout --' on one is a fatal pathspec error. Reset what is tracked,
    # delete what is not.
    for f in $FILES; do
        if git -C "$SUB" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
            git -C "$SUB" checkout -- "$f"
        else
            rm -f "$SUB/$f"
        fi
    done
    if git -C "$SUB" apply "$PATCH"; then
        echo "    reapplied $(basename "$PATCH")"
    else
        echo "    FATAL: patch does not apply even to a clean tree." >&2
        git -C "$SUB" log -1 --format='      at %h %s' >&2
        exit 1
    fi
fi

echo "==> sanity"
# Behavioural, not marker-based: `git pull` updates patches/*.patch without
# reapplying it to the submodule working tree, so a fixed marker passes on a stale
# tree and the run then dies on argparse tens of minutes in. Ask the entry point
# what it actually accepts, exactly as run_qafd_probe.sh does.
_help=$( cd "$SUB" && python3 main.py --help 2>&1 ) || {
    echo "    FATAL: main.py --help failed:" >&2; echo "$_help" | tail -5 >&2; exit 1; }
for flag in --skip_qa --data_dir --dump --rerank_mode --rerank_candidate_k \
            --rerank_keep_k --export_facts --num_queries --query_shard; do
    case "$_help" in
        *"$flag"*) echo "    ok: $flag" ;;
        *) cat >&2 <<EOF
    FATAL: main.py does not accept $flag — the submodule has a stale patch.
      git pull updates patches/hipporag_rerank.patch; it does not reapply it.
      Fix:  bash scripts/setup_hipporag.sh
EOF
           exit 1 ;;
    esac
done

cat <<'EOF'

==> what the patch does

  Their main.py always runs index() then rag_qa(), and writes no dump this repo can
  score. The patch adds a retrieval-only path that emits {qid: [pid]} with pid an
  index into OUR corpus, so score_qafd.py reads it unchanged:

    --skip_qa        call retrieve() instead of rag_qa()
    --data_dir       read data/<ds>.json and data/<ds>_corpus.json instead of
                     reproduce/dataset/, so the corpus is ours by construction and
                     the doc->pid map is byte-for-byte identity
    --dump           where to write {qid: [pid]}
    --num_queries    truncate, for smoke runs
    --query_shard    i/n strided sharding, merged by the runner

  And the intervention itself:

    --rerank_candidate_k / --rerank_keep_k
                     DECOUPLE the candidate pool and the survivor count from
                     linking_top_k. Upstream that one number is three things at once:
                     the pool handed to the reranker (HippoRAG.py:1688), the survivor
                     count (:1699), and the seed-phrase count (:1621). So their LLM
                     filter never sees more than 5 facts and can prune but never
                     promote -- an LLM cost bottleneck, not a design choice. Both
                     default to linking_top_k, so unset they are bit-for-bit upstream.
                     NOTE graph_search_with_fact_entities weights a surviving fact by
                     its ORIGINAL embedding score, not by the reranker's, so the
                     reranker's output ORDER is discarded and only the SET it keeps
                     matters. That is why the objective is recall@keep_k.
    --rerank_mode    llm       their DSPyFilter (the paper)
                     norerank  keep the pool in embedding order, no filter
                     learned   a trained ranker from --rerank_model, which implements
                               the same rerank() signature and is swapped into
                               rag.rerank_filter with no pipeline fork
    --export_facts   dump (query, candidate facts, embeddings, labels) for training

==> next, by hand

  Env (3.10, matching their README's conda pin):
    uv venv --python 3.10 .venv-hippo
    . .venv-hippo/bin/activate
    uv pip install -r third_party/HippoRAG/requirements.txt

  Data — ours, which is already theirs: scripts/download_data.py pulls
  data/<ds>.json and data/<ds>_corpus.json from osunlp/HippoRAG_v2, the authors' own
  release. Verify before indexing:
    python scripts/check_hipporag_data.py musique

  Serve the indexing LLM:
    DEVICE=1 bash scripts/serve_gptoss.sh
  NOTE their DSPy filter prompt is tuned for Llama-3.3-70B-Instruct
  (main.py: rerank_dspy_file_path=...filter_llama3.3-70B-Instruct.json). Running it
  against gpt-oss-20B is a deviation worth recording in the write-up; it applies to
  the `llm` arm only.

  Then index + run:
    bash scripts/run_hipporag.sh
EOF
