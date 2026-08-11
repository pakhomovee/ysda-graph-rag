#!/usr/bin/env bash
# Bring up the patched QAFD-RAG working copy.
#
# Same shape as setup_linearrag.sh: the submodule is pinned to the commit the
# patch was written against, referenced rather than vendored, and idempotent.
#
#     bash scripts/setup_qafd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/QAFD-RAG"
PATCH="$ROOT/patches/qafd_sigma_max.patch"

echo "==> submodule"
git -C "$ROOT" submodule update --init --recursive third_party/QAFD-RAG
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
    # shellcheck disable=SC2086
    git -C "$SUB" checkout -- $FILES
    if git -C "$SUB" apply "$PATCH"; then
        echo "    reapplied $(basename "$PATCH")"
    else
        echo "    FATAL: patch does not apply even to a clean tree." >&2
        git -C "$SUB" log -1 --format='      at %h %s' >&2
        exit 1
    fi
fi

echo "==> sanity"
grep -q "sigma_max" "$SUB/src/passage_entity/graph_adapter.py" && echo "    _cosine_similarity takes a query set"
grep -q "subq_scope" "$SUB/src/passage_entity/config.py"       && echo "    config exposes subq_file / subq_scope"
grep -q "_query_matrix" "$SUB/src/passage_entity/retriever.py" && echo "    retriever builds the query set"

cat <<EOF

==> what the patch does
  Every query-aware similarity in the passage-entity pipeline funnels through
  one function, _cosine_similarity(node_emb, query_embedding). It now accepts a
  (m, d) query set and returns the max, so the intervention reaches all of:
    * edge weighting        w * sim(u,q) * sim(v,q)   <- the flow diffusion itself
    * sink / warm-start node similarities
    * the post-hoc lambda reweighting
  A 1-D query embedding is bit-for-bit unchanged, so no --subq_file means the
  original algorithm.

  subq_scope separates the two places the query is used:
    edges  (default)  modulate diffusion edge weights only
    seeds             sigma_max on fact scores, which drive node seeding
    both
  Keep them separable. In LinearRAG the gate improved the activation frontier
  while seeding stayed pooled, and the two effects could not be told apart
  afterwards.

==> next, by hand
  Python 3.10 env:
    uv venv --python 3.10 .venv-qafd
    . .venv-qafd/bin/activate
    uv pip install -r $SUB/requirements.txt

  Their registry shipped no sentence-transformers option — NV-Embed-v2 (7B),
  GritLM (7B), Jina-v3, OpenAI — so the patch adds one and registers it as
  "mpnet". Use it: every other row in this repo is all-mpnet-base-v2 (110M), and
  a 7B encoder would win on encoder strength alone, saying nothing about flow
  diffusion.
    --embedding_model mpnet

  Indexing runs OpenIE over the corpus, so budget for ~11.7k LLM calls on
  MuSiQue. Point it at your local vLLM server, on whatever port you serve:
    curl -s http://localhost:<PORT>/v1/models | python3 -m json.tool
  llm_model must equal the "id" that comes back — with --served-model-name it is
  not the checkpoint path, and a mismatch fails per request, so you discover it
  ~11.7k times during indexing rather than once at startup.
    llm_base_url = http://localhost:<PORT>/v1
    llm_api_key  = any non-empty string (vLLM ignores it, the client requires it)

  Feed it OUR corpus, exactly as with musique_fine, so recall lands in the same
  units as baselines.py, eval_subq.py and LinearRAG:
    python scripts/export_corpus_for_linearrag.py musique --name musique_fine
  and the sub-questions re-keyed by question text:
    python scripts/export_subq_for_linearrag.py musique --subq out/subq_musique_generated.json
EOF
