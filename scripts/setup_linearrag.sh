#!/usr/bin/env bash
# Bring up the patched LinearRAG working copy.
#
# The submodule is pinned to the commit the patch was written against, so a
# silent upstream change cannot make the patch apply to the wrong lines. It is
# GPL-3.0 and is referenced, never vendored — nothing of theirs lands in this
# repo's history.
#
# Idempotent: re-running detects an already-patched tree and leaves it alone.
#
#     bash scripts/setup_linearrag.sh
#
# Not covered here, deliberately: their dataset bundle is multi-GB git-lfs and
# has no business in a git submodule. See the end of this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/LinearRAG"
PATCH="$ROOT/patches/linearrag_sigma_max.patch"

echo "==> submodule"
git -C "$ROOT" submodule update --init --recursive third_party/LinearRAG
echo "    at $(git -C "$SUB" rev-parse --short HEAD)"

echo "==> patch"
if git -C "$SUB" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "    already applied — nothing to do"
elif git -C "$SUB" apply --check "$PATCH" 2>/dev/null; then
    git -C "$SUB" apply "$PATCH"
    echo "    applied $(basename "$PATCH")"
else
    echo "    FATAL: patch neither applies nor is already applied." >&2
    echo "    The submodule is not at the pinned commit, or has local edits:" >&2
    git -C "$SUB" status --short >&2
    exit 1
fi

echo "==> sanity"
grep -q "sigma_max(sentence_embeddings" "$SUB/src/LinearRAG.py" \
    && echo "    BFS gate patched"
grep -q "sigma_max(self.sentence_embeddings" "$SUB/src/LinearRAG.py" \
    && echo "    vectorized gate patched"
grep -q "question_emb.reshape(1, -1)" "$SUB/src/LinearRAG.py" 2>/dev/null \
    || grep -q "question_embedding.reshape(1, -1)" "$SUB/src/LinearRAG.py" \
    && echo "    dense_passage_retrieval left pooled (intended)"

cat <<EOF

==> next, by hand
  A Python 3.9 env. spacy-transformers is missing from their requirements even
  though en_core_web_trf needs it:
    uv venv --python 3.9 .venv-linear
    . .venv-linear/bin/activate
    uv pip install -r $SUB/requirements.txt
    uv pip install spacy-transformers
    python -m spacy download en_core_web_trf
    deactivate

  Their dataset bundle — git-lfs, so not a submodule, but small. Only the two
  datasets we use, ~11 MB:
    git clone https://huggingface.co/datasets/Zly0523/linear-rag /tmp/linear-rag
    cp -r /tmp/linear-rag/musique /tmp/linear-rag/2wikimultihop $SUB/dataset/
  Their 2Wiki is named 2wikimultihop, ours is 2wikimultihopqa.

  Then smoke-test on 20 questions before committing to a full run:
    LIMIT=20 PY39=.venv-linear/bin/python bash scripts/run_linear_musique.sh
    PY39=.venv-linear/bin/python bash scripts/run_linear_musique.sh

  The run scripts export PYTHONPATH themselves. Only exporting it by hand
  matters if you drive run_linearrag_retrieval.py directly:
    export PYTHONPATH=$ROOT:\$PYTHONPATH

  On "reproducing their numbers": you cannot match their published table. They
  report QA EM/F1 through an LLM, and their bundle ships no gold passages, so our
  recall uses our own gold mapped onto their chunks. What the vanilla arm gives
  you is a baseline on identical footing with the sigma_max arm — which is the
  only comparison the claim needs. Sanity-check it is far above chance, then read
  the paired delta.
EOF
