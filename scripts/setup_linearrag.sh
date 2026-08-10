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
  Python 3.9 env, then inside $SUB:
    pip install -r requirements.txt
    python -m spacy download en_core_web_trf

  Their dataset bundle (multi-GB, git-lfs — not a submodule):
    git clone https://huggingface.co/datasets/Zly0523/linear-rag
    cp -r linear-rag/* $SUB/dataset/
  Note their MuSiQue sibling is named 2wikimultihop, not 2wikimultihopqa.

  So mbuzai.gate and mbuzai.subq_io are importable on 3.9:
    export PYTHONPATH=$ROOT:\$PYTHONPATH

  Reproduce their vanilla numbers BEFORE running the sigma_max arm. The patch is
  inert without --subq_file (bitwise verified), so any gap there is the harness.
EOF
