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
    # Neither clean nor current: almost always an OLDER revision of this same
    # patch, left over from before the patch file was updated. Every change we
    # make to LinearRAG lives in the patch and nowhere else, so resetting the
    # files it touches and reapplying is safe and is the only way a patch update
    # ever lands. Anything else here is a genuine conflict.
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
        echo "    The submodule is not at the pinned commit:" >&2
        git -C "$SUB" log -1 --format='      at %h %s' >&2
        exit 1
    fi
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
  'python -m spacy download' needs pip, which uv venvs do not ship. Install the
  model wheel directly instead — 3.6.1 to match their pinned spacy, and
  spacy-transformers <1.3 because 1.3+ requires spacy 3.7:
    M=https://github.com/explosion/spacy-models/releases/download
    uv pip install "spacy-transformers<1.3" \
      \$M/en_core_web_trf-3.6.1/en_core_web_trf-3.6.1-py3-none-any.whl
    python -c "import spacy; spacy.load('en_core_web_trf'); print('spacy ok')"
    deactivate

  Their dataset bundle, ~11 MB for the two datasets we use. Fetch the files
  directly: cloning the HF repo without git-lfs gives pointer stubs, not JSON.
    B=https://huggingface.co/datasets/Zly0523/linear-rag/resolve/main
    for d in musique 2wikimultihop; do
      mkdir -p $SUB/dataset/\$d
      curl -L -o $SUB/dataset/\$d/chunks.json    \$B/\$d/chunks.json
      curl -L -o $SUB/dataset/\$d/questions.json \$B/\$d/questions.json
    done
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
