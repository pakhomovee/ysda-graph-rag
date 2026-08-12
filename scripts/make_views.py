"""Query views without an LLM — the free stand-in for sub-question generation.

A pooled query vector conflates the several information needs a multi-hop
question contains, which is the whole reason sigma_max over sub-questions helps
at the seeding stage. Sub-questions are one way to get multiple views; they cost
an LLM call per query, which is precisely what LinearRAG and QAFD-RAG sell
themselves on avoiding.

MuSiQue questions are *composed* from single-hop questions, so the structure is
present in the syntax:

    "When was the person who Messi's goals were compared to signed by Barcelona?"
     └── matrix clause: hop 2 ──┘└── relative clause: hop 1 ──┘

Measured on MuSiQue, best-over-views top-10 recall of hop-i's gold passage over
the full corpus (lexical scoring, no graph, 300 questions):

    pooled question only        43.7%   (1.0 views)
    these views                 53.3%   (3.7 views)   <- LLM-free
    gold sub-questions          63.5%   (3.7 views)   <- the LLM ceiling

About half the decomposition gain at zero inference cost.

Read that number with care: best-over-views is MONOTONE in the number of views,
so the metric rewards adding views whether or not they are any good. An earlier,
junkier generator scored 57.3% purely by emitting 5.2 views instead of 3.7.
Under sigma_max on embeddings a bad view is not free — it raises the score of
whatever it spuriously matches — so --max-views should be tuned on the real
metric, not this one. eval_subq.py is the cheap place to do that: seconds per
run, real embeddings, and the shuffled control catches views that only help by
inflating the max.

Output matches the sub-question file format exactly ({qid: [view, ...]}), so
eval_subq.py, export_subq_for_linearrag.py and QAFD's --subq_file all consume it
with no changes. View 0 is always the full question, which keeps
sigma_max >= sigma_q and the "cannot score below vanilla" guarantee.

    python scripts/make_views.py musique
    python scripts/make_views.py musique --shuffle    # the content control
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):  # dataio/metrics use PEP 604 annotations
    sys.exit(
        f"\nFATAL: this script needs the mbuzai env (Python 3.10+), got "
        f"{sys.version.split()[0]}\n  interpreter: {sys.executable}"
    )

from mbuzai import dataio  # noqa: E402

OUT = ROOT / "out"

REL = re.compile(r"\s*\b(?:who|whom|whose|which|that|where|when|in which|by which)\b\s*", re.I)
PREP = re.compile(r"\s*\b(?:of|in|for|from|at|on|by)\s+the\b\s*", re.I)
CAP = re.compile(r"\b([A-Z][a-z0-9]+(?:\s+(?:of|the|de|and)?\s*[A-Z][a-z0-9]+)*)")
STOP = set("the a an of in on at to for and or is was were be been by with from as that "
           "which who whom whose what when where how did does do its it his her their this "
           "these those not no".split())


def _content(s: str) -> int:
    """Content words, so all-stopword fragments like 'was the person' are dropped."""
    return sum(1 for w in re.findall(r"[a-zA-Z0-9']+", s.lower()) if w not in STOP)


def _entities(question: str) -> list[str]:
    """Capitalised spans, minus the sentence-initial word.

    Every question starts with a capital, so position 0 is almost always a
    wh-word rather than an entity — 'When was the person ... When' was a real
    view before this.
    """
    out = []
    for m in CAP.finditer(question):
        if m.start() == 0 and m.group().lower() in STOP:
            continue
        out.append(m.group())
    return out


def _head(question: str, n: int = 4) -> str:
    """Opening of the question, trimmed so it does not end mid-construction."""
    words = question.split()[:n]
    while words and (words[-1].lower() in STOP or words[-1].endswith("'s")):
        words.pop()
    return " ".join(words)


def _clean(s: str) -> str:
    return s.strip(" ,.?!;:").strip()


def views_regex(question: str, min_words: int, max_views: int) -> list[str]:
    """Clause and entity views from surface patterns. No parser, no model."""
    out = [question]

    def add(s):
        s = _clean(s)
        if (_content(s) >= 2 and len(s.split()) >= min_words
                and s.lower() != question.lower().strip("?")):
            out.append(s)

    for part in REL.split(question):
        add(part)
        for sub in PREP.split(part):
            add(sub)

    # Entity-anchored: the question's opening (which carries the wh-word and the
    # relation being asked) paired with each named entity in it. Short fragments
    # embed poorly on their own; this keeps them in a sentence-like context.
    head = _head(question)
    for ent in _entities(question)[:3]:
        add(f"{head} {ent}" if head else ent)

    seen, uniq = set(), []
    for v in out:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq[:max_views]


def views_spacy(nlp, question: str, min_words: int, max_views: int) -> list[str]:
    """Same idea via a dependency parse: clause subtrees rather than regex splits.

    Better than the surface patterns because it recovers clause boundaries the
    regex misses (coordination, nested modifiers) and gets noun-phrase extents
    right — 2Wiki `comparison` questions in particular split on coordination, not
    on a relative pronoun.
    """
    doc = nlp(question)
    out = [question]

    def add(s):
        s = _clean(s)
        if (_content(s) >= 2 and len(s.split()) >= min_words
                and s.lower() != question.lower().strip("?")):
            out.append(s)

    for tok in doc:
        if tok.dep_ in ("relcl", "acl", "advcl", "ccomp", "xcomp", "conj"):
            add("".join(t.text_with_ws for t in tok.subtree))
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= min_words:
            add(chunk.text)

    head = _head(question)
    for ent in doc.ents:
        add(f"{head} {ent.text}" if head else ent.text)

    seen, uniq = set(), []
    for v in out:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq[:max_views]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--min-words", type=int, default=3,
                    help="drop fragments shorter than this; very short spans embed badly")
    ap.add_argument("--max-views", type=int, default=4,
                    help="cap per question. The seed budget is fixed, so more views "
                         "means better selection within it, not more seeds.")
    ap.add_argument("--spacy-model", default=None,
                    help="e.g. en_core_web_sm. Omit to use surface patterns only.")
    ap.add_argument("--shuffle", action="store_true",
                    help="content control: emit views attached to the WRONG question, "
                         "same count and same generator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = dataio.load(args.dataset)

    nlp = None
    if args.spacy_model:
        try:
            import spacy
            nlp = spacy.load(args.spacy_model, disable=["ner"] if False else [])
            print(f"using spacy {args.spacy_model}")
        except Exception as exc:
            print(f"spacy unavailable ({exc}); falling back to surface patterns")

    gen = (lambda q: views_spacy(nlp, q, args.min_words, args.max_views)) if nlp \
        else (lambda q: views_regex(q, args.min_words, args.max_views))

    out = {q.qid: gen(q.question) for q in ds.queries}

    tag = "views"
    if args.shuffle:
        # Same generator, same counts, attached to the wrong question. If this
        # gains too, the effect is the max operator rather than the views.
        keys = list(out)
        vals = [out[k][1:] for k in keys]           # keep each question's own view 0
        random.Random(args.seed).shuffle(vals)
        out = {k: [out[k][0]] + v for k, v in zip(keys, vals)}
        tag = "viewsshuf"

    dest = OUT / f"subq_{ds.name}_{tag}.json"
    dest.write_text(json.dumps(out, indent=1))

    sizes = [len(v) for v in out.values()]
    print(f"{ds.name}: {len(out)} questions, mean {sum(sizes)/len(sizes):.2f} views each "
          f"(min {min(sizes)}, max {max(sizes)})")
    print(f"  {sum(1 for s in sizes if s == 1)} questions got no view beyond the question itself")
    q = ds.queries[0]
    print(f"\nexample  {q.qid}\n  Q  {q.question}")
    for v in out[q.qid][1:]:
        print(f"  -  {v}")
    print(f"\nwrote {dest}")
    print("\nView 0 is the question itself, so sigma_max >= sigma_q holds and the arm "
          "cannot score below pooled by construction.")


if __name__ == "__main__":
    main()
