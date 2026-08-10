"""Query-set plumbing that has to work on both sides of the version gap.

Our sub-question files are keyed by dataset qid (`2hop__13548_13529`). LinearRAG
never sees a qid — `retrieve()` is handed `question_info["question"]` — and it
runs on Python 3.9 with its own dataset bundle in a different schema. So the join
is on question text, and it is baked on our side by
`scripts/export_subq_for_linearrag.py`; this module is what both sides import so
there is exactly one definition of "the same question".

Deliberately stdlib-only and `from __future__`-guarded: it is imported inside
LinearRAG's 3.9 environment, where `mbuzai.dataio` and `mbuzai.metrics` would
both fail on their PEP 604 annotations.
"""

from __future__ import annotations

import json
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """A key that survives the trip between two dataset bundles.

    Casefold, strip accents to their base characters, drop punctuation, collapse
    whitespace. The two copies of a dataset differ in quoting and spacing far
    more often than in wording, and an exact-match join loses those rows
    silently — which would look like a weak result rather than a plumbing bug.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.casefold())
    return _SPACE.sub(" ", text).strip()


def load_query_sets(path):
    """Load an exported {normalized question: [sub-question, ...]} file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def match_qid(their_id, our_qids):
    """Their question id expressed as ours, or None.

    The bundle is not consistent about this: MuSiQue rows are prefixed with the
    source (`musique_2hop__13548_13529`) while 2Wiki rows are the bare hex id
    that we already use. Splitting on "_" unconditionally works on one and raises
    IndexError on the other, so try the id as-is first and only then strip a
    leading source token — and only if what remains is a qid we recognise.
    """
    if their_id in our_qids:
        return their_id
    _head, sep, tail = their_id.partition("_")
    if sep and tail in our_qids:
        return tail
    return None


def lookup(query_sets, question):
    """Sub-questions for `question`, or None if this question has no set.

    None is the vanilla signal all the way down: the gate falls back to the
    pooled query, which is exactly what an unmatched question should do.
    """
    return query_sets.get(normalize_question(question)) or None
