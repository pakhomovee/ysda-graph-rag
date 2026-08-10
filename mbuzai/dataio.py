"""Per-dataset adapters onto one common shape.

MuSiQue and 2Wiki ship different schemas (`paragraphs`/`is_supporting` vs
`context`/`supporting_facts`), so every downstream script reads through here
instead of touching the raw JSON.

Passages are identified by their position in the corpus (`pid`), resolved via a
hash of (title, text). Keying on title alone is wrong: 2,465 of MuSiQue's 11,656
passages share a title with another, and 47/1000 questions have gold paragraphs
that collide on it.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _key(title: str, text: str) -> str:
    return hashlib.md5(f"{title}||{text}".encode()).hexdigest()


@dataclass
class Hop:
    """One reasoning step. `gold_pid` is None when the step's passage is unresolvable."""

    question: str
    answer: str
    gold_pid: int | None


@dataclass
class Query:
    qid: str
    question: str
    answer: str
    gold_pids: set[int]
    hops: list[Hop] = field(default_factory=list)
    shape: str = ""       # musique: 2hop / 3hop1 / 4hop2 ...  2wiki: reasoning type
    n_hops: int = 0
    is_join: bool = False  # a sub-question referencing two earlier answers


@dataclass
class Dataset:
    name: str
    corpus: list[dict]            # [{"title", "text"}]
    queries: list[Query]

    @property
    def docs(self) -> list[str]:
        return [f"{c['title']}\n{c['text']}" for c in self.corpus]


def _load_corpus(name: str):
    corpus = json.loads((DATA / f"{name}_corpus.json").read_text())
    index = {_key(c["title"], c["text"]): i for i, c in enumerate(corpus)}
    by_title: dict[str, list[int]] = {}
    for i, c in enumerate(corpus):
        by_title.setdefault(c["title"], []).append(i)
    return corpus, index, by_title


def load_musique(name: str = "musique") -> Dataset:
    corpus, index, _ = _load_corpus(name)
    raw = json.loads((DATA / f"{name}.json").read_text())

    queries = []
    for ex in raw:
        paras = {p["idx"]: p for p in ex["paragraphs"]}
        pid_of = {
            idx: index.get(_key(p["title"], p["paragraph_text"]))
            for idx, p in paras.items()
        }
        gold = {
            pid_of[p["idx"]]
            for p in ex["paragraphs"]
            if p.get("is_supporting") and pid_of[p["idx"]] is not None
        }
        hops = [
            Hop(d["question"], d["answer"], pid_of.get(d["paragraph_support_idx"]))
            for d in ex["question_decomposition"]
        ]
        refs = [
            sorted(int(m) for m in re.findall(r"#(\d)", d["question"]))
            for d in ex["question_decomposition"]
        ]
        queries.append(
            Query(
                qid=ex["id"],
                question=ex["question"],
                answer=ex["answer"],
                gold_pids=gold,
                hops=hops,
                shape=ex["id"].split("__")[0],
                n_hops=len(hops),
                is_join=any(len(r) > 1 for r in refs),
            )
        )
    return Dataset(name, corpus, queries)


def load_2wiki(name: str = "2wikimultihopqa") -> Dataset:
    corpus, _, by_title = _load_corpus(name)
    raw = json.loads((DATA / f"{name}.json").read_text())

    queries = []
    for ex in raw:
        # supporting_facts references passages by title; 2Wiki titles are unique
        # within its corpus, so a title lookup is safe here (unlike MuSiQue).
        gold = {
            pid
            for title, _sent_idx in ex["supporting_facts"]
            for pid in by_title.get(title, [])
        }
        queries.append(
            Query(
                qid=ex["_id"],
                question=ex["question"],
                answer=ex["answer"],
                gold_pids=gold,
                hops=[],  # 2Wiki has no sub-questions; it has gold triples instead
                shape=ex["type"],
                n_hops=len(ex.get("evidences", [])) or len(ex["supporting_facts"]),
            )
        )
    return Dataset(name, corpus, queries)


def load_generic(name: str) -> Dataset:
    """HippoRAG-format single-hop sets (popqa, nq_rear) — no supporting labels."""
    corpus, index, _ = _load_corpus(name)
    raw = json.loads((DATA / f"{name}.json").read_text())
    queries = []
    for i, ex in enumerate(raw):
        paras = ex.get("paragraphs", [])
        gold = {
            index[k]
            for p in paras
            if p.get("is_supporting")
            and (k := _key(p["title"], p.get("paragraph_text", p.get("text", "")))) in index
        }
        queries.append(
            Query(
                qid=str(ex.get("id", i)),
                question=ex["question"],
                answer=str(ex.get("answer", "")),
                gold_pids=gold,
                shape="single",
                n_hops=1,
            )
        )
    return Dataset(name, corpus, queries)


LOADERS = {
    "musique": load_musique,
    "2wikimultihopqa": load_2wiki,
    "hotpotqa": load_generic,
    "popqa": load_generic,
    "nq_rear": load_generic,
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"no adapter for {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](name)
