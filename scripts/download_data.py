"""Fetch the HippoRAG-protocol datasets used by the analysis scripts.

These are the copies the diagnostics read. To *run* LinearRAG itself you also
need its own bundle, which is a different repo and a different schema:

    git clone https://huggingface.co/datasets/Zly0523/linear-rag
    cp -r linear-rag/* <LinearRAG>/dataset/

    python scripts/download_data.py              # the five defaults
    python scripts/download_data.py musique      # just one
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / "data"
REPO = "osunlp/HippoRAG_v2"
DEFAULT = ["musique", "2wikimultihopqa", "hotpotqa", "popqa", "nq_rear"]


def fetch(name: str):
    DATA.mkdir(parents=True, exist_ok=True)
    for fname in (f"{name}.json", f"{name}_corpus.json"):
        dest = DATA / fname
        if dest.exists():
            print(f"  have {fname} ({dest.stat().st_size // 1024} KB)")
            continue
        src = hf_hub_download(REPO, fname, repo_type="dataset")
        shutil.copy(src, dest)
        print(f"  got  {fname} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT:
        print(name)
        fetch(name)
