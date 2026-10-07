from search import search

import json
from pathlib import Path

# Real eval questions reference private notes, so they live in a git-ignored file.
# Copy eval_set.example.json to eval_set.json and fill in your own.
EVAL_FILE = Path(__file__).parent / "eval_set.json"
if not EVAL_FILE.exists():
    raise SystemExit("eval_set.json not found: copy eval_set.example.json and edit it")
EVAL = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

from hybrid import hybrid_search
def evaluate(retriever=search, k=10):
    hits_count = 0
    for e in EVAL:
        sources = [c["source"] for _, c in retriever(e["q"], k)]
        ok = e["expect"] in sources
        hits_count += ok
        if not ok:
            print(f"FAIL  {e['q']}  ->  {sources[:3]}")
        else:
            print(f"PASS  {e['q']}  ->  {sources[:3]}")
    print(f"recall@{k}: {hits_count}/{len(EVAL)} = {hits_count/len(EVAL):.3f}")

if __name__ == "__main__":
    # evaluate(search)
    evaluate(hybrid_search)