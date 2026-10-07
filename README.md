# ai-eng: retrieval over your own notes

A from-scratch retrieval-augmented generation (RAG) pipeline built while learning AI engineering. It turns a folder of Markdown notes into a searchable index and answers questions about them with a local LLM, so your notes never leave your machine.

```
Markdown notes ──► chunk ──► embed ──► corpus.json + vectors.npy
                                              │
          question ──► dense search ─┐        │
                   └─► BM25 search ──┴─► RRF fusion ──► top-k passages ──► local LLM ──► cited answer
```

## What's inside

| File | What it does |
|---|---|
| `01_tokenizers.py` | Explores how `tiktoken` (cl100k_base) splits text, including non-English scripts. |
| `file_load.py` | Lists every note in the vault with its character and token counts. |
| `chunking.py` | Five chunking strategies (fixed, overlapping, recursive, token-based, semantic) plus a token limit. Running it builds the corpus and embeddings. |
| `search.py` | Dense search: cosine similarity over normalised `all-MiniLM-L6-v2` embeddings. |
| `hybrid.py` | Hybrid search: dense + BM25 keyword search, merged with Reciprocal Rank Fusion. |
| `evaluate.py` | Measures recall@k against a hand-written set of questions and expected source notes. |
| `generate.py` | Builds a numbered context and asks a local Llama 3.2 (via Ollama) for an answer that cites its sources. |

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com) for the answer step.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

## Usage

**1. Point it at your notes.** Any folder of `.md` files works, for example an Obsidian vault. It defaults to `./notes`.

```bash
export VAULT_PATH=~/path/to/your/notes
```

**2. Build the index.** This writes `corpus.json` (the chunks) and `vectors.npy` (their embeddings).

```bash
python chunking.py
```

**3. Search.**

```python
from search import show
show("what did I plan for the launch")

from hybrid import hybrid_search
hybrid_search("launch plan", k=5)
```

**4. Ask questions.**

```python
from generate import answer
answer("what did I plan for the launch")
```

**5. Evaluate retrieval.**

```bash
cp eval_set.example.json eval_set.json   # then write questions about your own notes
python evaluate.py                        # prints PASS/FAIL per question and recall@k
```

## Privacy

Your notes, and everything made from them, stay local and are git-ignored:

- `corpus.json`: chunked note text
- `vectors.npy`: embeddings of that text
- `eval_set.json`: your evaluation questions
- `notes/`: the default notes folder

Generation runs on a local model through Ollama, so nothing is sent to an external API.

## Concepts covered

Tokenisation · chunking trade-offs (size, overlap, semantic boundaries) · sentence embeddings and cosine similarity · BM25 · Reciprocal Rank Fusion · recall@k evaluation · grounded generation with citations
