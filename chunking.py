# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
# # print(model.max_seq_length)

# # v = model.encode("I was stressed about work today")
# # print(v.shape)
# # print(v[:8])

# from sentence_transformers.util import cos_sim

# # sents = [
# #     "I was stressed about work today",
# #     "Work has been really overwhelming lately",
# #     "I went for a long walk in the evening",
# #     "The capital of France is Paris",
# # ]
# # e = model.encode(sents)
# # print(cos_sim(e, e))
# from pathlib import Path
# file = Path('path/to/a/note.md')
# text = file.read_text(encoding="utf-8")


# #fixed size chunking -
# #chunk = 1000
# # for i in range(0, len(text), chunk):
# #     print("length of chunk:", len(text[i:i + chunk]))
# #     print("last 50 characters of chunk:", text[i+950:i + chunk])
# #     print("first 50 characters of chunk:", text[i:i + 50])
# #     print("chunk number:", i//chunk, "----------------")

# #fixed size chunking with overlap - 1st chunk - 0:1050 , 2nd -> 800:1850 
# #chunk size = 1000, overlap = 200
# overlap = 200
# count =0 
# step = 800
# for i in range(0, len(text), step):
#     count += 1
#     print("chunk number:", count, "----------------")
#     text_chunk = text[i:i + step + overlap]
#     print("length of chunk:", len(text_chunk))
#     print("first 50 characters of chunk:", text_chunk[:200])
#     print("last 50 characters of chunk:", text_chunk[-200:])
#     print( "----------------")


# full = model.encode(text)
# head = model.encode(text[:1075])
# middle = model.encode(text[5000:6000])

# print("full vs head: ", cos_sim(full, head).item())
# print("full vs middle: ", cos_sim(full, middle).item())

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
from pathlib import Path
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_fixed(text: str, size: int = 800) -> list[str]:
    """Split into consecutive pieces of `size` characters."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def chunk_overlap(text: str, size: int = 800, overlap: int = 200) -> list[str]:
    """Split into `size`-character pieces, each repeating the previous `overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

def chunk_recursive(text: str, size: int = 800, seps=None) -> list[str]:
    if seps is None:
        seps = ["\n\n", "\n", ". ", " "]
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                if len(current) + len(part) + len(sep) > size:
                    if current:
                        chunks.append(current)
                    current = part
                else:
                    if current:
                        current += sep + part
                    else:
                        current = part
            if current:
                chunks.append(current)
            # Recursively chunk any pieces that are still too large
            final_chunks = []
            for chunk in chunks:
                final_chunks.extend(chunk_recursive(chunk, size=size, seps=seps[1:]))
            return final_chunks

def chunk_tokens(text: str, max_tokens: int = 250) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for token in tokens:
        if current_length + 1 > max_tokens:
            chunks.append(enc.decode(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1
    if current_chunk:
        chunks.append(enc.decode(current_chunk))
    return chunks


import re
import numpy as np
from sentence_transformers.util import cos_sim

def enforce_limit(chunks: list[str], max_tokens: int = 250) -> list[str]:
    out = []
    for c in chunks:
        if len(enc.encode(c)) <= max_tokens:
            out.append(c)
        else:
            out.extend(chunk_tokens(c, max_tokens))
    return out

def chunk_semantic(text: str, model, percentile: int = 10) -> list[str]:
    """Split where consecutive sentences are least similar."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) < 3:
        return [text]

    e = model.encode(sentences)

    # pass 1 — score every gap
    scores = [cos_sim(e[i], e[i + 1]).item() for i in range(len(e) - 1)]

    # threshold relative to this document, not an absolute number
    threshold = np.percentile(scores, percentile)
    boundaries = {i for i, s in enumerate(scores) if s < threshold}

    # pass 2 — group
    chunks, current = [], []
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i in boundaries:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))

    return chunks



def show_seams(chunks: list[str], n: int = 50) -> None:
    for i, (a, b) in enumerate(zip(chunks, chunks[1:])):
        print(f"\n--- {i} -> {i+1} ---")
        print("end  :", repr(a[-n:]))
        print("start:", repr(b[:n]))


def report(name: str, chunks: list[str]) -> None:
    tokens = [len(enc.encode(c)) for c in chunks]
    over = sum(t > 256 for t in tokens)
    print(f"{name}: {len(chunks)} chunks, max {max(tokens)} tokens, {over} over limit")


import os
Vault = Path(os.environ.get("VAULT_PATH", "notes")).expanduser()
files = list(Vault.rglob("*.md"))
import numpy as np
import json

# f.name    # 'why-leaving.md'   filename with extension
# f.stem    # 'why-leaving'      filename without extension
# f.parent  # the Archive folder
# str(f)    # the full path as a string

def get_file(files: list[Path]):
    count = 1
    for f in files:
        print(count, "---", f.name,"---",)
        text = f.read_text(encoding="utf-8")
        if not text.strip():
            print(f"{count}. --- File {f.name} is empty")
        count += 1

def build_corpus(files: list[Path]) -> list[dict]:
    corpus = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chunks = enforce_limit(chunk_recursive(text, size=800), max_tokens=250)
        # print(f"File {f.name} has {len(chunks)} chunks ---------", "\n")
        for i, chunk in enumerate(chunks):
            corpus.append({
                "text": chunk,
                "source": f.name,
                "position":i
            })
    print("total number of chunks in corpus:", len(corpus))
    # print(corpus[0])
    return corpus

def embed_corpus(corpus: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in corpus]
    vectors = model.encode(texts, show_progress_bar=True)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


if __name__ == "__main__":
    corpus = build_corpus(files)
    vectors = embed_corpus(corpus)
    np.save("vectors.npy", vectors)
    with open("corpus.json", "w") as fh:
        json.dump(corpus, fh)










# if __name__ == "__main__":
#     build_corpus(files)
    
    # fixed = chunk_fixed(text)
    # over = chunk_overlap(text)
    # recursive = chunk_recursive(text)
    # tokens = chunk_tokens(text)
    # semantic = chunk_semantic(text, model)
    # report("fixed  ", fixed)
    # report("overlap", over)
    # report("recursive", recursive)
    # report("tokens", tokens)
    # report("semantic", semantic)
    # report("semantic enforced", enforce_limit(semantic))
    # show_seams(recursive[:3])