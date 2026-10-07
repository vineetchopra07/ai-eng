import json, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = np.load("vectors.npy")
corpus = json.load(open("corpus.json"))

def search(query: str, k: int = 5):
    q = model.encode(query)
    q = q / np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [(scores[i], corpus[i]) for i in top]

def show(query: str, k: int = 5):
    print(f"\nQ: {query}")
    for score, c in search(query, k):
        print(f"{score:.3f}  {c['source']}  [{c['position']}]")
        print(f"   {c['text'][:300].replace(chr(10), ' ')}...")