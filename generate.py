import ollama
from search import search, corpus, vectors

def build_context(hits) -> str:
    blocks = []
    for n, (score, c) in enumerate(hits, start=1):
        blocks.append(f"[{n}] {c['source']}\n{c['text']}")
    return "\n\n".join(blocks)

SYSTEM = """You answer questions about the user's personal journal.

Rules:
- Use ONLY the numbered context passages provided. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly. Do not guess.
- Cite the passages you used by their number, like [1] or [3].
- Quote the user's own words where useful. Keep the answer brief."""

def build_user_message(query: str, context: str) -> str:
    return f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}"

def ask_llm(system: str, user: str) -> str:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]

def answer(query: str, k: int = 5) -> None:
    hits = search(query, k)
    context = build_context(hits)
    user_msg = build_user_message(query, context)

    print(f"\nQ: {query}\n")
    print(ask_llm(SYSTEM, user_msg))

    print("\nSources:")
    for n, (score, c) in enumerate(hits, start=1):
        print(f"  [{n}] {score:.3f}  {c['source']}")