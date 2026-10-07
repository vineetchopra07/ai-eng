import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# def show(text):
#     ids = enc.encode(text)
#     print(f"\n{text!r} -> {len(ids)} tokens")
#     print("  ids:  ", ids)
#     print("  parts:", [enc.decode([i]) for i in ids])

# show("The cat sat on the mat")
# show("cat"); show(" cat"); show("Cat")
# show("1234567890")
# show("Hello how are you")
# show("नमस्ते आप कैसे हैं")

for i in enc.encode("नमस्ते"):
    print(f"{i} -> {enc.decode([i])}")