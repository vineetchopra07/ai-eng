import os
from pathlib import Path

vault = Path(os.environ.get("VAULT_PATH", "notes")).expanduser()
files = list(vault.rglob("*.md"))
# print(len(files))

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

for f in files:
    text = f.read_text(encoding="utf-8")
    if(text.strip()):
        print(f.name,len(text), len(enc.encode(text)))
