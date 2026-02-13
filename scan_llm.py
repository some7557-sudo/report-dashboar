import re
from pathlib import Path

pats = [
    r"https?://api\.groq\.com",
    r"generativelanguage\.googleapis\.com",
    r"api\.openai\.com",
    r"Bearer\s+",
    r"OPENAI_API_KEY",
    r"GROQ_API_KEY",
    r"GEMINI_API_KEY",
]

files = list(Path(".").rglob("*.py"))
print("py files:", len(files))

hits = []
for f in files:
    try:
        t = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if any(re.search(p, t) for p in pats):
        hits.append(str(f))

if not hits:
    print("HIT: (none)")
else:
    for h in hits:
        print("HIT:", h)
