# utils/helpers.py
import re
from difflib import SequenceMatcher

def highlight_terms(text: str, terms: list):
    # naive highlight: wrap matches with <b>...</b> for Streamlit's unsafe_allow_html
    safe = text
    for t in sorted(set(terms), key=len, reverse=True):
        if not t.strip():
            continue
        # simple case-insensitive replace
        safe = re.sub("(?i)"+re.escape(t), lambda m: f"<b>{m.group(0)}</b>", safe)
    return safe

def top_query_terms(query: str, n=5):
    # split, filter stopwords minimally
    words = re.findall(r"\w+", query.lower())
    words = [w for w in words if len(w) > 2]
    # dedup maintain order
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= n:
            break
    return out
