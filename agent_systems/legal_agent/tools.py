from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]

class ToolError(Exception):
    pass

def _load_doc(doc_id: str, corpus_dir: str) -> dict:
    # We only allow CH*-style files produced by our converter
    if not (isinstance(doc_id, str) and doc_id.startswith("CH")):
        raise ToolError(f"Unknown doc_id format: {doc_id!r}. Expected CH* ids from the local corpus.")
    root = corpus_dir or os.environ.get("LEGAL_CORPUS_DIR", "") or CORPUS_DIR
    fp = os.path.join(root, f"{doc_id}.json")
    if not os.path.exists(fp):
        raise ToolError(f"Document not found: {doc_id} ({fp})")
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

# ---- Simple local "index": list JSON docs from a folder ----
# Each doc file: {"doc_id": "001", "title": "...", "text": "..."}
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "legal_corpus")

_STOP = set("the a an and or but if in on to of for by with as at from this that those these is are was were be been being it its".split())

def _load_corpus() -> List[Dict[str, str]]:
    docs = []
    if not os.path.isdir(CORPUS_DIR):
        return docs
    for fn in sorted(os.listdir(CORPUS_DIR)):
        if not fn.endswith(".json"): continue
        p = os.path.join(CORPUS_DIR, fn)
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            docs.append(d)
    return docs

_DOCS_CACHE: List[Dict[str, str]] | None = None
def _docs() -> List[Dict[str, str]]:
    global _DOCS_CACHE
    if _DOCS_CACHE is None:
        _DOCS_CACHE = _load_corpus()
    return _DOCS_CACHE

def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP]

def _score(query: str, text: str) -> int:
    q = set(_tokens(query))
    t = _tokens(text)
    return sum(1 for w in t if w in q)

# ---- Tools ----
def retrieve(query: str, k: int = 3) -> List[Dict[str, str]]:
    docs = _docs()
    scored = [( _score(query, d["title"] + " " + d["text"]), d ) for d in docs]
    scored.sort(key=lambda x: (-x[0], dkey(x[1])))
    out = [d for s,d in scored if s>0][:k] or [d for _,d in scored[:k]]
    # Return minimal stable fields
    return [{"doc_id": d["doc_id"], "title": d["title"], "text": d["text"]} for d in out]

def dkey(d: Dict[str,str]) -> tuple:
    # Tie-break deterministically: by doc_id then title
    return (str(d.get("doc_id","")), str(d.get("title","")))

def quote(doc_id: str, start: int, end: int) -> str:
    """Return text[start:end] from the local corpus doc. Clamps to valid bounds."""
    doc = _load_doc(doc_id, os.environ.get("LEGAL_CORPUS_DIR", ""))
    text = doc.get("text", "")
    n = len(text)

    if not isinstance(start, int) or not isinstance(end, int):
        raise ToolError("quote: start/end must be integers")
    if start < 0: start = 0
    if end < 0: end = 0
    if start > n: start = n
    if end > n: end = n
    if end < start: end = start

    MAX_SPAN = 400
    if end - start > MAX_SPAN:
        end = start + MAX_SPAN

    return text[start:end]

def summarize(text: str, max_words: int = 80) -> str:
    # Deterministic extractive summary: first N words
    words = re.findall(r"\S+", text)
    return " ".join(words[:max_words])

def compare(text_a: str, text_b: str, criterion: str) -> str:
    # Deterministic heuristic comparator: length & keyword hits
    cw = set(_tokens(criterion))
    a_hits = sum(1 for w in _tokens(text_a) if w in cw)
    b_hits = sum(1 for w in _tokens(text_b) if w in cw)
    if a_hits != b_hits:
        return "A" if a_hits > b_hits else "B"
    return "A" if len(text_a) >= len(text_b) else "B"

def run_tool(name: str, **kwargs) -> Tuple[Dict[str, Any], Any]:
    call = {"name": name, "args": kwargs}
    if name == "retrieve":
        res = retrieve(**kwargs)
    elif name == "quote":
        try:
            a = int(float(kwargs.get("start", 0)))
            b = int(float(kwargs.get("end", 0)))
        except Exception:
            raise ToolError("quote: start/end must be numeric")
        res = quote(str(kwargs.get("doc_id", "")), a, b)
    elif name == "summarize":
        res = summarize(**kwargs)
    elif name == "compare":
        res = compare(**kwargs)
    elif name == "stop":
        res = None
    else:
        raise ToolError(f"Unknown tool: {name}")
    return call, res
