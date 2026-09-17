import json
import os
from rank_bm25 import BM25Okapi

# Path to cases.json relative to this file, or we can use absolute.
# We'll use absolute based on project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CASES_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "processed", "cases.json")

_bm25_index = None
_cases_data = None

def _load_data():
    global _bm25_index, _cases_data
    if _bm25_index is not None:
        return
    
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        _cases_data = json.load(f)
        
    corpus = []
    for case in _cases_data:
        # Combine title, summary, facts, judgment
        text_parts = [
            case.get("title", ""),
            case.get("summary", ""),
            case.get("facts", ""),
            case.get("judgment", "")
        ]
        text = " ".join(text_parts).lower()
        # Tokenize by splitting on space for basic BM25
        corpus.append(text.split())
        
    _bm25_index = BM25Okapi(corpus)

def bm25_search(query_text: str, top_k: int = 100):
    _load_data()
    
    tokenized_query = query_text.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)
    
    # rank
    results = []
    for idx, score in enumerate(scores):
        if score > 0:
            results.append((score, _cases_data[idx]))
            
    # sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    
    # top_k
    return [item[1] for item in results[:top_k]]
