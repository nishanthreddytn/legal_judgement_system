from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE = Path(__file__).resolve().parents[2]

CASES_FILE = BASE / "data" / "processed" / "cases.json"
EMBEDDINGS_FILE = BASE / "data" / "embeddings" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_cases = None
_embeddings = None
_model = None
_index = None


def _load():
    global _cases, _embeddings, _model, _index

    if _cases is None:
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            _cases = json.load(f)

    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_FILE).astype('float32')

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        
    if _index is None:
        d = _embeddings.shape[1]
        _index = faiss.IndexFlatIP(d)
        _index.add(_embeddings)

def faiss_search(query_text, top_k=100):
    _load()
    
    query = _model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype('float32')
    
    k_search = min(top_k, len(_cases))
    distances, indices = _index.search(query, k_search)
    
    results = []
    for i in range(k_search):
        idx = indices[0][i]
        score = distances[0][i]
        
        if idx < 0 or idx >= len(_cases):
            continue
            
        case = _cases[idx]
        case_result = case.copy()
        case_result['similarity'] = float(score)
        results.append(case_result)
        
    return results
