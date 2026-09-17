from app.services.faiss_search import faiss_search, _load, _cases
from app.services.bm25_search import bm25_search

def _format_case(case_dict, score):
    return {
        "case_id": case_dict.get("case_id", ""),
        "title": case_dict.get("title", "Indian Legal Judgment"),
        "year": case_dict.get("year"),
        "case_type": case_dict.get("case_type", ""),
        "sub_case_type": case_dict.get("sub_case_type", ""),
        "similarity": score,
        "summary": case_dict.get("summary", ""),
        "facts": case_dict.get("facts", ""),
        "judgment": case_dict.get("judgment", ""),
    }

def _filter_results(raw_results, case_type, sub_case_type, top_k):
    results = []
    used_ids = set()

    for case in raw_results:
        case_id = case.get("case_id", "")
        if not case_id or case_id in used_ids:
            continue

        if case_type and case.get("case_type", "").lower() != case_type.lower():
            continue
        if sub_case_type and case.get("sub_case_type", "").lower() != sub_case_type.lower():
            continue

        used_ids.add(case_id)
        results.append(case)
        if len(results) >= top_k:
            break
            
    return results

def reciprocal_rank_fusion(faiss_results: list, bm25_results: list, k: int = 60) -> list:
    rrf_scores = {}
    
    def process_results(results):
        for rank, case in enumerate(results):
            current_rank = rank + 1
            case_id = case.get("case_id")
            if not case_id:
                continue
                
            if case_id not in rrf_scores:
                rrf_scores[case_id] = {"score": 0.0, "case": case}
            
            rrf_scores[case_id]["score"] += 1.0 / (k + current_rank)
            
    process_results(faiss_results)
    process_results(bm25_results)
    
    sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    max_possible_score = 2.0 / (k + 1)
    
    final_results = []
    for item in sorted_rrf:
        case_copy = dict(item["case"])
        normalized_score = item["score"] / max_possible_score
        case_copy["similarity"] = round(normalized_score * 100, 2)
        final_results.append(case_copy)
        
    return final_results

def find_similar_cases(
    text,
    case_type=None,
    sub_case_type=None,
    top_k=5,
):
    _load()
    
    # 1. Fetch from FAISS
    # We fetch a large number (e.g. 50) to allow for fusion and filtering
    raw_faiss = faiss_search(text, top_k=50)
    faiss_candidates = _filter_results(raw_faiss, case_type, sub_case_type, top_k=20)
    
    # Format FAISS
    faiss_formatted = [_format_case(c, c.get("similarity", 0)) for c in faiss_candidates]
    
    # 2. Fetch from BM25
    raw_bm25 = bm25_search(text, top_k=50)
    bm25_candidates = _filter_results(raw_bm25, case_type, sub_case_type, top_k=20)
    
    # Format BM25
    bm25_formatted = [_format_case(c, c.get("similarity", 0)) for c in bm25_candidates]
    
    # 3. Apply RRF
    fused_results = reciprocal_rank_fusion(faiss_formatted, bm25_formatted, k=60)
    
    return fused_results[:top_k]