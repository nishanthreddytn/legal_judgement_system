from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[2]

CASES_FILE = BASE / "data" / "processed" / "cases.json"
EMBEDDINGS_FILE = BASE / "data" / "embeddings" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_cases = None
_embeddings = None
_model = None


def _load():
    global _cases, _embeddings, _model

    if _cases is None:
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            _cases = json.load(f)

    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_FILE)

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)


def _case_text(case):
    return " ".join([
        case.get("title", ""),
        case.get("summary", ""),
        case.get("facts", ""),
        case.get("judgment", ""),
    ])


def find_similar_cases(
    text,
    case_type=None,
    sub_case_type=None,
    top_k=5,
):
    _load()

    query = _model.encode(
        [text],
        normalize_embeddings=True
    )[0]

    scores = np.dot(_embeddings, query)

    candidates = []

    for index, case in enumerate(_cases):

        # Never return the same case twice
        case_id = case.get("case_id", "")

        if not case_id:
            continue

        # Apply case-type filtering
        if case_type:
            if case.get("case_type", "").lower() != case_type.lower():
                continue

        # Apply sub-case filtering
        if sub_case_type:
            if case.get("sub_case_type", "").lower() != sub_case_type.lower():
                continue

        candidates.append(
            (
                float(scores[index]),
                index,
                case
            )
        )

    # Highest similarity first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    used_ids = set()

    for score, index, case in candidates:

        case_id = case["case_id"]

        if case_id in used_ids:
            continue

        used_ids.add(case_id)

        results.append({
            "case_id": case_id,
            "title": case.get(
                "title",
                "Indian Legal Judgment"
            ),
            "year": case.get("year"),
            "case_type": case.get(
                "case_type",
                ""
            ),
            "sub_case_type": case.get(
                "sub_case_type",
                ""
            ),
            "similarity": round(
                max(0.0, score) * 100,
                2
            ),
            "summary": case.get(
                "summary",
                ""
            ),
            "facts": case.get(
                "facts",
                ""
            ),
            "judgment": case.get(
                "judgment",
                ""
            ),
        })

        if len(results) >= top_k:
            break

    return results