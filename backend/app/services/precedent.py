import re
from typing import List, Dict


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    text = _clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


# ---------------------------------------------------------
# Legal issue detection
# ---------------------------------------------------------

LEGAL_ISSUE_KEYWORDS = {
    "evidence": [
        "evidence",
        "proof",
        "testimony",
        "witness",
        "statement",
        "confession",
        "corroboration",
        "eyewitness",
    ],

    "bail": [
        "bail",
        "anticipatory bail",
        "regular bail",
        "custody",
        "bail application",
    ],

    "murder": [
        "murder",
        "homicide",
        "death",
        "killing",
        "section 302",
    ],

    "contract": [
        "contract",
        "agreement",
        "breach",
        "consideration",
        "specific performance",
    ],

    "property": [
        "property",
        "land",
        "possession",
        "ownership",
        "title",
    ],

    "negligence": [
        "negligence",
        "negligent",
        "duty of care",
        "damages",
    ],

    "appeal": [
        "appeal",
        "appellant",
        "appellate",
    ],
}


def _detect_issues(text: str) -> List[str]:

    text_lower = text.lower()

    detected = []

    for issue, keywords in LEGAL_ISSUE_KEYWORDS.items():

        matches = sum(
            1
            for keyword in keywords
            if keyword.lower() in text_lower
        )

        if matches > 0:
            detected.append(issue)

    return detected


# ---------------------------------------------------------
# Principle extraction
# ---------------------------------------------------------

PRINCIPLE_MARKERS = [
    "held that",
    "we hold",
    "the court held",
    "it was held",
    "the court observed",
    "the court finds",
    "the court found",
    "the principle",
    "it is settled law",
    "settled position",
    "law is well settled",
    "therefore",
    "accordingly",
]


def _extract_principles(text: str) -> List[str]:

    sentences = _split_sentences(text)

    principles = []

    for sentence in sentences:

        lower = sentence.lower()

        if any(
            marker in lower
            for marker in PRINCIPLE_MARKERS
        ):
            principles.append(sentence)

    return principles


# ---------------------------------------------------------
# Relevance analysis
# ---------------------------------------------------------

def _calculate_issue_overlap(
    current_issues: List[str],
    previous_issues: List[str],
) -> float:

    if not current_issues or not previous_issues:
        return 0.0

    current = set(current_issues)
    previous = set(previous_issues)

    intersection = current.intersection(previous)

    union = current.union(previous)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def _relevance_level(score: float) -> str:

    if score >= 75:
        return "High"

    if score >= 50:
        return "Medium"

    return "Low"


# ---------------------------------------------------------
# Analyse one precedent
# ---------------------------------------------------------

def analyze_precedent(
    current_text: str,
    previous_case: Dict,
) -> Dict:

    current_text = _clean_text(current_text)

    previous_text = " ".join([
        previous_case.get("title", ""),
        previous_case.get("summary", ""),
        previous_case.get("facts", ""),
        previous_case.get("judgment", ""),
    ])

    previous_text = _clean_text(previous_text)

    current_issues = _detect_issues(
        current_text
    )

    previous_issues = _detect_issues(
        previous_text
    )

    issue_overlap = _calculate_issue_overlap(
        current_issues,
        previous_issues,
    )

    similarity = float(
        previous_case.get(
            "similarity",
            0
        )
    )

    # -----------------------------------------------------
    # Combine semantic similarity and legal issue overlap
    # -----------------------------------------------------

    relevance_score = (
        similarity * 0.70
        + issue_overlap * 100 * 0.30
    )

    relevance_score = round(
        relevance_score,
        2
    )

    level = _relevance_level(
        relevance_score
    )

    principles = _extract_principles(
        previous_text
    )

    # Keep the output manageable
    principles = principles[:3]

    if principles:

        legal_principle = principles[0]

    else:

        legal_principle = (
            "No explicit legal principle could be "
            "reliably extracted from the available "
            "case text."
        )

    shared_issues = sorted(
        set(current_issues)
        .intersection(
            previous_issues
        )
    )

    if shared_issues:

        why_relevant = (
            "This judgment may be relevant because "
            "both cases involve the legal issue(s): "
            + ", ".join(shared_issues)
            + "."
        )

    else:

        why_relevant = (
            "This judgment was retrieved because "
            "of its semantic similarity to the "
            "uploaded case, but no strong shared "
            "legal issue was automatically identified."
        )

    return {
        "case_id": previous_case.get(
            "case_id"
        ),

        "title": previous_case.get(
            "title",
            "Indian Legal Judgment"
        ),

        "year": previous_case.get(
            "year"
        ),

        "similarity": similarity,

        "precedent_relevance": relevance_score,

        "relevance_level": level,

        "legal_issues": previous_issues,

        "shared_legal_issues": shared_issues,

        "legal_principle": legal_principle,

        "principles": principles,

        "why_relevant": why_relevant,

        "case_type": previous_case.get(
            "case_type",
            ""
        ),

        "sub_case_type": previous_case.get(
            "sub_case_type",
            ""
        ),
    }


# ---------------------------------------------------------
# Analyse all retrieved similar cases
# ---------------------------------------------------------

def analyze_precedents(
    current_text: str,
    similar_cases: List[Dict],
    top_k: int = 5,
) -> List[Dict]:

    if not similar_cases:
        return []

    analyzed = []

    for case in similar_cases:

        try:

            result = analyze_precedent(
                current_text,
                case,
            )

            analyzed.append(result)

        except Exception as error:

            print(
                "PRECEDENT ANALYSIS ERROR:",
                repr(error)
            )

    analyzed.sort(
        key=lambda x: (
            x.get(
                "precedent_relevance",
                0
            ),
            x.get(
                "similarity",
                0
            ),
        ),
        reverse=True,
    )

    return analyzed[:top_k]