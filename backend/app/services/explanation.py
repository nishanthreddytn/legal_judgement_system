import re


def _clean_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    replacements = {
        "!PC": "IPC",
        "u/ss.": "under Sections",
        "u/s.": "under Section",
        "fi1ll": "full",
        "jiwn": "from",
        "Furthe1;": "Further;",
        "011": "on",
        "p ractice": "practice",
        "corroborati ve": "corroborative",
        "corroborati on": "corroboration",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _remove_junk(text):
    """
    Remove common OCR/header noise.
    """

    if not text:
        return ""

    # Remove repeated page numbers
    text = re.sub(r"\b\d{1,4}\b(?=\s+[A-Z]{3,})", " ", text)

    # Remove obvious judge/header fragments
    text = re.sub(
        r"\b[A-Z][A-Z\s.,'-]{5,}\b(?=\s+(?:JJ\.|J\.)\b)",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _get_case_section(text):
    """
    Get a useful beginning portion of the judgment.
    """

    text = _clean_text(text)

    markers = [
        "Evidence Act",
        "Indian Evidence Act",
        "Penal Code",
        "IPC",
        "Confessional statement",
        "Murder",
        "Robbery",
        "Conviction",
    ]

    positions = []

    lower_text = text.lower()

    for marker in markers:
        pos = lower_text.find(marker.lower())

        if pos != -1:
            positions.append(pos)

    if not positions:
        return text[:5000]

    start = max(0, min(positions) - 150)

    return text[start:start + 5000]


def _extract_first_meaningful_part(
    text,
    keywords,
    max_length=500
):
    """
    Extract a reasonably meaningful piece of text.

    This is mainly used internally and is NOT used
    for the citizen evidence explanation.
    """

    text = _clean_text(text)

    lower = text.lower()

    for keyword in keywords:

        pos = lower.find(keyword.lower())

        if pos == -1:
            continue

        start = max(0, pos - 100)

        end = min(
            len(text),
            pos + max_length
        )

        result = text[start:end].strip()

        if len(result) > 80:

            match = re.search(
                r"[.!?]\s+",
                result[100:]
            )

            if match:
                result = result[:100 + match.end()]

            return result.strip()

    return ""


def _extract_judgment_decision(text):

    text = _clean_text(text)

    lower = text.lower()

    decisions = [
        "dismissing the appeal",
        "dismissed the appeal",
        "appeal is dismissed",
        "appeal was dismissed",
        "appeal dismissed",
        "affirming the conviction",
        "affirmed the conviction",
        "conviction was affirmed",
        "conviction is affirmed",
        "conviction upheld",
        "conviction was upheld",
        "conviction is upheld",
        "acquitted",
        "acquittal",
    ]

    for phrase in decisions:

        pos = lower.find(phrase)

        if pos != -1:

            start = max(0, pos - 250)

            end = min(
                len(text),
                pos + 500
            )

            result = text[start:end].strip()

            return result

    held = re.search(
        r"\bHeld\s*:\s*(.{0,700})",
        text,
        re.IGNORECASE
    )

    if held:
        return held.group(1).strip()

    return (
        "The court reached its decision after considering "
        "the evidence and circumstances of the case."
    )


def _extract_what_happened(text):

    text = _clean_text(text)

    keywords = [
        "murder of",
        "murder",
        "robbery",
        "theft",
        "incident",
        "prosecution",
        "accused",
    ]

    result = _extract_first_meaningful_part(
        text,
        keywords,
        max_length=600
    )

    if result:

        result = _remove_junk(result)

        return result

    return (
        "The case involved allegations against the accused. "
        "The court examined what happened and considered the "
        "evidence presented by both sides."
    )


def _extract_summary(
    case_type,
    sub_case_type,
    text
):
    """
    Generate a simple case summary.
    """

    case_type = (case_type or "").strip()
    sub_case_type = (sub_case_type or "").strip()

    text_lower = text.lower()

    if sub_case_type:

        if "murder" in sub_case_type.lower():

            return (
                "This case is about a murder allegation. "
                "The court examined what happened, the information "
                "given by witnesses and other evidence before deciding "
                "whether the accused was responsible."
            )

        if "robbery" in sub_case_type.lower():

            return (
                "This case is about a robbery allegation. "
                "The court examined what happened and considered "
                "the evidence before deciding whether the accused "
                "was responsible."
            )

        if "theft" in sub_case_type.lower():

            return (
                "This case is about a theft allegation. "
                "The court examined the events and evidence before "
                "deciding whether the accused was responsible."
            )

    if "murder" in text_lower:

        return (
            "This case concerns a murder allegation. "
            "The court examined what happened and considered "
            "the evidence before reaching its decision."
        )

    if "robbery" in text_lower:

        return (
            "This case concerns a robbery allegation. "
            "The court examined the events and evidence before "
            "reaching its decision."
        )

    if case_type:

        return (
            f"This case concerns a {case_type.lower()} matter. "
            "The court examined the events, evidence and arguments "
            "before reaching its decision."
        )

    return (
        "This case concerns a dispute that was considered by the court. "
        "The court examined the events, evidence and arguments before "
        "reaching its decision."
    )


def _extract_evidence(text):
    """
    Identify the important types of evidence mentioned in the judgment
    and explain them in simple English.

    Do NOT return raw OCR text.
    """

    text = _clean_text(text)

    lower = text.lower()

    evidence_points = []

    # Accomplice / approver evidence
    if (
        "accomplice" in lower
        or "approver" in lower
        or "accomplice witness" in lower
    ):
        evidence_points.append(
            "the statement of a person who was involved in the crime"
        )

    # Confession
    if (
        "confessional statement" in lower
        or "confession" in lower
        or "admitted involvement" in lower
    ):
        evidence_points.append(
            "a statement in which a person admitted involvement in the crime"
        )

    # Witness evidence
    if (
        "witness" in lower
        or "testimony" in lower
        or "eye witness" in lower
    ):
        evidence_points.append(
            "statements or testimony given by witnesses"
        )

    # Recovery of objects
    if (
        "recovered" in lower
        or "recovery" in lower
        or "recovered at the instance" in lower
    ):
        evidence_points.append(
            "objects or other material recovered during the investigation"
        )

    # Medical evidence
    if (
        "medical evidence" in lower
        or "post-mortem" in lower
        or "postmortem" in lower
        or "medical examination" in lower
    ):
        evidence_points.append(
            "medical and post-mortem findings"
        )

    # Police report / FIR
    if (
        "first information report" in lower
        or "fir" in lower
    ):
        evidence_points.append(
            "the initial report given to the police"
        )

    # Forensic evidence
    if (
        "forensic" in lower
        or "fingerprint" in lower
        or "dna evidence" in lower
        or "dna" in lower
    ):
        evidence_points.append(
            "forensic evidence such as fingerprints or DNA findings"
        )

    # Documents
    if (
        "documentary evidence" in lower
        or "documents" in lower
    ):
        evidence_points.append(
            "documents and records connected with the case"
        )

    # Avoid duplicate concepts
    unique_points = []

    for point in evidence_points:

        if point not in unique_points:
            unique_points.append(point)

    if not unique_points:

        return (
            "The court considered the information available in the "
            "case, including the statements, facts and other material "
            "presented by the parties."
        )

    if len(unique_points) == 1:

        return (
            "The court mainly considered "
            + unique_points[0]
            + ". It also examined whether this evidence was reliable "
              "and supported the accused person's involvement."
        )

    if len(unique_points) == 2:

        return (
            "The court considered "
            + unique_points[0]
            + " and "
            + unique_points[1]
            + ". It examined whether these pieces of evidence "
              "supported the case against the accused."
        )

    # More than two
    first_points = unique_points[:3]

    evidence_text = ", ".join(first_points[:-1])

    evidence_text += " and " + first_points[-1]

    return (
        "The court considered "
        + evidence_text
        + ". It examined whether these pieces of evidence "
          "supported the case and connected the accused with "
          "the alleged offence."
    )


def _extract_legal_issue(text):

    text = _clean_text(text)

    lower = text.lower()

    # Accomplice-related case
    if (
        "accomplice" in lower
        or "approver" in lower
    ):

        return (
            "The main question was whether the statement given by "
            "a person involved in the crime was reliable and whether "
            "other evidence sufficiently supported that statement "
            "to establish the accused person's involvement."
        )

    # Confession-related case
    if "confession" in lower:

        return (
            "The main question was whether the statement admitting "
            "involvement in the crime could be relied upon and whether "
            "the other evidence supported it."
        )

    # General evidence issue
    if "evidence" in lower:

        return (
            "The main question was whether the evidence presented "
            "to the court was sufficient and reliable enough to "
            "establish the accused person's involvement."
        )

    return (
        "The main question was whether the information and evidence "
        "presented to the court were sufficient to establish what "
        "happened and who was responsible."
    )


def _simple_decision(text):

    text = _clean_text(text)

    lower = text.lower()

    # Appeal dismissed
    if (
        "dismissing the appeal" in lower
        or "dismissed the appeal" in lower
        or "appeal is dismissed" in lower
        or "appeal was dismissed" in lower
        or "appeal dismissed" in lower
    ):

        return (
            "The higher court rejected the appeal and did not change "
            "the earlier decision."
        )

    # Conviction affirmed
    if (
        "affirmed the conviction" in lower
        or "conviction was affirmed" in lower
        or "conviction is affirmed" in lower
        or "conviction upheld" in lower
        or "conviction was upheld" in lower
        or "conviction is upheld" in lower
    ):

        return (
            "The higher court agreed with the earlier decision and "
            "left the conviction unchanged."
        )

    # Acquittal
    if "acquitted" in lower:

        return (
            "The court found that the accused should not be held "
            "guilty on the basis of the evidence considered."
        )

    # Acquittal noun
    if "acquittal" in lower:

        return (
            "The court upheld the decision to acquit the accused."
        )

    return (
        "The court reached its decision after considering the "
        "evidence and circumstances recorded in the judgment."
    )


def explain(
    case_type,
    sub_case_type,
    text
):
    """
    Main explanation function.

    Returns the five sections required by the citizen page.
    """

    text = _clean_text(text)

    summary = _extract_summary(
        case_type,
        sub_case_type,
        text
    )

    what_happened = _extract_what_happened(text)

    evidence_considered = _extract_evidence(text)

    main_question = _extract_legal_issue(text)

    court_decision = _simple_decision(text)

    return {

        "summary": summary,

        "what_happened": what_happened,

        "evidence_considered": evidence_considered,

        "legal_issue": main_question,

        "court_decision": court_decision,

        "important_note": (
            "This explanation is generated from the uploaded judgment. "
            "The original judgment should be checked for complete details."
        ),

        "disclaimer": (
            "This system provides an explanation of the uploaded "
            "document for research and understanding. It does not "
            "predict or replace a court's decision."
        )
    }