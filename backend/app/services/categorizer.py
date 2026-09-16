import re


def categorize_case(text: str) -> dict:
    """
    Classify an uploaded legal judgment into
    case type and sub-case type.
    """

    text_lower = text.lower()

    # Criminal Law
    criminal_words = [
        "murder",
        "robbery",
        "theft",
        "stolen",
        "accused",
        "criminal",
        "fir",
        "ipc",
        "penal code",
        "conviction",
        "bail",
        "prosecution",
        "offence",
        "offense",
    ]

    criminal_score = sum(
        1 for word in criminal_words
        if word in text_lower
    )

    # Contract Law
    contract_words = [
        "contract",
        "agreement",
        "employment",
        "employee",
        "employer",
        "breach of contract",
        "contractual obligation",
    ]

    contract_score = sum(
        1 for word in contract_words
        if word in text_lower
    )

    # Civil Law
    civil_words = [
        "property dispute",
        "ownership",
        "partition",
        "civil suit",
        "plaintiff",
        "defendant",
        "land",
        "property",
    ]

    civil_score = sum(
        1 for word in civil_words
        if word in text_lower
    )

    scores = {
        "Criminal Law": criminal_score,
        "Contract Law": contract_score,
        "Civil Law": civil_score,
    }

    case_type = max(scores, key=scores.get)

# SUB-CASE CLASSIFICATION

    if case_type == "Criminal Law":

        if any(
            word in text_lower
            for word in [
                "murder",
                "homicide",
                "302 ipc",
                "section 302",
            ]
        ):
            sub_case_type = "Murder"

        elif any(
            word in text_lower
            for word in [
                "theft",
                "stolen",
                "stealing",
                "stole",
            ]
        ):
            sub_case_type = "Theft"

        elif any(
            word in text_lower
            for word in [
                "fraud",
                "cheating",
                "dishonest",
                "financial fraud",
            ]
        ):
            sub_case_type = "Fraud"

        elif any(
            word in text_lower
            for word in [
                "bail",
                "anticipatory bail",
                "regular bail",
            ]
        ):
            sub_case_type = "Bail"

        elif any(
            word in text_lower
            for word in [
                "rape",
                "sexual assault",
                "sexual offence",
                "sexual offense",
            ]
        ):
            sub_case_type = "Sexual Offence"

        else:
            sub_case_type = "Other Criminal"

    elif case_type == "Contract Law":

        if "employment" in text_lower or "employee" in text_lower:
            sub_case_type = "Employment"

        elif "breach of contract" in text_lower:
            sub_case_type = "Breach of Contract"

        else:
            sub_case_type = "Contract"

    elif case_type == "Civil Law":

        if "partition" in text_lower:
            sub_case_type = "Partition"

        elif "property" in text_lower or "ownership" in text_lower:
            sub_case_type = "Property"

        else:
            sub_case_type = "Civil Dispute"

    else:
        case_type = "Legal Case"
        sub_case_type = "General"

    return {
        "case_type": case_type,
        "sub_case_type": sub_case_type,
    }