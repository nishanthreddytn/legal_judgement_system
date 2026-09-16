from datasets import load_dataset
from pathlib import Path
import json
import re

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading legal judgment dataset...")
ds = load_dataset("joelniklaus/legal_case_document_summarization")

cases = []
seen = set()

def clean(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_title(text, index):
    text = clean(text)

    # Try to find a case title from the beginning of the judgment
    patterns = [
        r"([A-Z][A-Za-z .'-]+ v\.? [A-Z][A-Za-z .'-]+)",
        r"([A-Z][A-Za-z .'-]+ Vs\.? [A-Z][A-Za-z .'-]+)",
        r"([A-Z][A-Za-z .'-]+ versus [A-Z][A-Za-z .'-]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text[:3000], re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            if 5 < len(title) < 150:
                return title

    return f"Indian Legal Judgment {index}"

def extract_year(text):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text[:10000])
    if years:
        # Prefer the latest plausible year
        return int(years[-1])
    return None

def classify(text):
    t = text.lower()

    criminal_words = [
        "murder", "robbery", "theft", "burglary", "rape",
        "criminal", "accused", "ipc", "penal code", "conviction",
        "bail", "fir", "offence", "offense"
    ]

    civil_words = [
        "property", "civil suit", "plaintiff", "defendant",
        "ownership", "partition", "injunction"
    ]

    contract_words = [
        "contract", "agreement", "employment", "breach",
        "specific performance"
    ]

    if any(x in t for x in criminal_words):
        case_type = "Criminal Law"

        if any(x in t for x in ["theft", "stolen", "stole", "robbery"]):
            sub = "Theft"
        elif any(x in t for x in ["fraud", "cheating", "dishonest"]):
            sub = "Fraud"
        elif any(x in t for x in ["murder", "302 ipc", "homicide"]):
            sub = "Murder"
        elif any(x in t for x in ["rape", "sexual assault"]):
            sub = "Sexual Offence"
        elif any(x in t for x in ["bail", "anticipatory bail"]):
            sub = "Bail"
        else:
            sub = "Other Criminal"

    elif any(x in t for x in contract_words):
        case_type = "Contract Law"

        if "employment" in t:
            sub = "Employment"
        elif "breach" in t:
            sub = "Breach of Contract"
        else:
            sub = "Contract"

    elif any(x in t for x in civil_words):
        case_type = "Civil Law"

        if "property" in t or "ownership" in t:
            sub = "Property"
        elif "partition" in t:
            sub = "Partition"
        else:
            sub = "Civil Dispute"

    else:
        case_type = "Legal Case"
        sub = "General"

    return case_type, sub

def make_details(text):
    text = clean(text)

    if len(text) > 3500:
        return text[:3500] + "..."

    return text

def make_summary(dataset_summary, judgement):
    summary = clean(dataset_summary)

    if summary:
        return summary

    text = clean(judgement)

    if len(text) > 1200:
        return text[:1200] + "..."

    return text

def make_judgment(text):
    text = clean(text)

    lower = text.lower()

    markers = [
        "held that",
        "held:",
        "judgment",
        "accordingly",
        "therefore",
        "appeal is dismissed",
        "appeal dismissed",
        "appeal is allowed",
        "appeal allowed",
        "petition is dismissed",
        "petition dismissed",
    ]

    positions = [lower.find(m) for m in markers if lower.find(m) >= 0]

    if positions:
        start = min(positions)
        result = text[start:start + 1800]
    else:
        result = text[-1800:]

    return result.strip()

counter = 1

for split in ["train", "test"]:
    print(f"Processing {split}...")

    for row in ds[split]:
        judgement = clean(row.get("judgement", ""))
        summary = clean(row.get("summary", ""))

        if len(judgement) < 200:
            continue

        # Deduplicate using judgment text
        fingerprint = judgement[:500].lower()

        if fingerprint in seen:
            continue

        seen.add(fingerprint)

        title = make_title(judgement, counter)
        year = extract_year(judgement)

        case_type, sub_case_type = classify(judgement)

        cases.append({
            "case_id": f"IND-JUDG-{counter:05d}",
            "title": title,
            "year": year,
            "case_type": case_type,
            "sub_case_type": sub_case_type,
            "summary": make_summary(summary, judgement),
            "facts": make_details(judgement),
            "judgment": make_judgment(judgement),
            "source": "joelniklaus/legal_case_document_summarization"
        })

        counter += 1

print(f"\nCreated {len(cases)} real judgment records.")

output = OUT / "cases.json"

output.write_text(
    json.dumps(cases, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Saved: {output}")