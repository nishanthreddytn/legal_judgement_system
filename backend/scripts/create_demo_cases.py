from pathlib import Path
import json
import random

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

theft_cases = [
    {
        "title": "Mohan Verma v. State",
        "summary": "The accused was alleged to have taken jewellery from a residence without the owner's permission.",
        "facts": "The prosecution relied on the owner's complaint, statements from neighbours and recovery of jewellery from a location connected with the accused.",
        "judgment": "The court examined the witness statements and recovery evidence and considered whether they sufficiently connected the accused with the theft."
    },
    {
        "title": "Rajan Sharma v. State",
        "summary": "The case concerned the alleged theft of electronic equipment from a commercial premises.",
        "facts": "The prosecution relied on CCTV evidence, statements from employees and recovery of some of the missing equipment.",
        "judgment": "The court assessed the reliability of the CCTV material and witness evidence before deciding whether the accused was responsible."
    },
    {
        "title": "Vikram Patel v. State",
        "summary": "The accused was charged with stealing valuable property from a private residence.",
        "facts": "The investigation involved statements from the residents, recovery records and evidence showing the accused's presence near the property.",
        "judgment": "The court considered the surrounding circumstances and recovery material before reaching its decision."
    },
    {
        "title": "Arjun Mehta v. State",
        "summary": "The prosecution alleged that the accused removed cash and personal belongings from a locked premises.",
        "facts": "The prosecution presented evidence about access to the premises, missing property and statements from persons who were present around the relevant time.",
        "judgment": "The court examined whether the evidence established the accused's involvement beyond the available circumstances."
    },
    {
        "title": "Suresh Rao v. State",
        "summary": "The case involved allegations that household valuables were unlawfully taken from the complainant.",
        "facts": "Witnesses described the missing valuables and the investigation traced part of the property to a person associated with the accused.",
        "judgment": "The court examined ownership, recovery and witness evidence while determining the accused's involvement."
    },
    {
        "title": "Kiran Das v. State",
        "summary": "The accused was alleged to have participated in the removal of goods from a warehouse.",
        "facts": "The prosecution relied on warehouse records, employee testimony and evidence concerning transportation of the missing goods.",
        "judgment": "The court considered the documentary and witness evidence and determined whether the prosecution had established the allegation."
    },
    {
        "title": "Naveen Kumar v. State",
        "summary": "The prosecution alleged that the accused unlawfully took a mobile device belonging to another person.",
        "facts": "The investigation considered ownership records, location information and recovery of the device.",
        "judgment": "The court assessed the recovery evidence and surrounding circumstances before deciding the appeal."
    },
    {
        "title": "Rahul Singh v. State",
        "summary": "The case concerned the alleged theft of money from a business establishment.",
        "facts": "The prosecution relied on transaction records, employee statements and evidence concerning access to the business premises.",
        "judgment": "The court examined whether the documentary and oral evidence sufficiently established the accused's participation."
    },
    {
        "title": "Amit Joshi v. State",
        "summary": "The accused was alleged to have removed agricultural equipment belonging to another person.",
        "facts": "The prosecution produced ownership documents, statements from local witnesses and evidence regarding recovery of the equipment.",
        "judgment": "The court considered the ownership documents and recovery evidence before reaching its conclusion."
    },
    {
        "title": "Deepak Yadav v. State",
        "summary": "The prosecution alleged theft of valuables during the absence of the property owner.",
        "facts": "Witnesses described the circumstances surrounding the disappearance and investigators recovered some items connected with the allegation.",
        "judgment": "The court evaluated the consistency of the witness testimony and the significance of the recovered property."
    },
]

fraud_cases = [
    {
        "title": "Mehta v. State",
        "summary": "The accused was alleged to have obtained money through misleading financial representations.",
        "facts": "The prosecution relied on transaction records, communications and statements from persons who transferred money.",
        "judgment": "The court examined whether the evidence established dishonest conduct and the accused's role in the transactions."
    },
    {
        "title": "Rao v. State",
        "summary": "The case concerned allegations of financial deception involving several transactions.",
        "facts": "Bank records and documentary evidence were examined along with statements from the affected parties.",
        "judgment": "The court considered whether the evidence established fraudulent conduct."
    },
    {
        "title": "Patel v. State",
        "summary": "The accused was alleged to have used false information to obtain financial benefit.",
        "facts": "The investigation relied on documents, account records and statements from the persons affected.",
        "judgment": "The court evaluated the documentary evidence and surrounding circumstances."
    },
]

property_cases = [
    {
        "title": "Sharma v. Singh",
        "summary": "The dispute concerned competing claims over ownership of a parcel of land.",
        "facts": "Both parties relied on title documents, property records and previous transactions concerning the land.",
        "judgment": "The court examined the ownership documents and competing claims before reaching its decision."
    },
    {
        "title": "Reddy v. Kumar",
        "summary": "The case concerned a disagreement regarding possession and ownership of property.",
        "facts": "The parties produced registration documents and evidence concerning possession of the property.",
        "judgment": "The court assessed the documentary evidence and competing claims."
    },
    {
        "title": "Patel v. Mehta",
        "summary": "The dispute involved ownership of residential property.",
        "facts": "The parties relied on sale documents, registration records and evidence regarding possession.",
        "judgment": "The court considered the documentary record before deciding the competing property claims."
    },
]

employment_cases = [
    {
        "title": "Employee v. National Industries",
        "summary": "The employee challenged a decision concerning termination of employment.",
        "facts": "The dispute involved the employment agreement, workplace records and communications between the employee and employer.",
        "judgment": "The court examined the contractual terms and evidence produced by both parties."
    },
    {
        "title": "Kumar v. Tech Industries",
        "summary": "The employee disputed the employer's decision regarding termination and unpaid benefits.",
        "facts": "The parties relied on employment records, salary documents and the terms of the employment agreement.",
        "judgment": "The court considered the contractual obligations and employment records."
    },
    {
        "title": "Rao v. Manufacturing Ltd.",
        "summary": "The dispute concerned an employee's claim arising from termination of service.",
        "facts": "The case involved the employment contract, internal records and correspondence between the parties.",
        "judgment": "The court examined the agreement and evidence before reaching its decision."
    },
]


def build_cases():

    cases = []

    case_groups = [
        ("CRIM-THEFT", "Criminal Law", "Theft", theft_cases),
        ("CRIM-FRAUD", "Criminal Law", "Fraud", fraud_cases),
        ("CIV-PROP", "Civil Law", "Property", property_cases),
        ("CON-EMP", "Contract Law", "Employment", employment_cases),
    ]

    counters = {
        "CRIM-THEFT": 0,
        "CRIM-FRAUD": 0,
        "CIV-PROP": 0,
        "CON-EMP": 0,
    }

    # Create 50 cases for each category.
    for prefix, case_type, sub_case_type, templates in case_groups:

        for i in range(50):

            counters[prefix] += 1

            template = templates[i % len(templates)]

            variation = i // len(templates)

            title = template["title"]

            if variation > 0:
                title = f"{title} ({variation + 1})"

            case = {
                "case_id": f"{prefix}-{counters[prefix]}",
                "title": title,
                "year": 2015 + ((i * 3 + variation) % 11),
                "case_type": case_type,
                "sub_case_type": sub_case_type,
                "summary": template["summary"],
                "facts": template["facts"],
                "judgment": template["judgment"],
            }

            cases.append(case)

    return cases


cases = build_cases()

(OUT / "cases.json").write_text(
    json.dumps(cases, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Created {len(cases)} cases.")