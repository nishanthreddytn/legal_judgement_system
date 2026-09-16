from pathlib import Path
import json
import random

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

criminal_theft_cases = [
    {
        "title": "Rajan Sharma v. State",
        "summary": "The case concerned the alleged theft of valuable property from a residential premises. The prosecution claimed that the accused was involved in taking the property without the owner's permission.",
        "facts": "The prosecution relied on statements from witnesses, ownership records and evidence recovered during the investigation. The defence disputed the connection between the accused and the missing property.",
        "judgment": "The court examined the witness statements, ownership records and recovery evidence before deciding whether the prosecution had established the accused's involvement beyond the required standard of proof."
    },
    {
        "title": "Mohan Verma v. State",
        "summary": "The accused was alleged to have unlawfully taken property belonging to another person. The case mainly concerned whether the available evidence connected the accused with the missing property.",
        "facts": "The prosecution presented evidence regarding the disappearance of the property and material recovered during the investigation. Witnesses also described the circumstances in which the property went missing.",
        "judgment": "The court considered the reliability of the witnesses and the recovery evidence and then determined whether the evidence was sufficient to establish the accused's involvement."
    },
    {
        "title": "Vikram Patel v. State",
        "summary": "The case concerned allegations that property belonging to another person had been taken without permission. The prosecution relied on evidence gathered during the investigation.",
        "facts": "Witnesses gave evidence about the missing property and the investigation traced some of the property to the accused. The defence challenged the connection between the accused and the recovered material.",
        "judgment": "The court examined the recovery evidence, witness testimony and surrounding circumstances before reaching its decision on the allegations."
    },
    {
        "title": "Arjun Mehta v. State",
        "summary": "The accused was prosecuted in connection with the alleged theft of property. The central question was whether the evidence sufficiently connected the accused to the property that had been reported missing.",
        "facts": "The prosecution relied on witness statements, records concerning ownership and evidence obtained during the investigation. The defence questioned the reliability and interpretation of that evidence.",
        "judgment": "After examining the evidence presented by both sides, the court reached its decision on whether the allegations against the accused had been established."
    },
    {
        "title": "Suresh Rao v. State",
        "summary": "The case involved allegations of theft from a private residence. The prosecution claimed that the accused had a connection with the stolen property.",
        "facts": "The investigation involved statements from witnesses and examination of property recovered after the alleged incident. The accused disputed the prosecution's version of events.",
        "judgment": "The court assessed the evidence as a whole, including witness testimony and recovery material, before deciding the appeal."
    }
]

criminal_fraud_cases = [
    {
        "title": "Ravi Kumar v. State",
        "summary": "The case concerned allegations of financial fraud involving disputed transactions.",
        "facts": "The prosecution relied on transaction records, documents and statements of persons connected with the disputed dealings.",
        "judgment": "The court examined the documentary and transaction evidence before deciding whether dishonest conduct had been established."
    }
]

civil_property_cases = [
    {
        "title": "Sharma v. Singh",
        "summary": "The dispute concerned competing claims over ownership of property.",
        "facts": "Both parties relied on ownership documents and other records relating to the property.",
        "judgment": "The court examined the title documents and surrounding evidence before deciding the competing claims."
    }
]

contract_employment_cases = [
    {
        "title": "Mehta v. National Industries",
        "summary": "The dispute concerned an employment-related decision and the rights of the employee.",
        "facts": "The parties relied on the employment agreement and records maintained during the employment relationship.",
        "judgment": "The court examined the contractual obligations and evidence produced by both parties before reaching its decision."
    }
]


cases = []

# ---------------------------------------------------------
# Criminal Theft - 50 cases
# ---------------------------------------------------------

for i in range(1, 51):

    base = criminal_theft_cases[(i - 1) % len(criminal_theft_cases)]

    year = 2015 + ((i - 1) % 10)

    cases.append({
        "case_id": f"CRIM-THEFT-{i}",
        "title": base["title"],
        "year": year,
        "case_type": "Criminal Law",
        "sub_case_type": "Theft",
        "summary": base["summary"],
        "facts": base["facts"],
        "judgment": base["judgment"]
    })


# ---------------------------------------------------------
# Criminal Fraud - 50 cases
# ---------------------------------------------------------

for i in range(51, 101):

    base = criminal_fraud_cases[(i - 51) % len(criminal_fraud_cases)]

    year = 2015 + ((i - 51) % 10)

    cases.append({
        "case_id": f"CRIM-FRAUD-{i}",
        "title": base["title"],
        "year": year,
        "case_type": "Criminal Law",
        "sub_case_type": "Fraud",
        "summary": base["summary"],
        "facts": base["facts"],
        "judgment": base["judgment"]
    })


# ---------------------------------------------------------
# Civil Property - 50 cases
# ---------------------------------------------------------

for i in range(101, 151):

    base = civil_property_cases[(i - 101) % len(civil_property_cases)]

    year = 2015 + ((i - 101) % 10)

    cases.append({
        "case_id": f"CIV-PROP-{i}",
        "title": base["title"],
        "year": year,
        "case_type": "Civil Law",
        "sub_case_type": "Property",
        "summary": base["summary"],
        "facts": base["facts"],
        "judgment": base["judgment"]
    })


# ---------------------------------------------------------
# Contract Employment - 50 cases
# ---------------------------------------------------------

for i in range(151, 201):

    base = contract_employment_cases[(i - 151) % len(contract_employment_cases)]

    year = 2015 + ((i - 151) % 10)

    cases.append({
        "case_id": f"CON-EMP-{i}",
        "title": base["title"],
        "year": year,
        "case_type": "Contract Law",
        "sub_case_type": "Employment",
        "summary": base["summary"],
        "facts": base["facts"],
        "judgment": base["judgment"]
    })


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = OUT_DIR / "cases.json"

output_file.write_text(
    json.dumps(cases, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Created {len(cases)} cases.")
print(f"Saved to: {output_file}")