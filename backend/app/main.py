from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from app.routes.auth import router as auth_router

from app.services.pdf import extract_text_from_pdf
from app.services.categorizer import categorize_case
from app.services.explanation import explain
from app.services.similarity import find_similar_cases
from app.services.precedent import analyze_precedents

app = FastAPI(title="Legal AI Judiciary System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Legal AI Judiciary Backend Running"
    }


@app.post("/upload")
async def upload_case(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_pdf(str(file_path))

        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from the PDF."
            )

# CASE CLASSIFICATION

        classification = categorize_case(text)

        case_type = classification.get(
            "case_type",
            "Criminal Law"
        )

        sub_case_type = classification.get(
            "sub_case_type",
            "Other Criminal"
        )

# EXPLANATION

        explanation = explain(
            case_type,
            sub_case_type,
            text
        )

# REAL CASE SIMILARITY SEARCH

        results = find_similar_cases(
            text=text,
            case_type=case_type,
            sub_case_type=sub_case_type,
            top_k=5
        )

# PRECEDENT ANALYSIS

        precedents = analyze_precedents(
            current_text=text,
            similar_cases=results,
            top_k=5
        )

# RESPONSE

        return {
            "case_type": case_type,
            "sub_case_type": sub_case_type,
            "explanation": explanation,
            "results": results,
            "precedents": precedents,
            "document_name": file.filename
        }

    except HTTPException:
        raise

    except Exception as e:
        print("UPLOAD ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"Case analysis failed: {str(e)}"
        )