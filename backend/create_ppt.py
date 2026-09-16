from pptx import Presentation
from pptx.util import Inches, Pt
import os

# Create presentation
prs = Presentation()

# Slide 1: Title Slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "Legal AI Judiciary: An End-to-End System for Case Analysis and Precedent Retrieval"
subtitle.text = "Review 1 - Capstone Project\nTeam Members: [Your Names]\nProject Guide: [Guide Name]\nDomain: AI / NLP"

def add_slide(prs, title_text, content_text):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = title_text
    
    content = slide.placeholders[1]
    tf = content.text_frame
    tf.word_wrap = True
    
    for line in content_text.strip().split('\n'):
        p = tf.add_paragraph()
        if line.startswith('* '):
            p.text = line[2:]
            p.level = 0
        elif line.startswith('  * '):
            p.text = line[4:]
            p.level = 1
        elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.'):
            p.text = line
            p.level = 0
        else:
            p.text = line
            p.level = 0

add_slide(prs, "Problem Statement", """* The Problem: The Indian judicial system suffers from a massive backlog of cases. Legal practitioners spend hundreds of hours manually reviewing unstructured legal documents.
* Current Limitations: Existing systems rely on basic keyword matching which fails to capture semantic meaning. Standard AI models struggle to identify strict boundaries of complex Indian legal entities.
* The Need: An automated, domain-specific AI system that can semantically search for case similarities and accurately extract legal metadata.""")

add_slide(prs, "Objectives", """1. Develop a Hybrid NER Pipeline: Build a Named Entity Recognition model combining InLegalBERT with a BiLSTM-CRF layer.
2. Build a Semantic Case Retrieval Engine: Chunk unstructured judgments and utilize Sentence-Transformers for hierarchical similarity search.
3. Deploy an End-to-End Platform: Transition from isolated scripts to a production-ready application (FastAPI + React).""")

add_slide(prs, "Literature Review", """* InLegalBERT (Kalamkar et al., 2022): Pre-trained on Indian court cases; outperforms generic BERT models on legal tasks.
* ILDC Corpus (Malik et al., 2021): Highlighted the extreme length and complexity of Indian judgments, justifying the need for hierarchical processing.
* Legal NER (LegalEval 2023): Demonstrated that hybrid architectures handle the nested, complex dependencies of Indian legal phrasing.
* Dense Embeddings (LePaRD, 2023): Proved that Sentence-Transformers map legal passages into dense vector spaces for accurate retrieval.""")

add_slide(prs, "Research Gap", """1. Boundary-Aware Extraction: Standard InLegalBERT classifiers fail at strict boundary detection. A BiLSTM-CRF decoding layer is required but highly under-explored.
2. Monolithic Document Processing: Current precedent retrieval systems compare entire cases at once without granular case/sub-case detection before computing similarity.
3. Absence of Production Pipelines: Most research exists as fragmented academic scripts. There is a severe lack of open-source, full-stack reference architectures.""")

add_slide(prs, "Proposed Architecture & System Design", """1. Data Ingestion Layer:
  * PDF Upload & OCR Text Extraction
  * Structural Chunking (Summary, Facts, Judgment)
2. AI Processing Layer:
  * Hybrid NER: Text -> InLegalBERT -> BiLSTM -> CRF
  * Similarity Engine: Chunked Text -> Sentence-Transformers
3. Application Layer:
  * Database: MongoDB & embeddings.npy
  * Backend: FastAPI (Python)
  * Frontend: React (Vite)""")

add_slide(prs, "Methodology", """1. Data Preparation: Download and preprocess authentic Indian datasets (OpenNyAI NER corpus and 7,875 real Indian Court Judgments).
2. Model Training: Train the Hybrid NER model utilizing PyTorch. Optimize BiLSTM hidden states and CRF transition matrices.
3. Vectorization: Compute cosine similarity on Sentence-Transformer embeddings for precedent retrieval.
4. System Integration: Connect trained models and vector databases to FastAPI endpoints.
5. Validation: Test semantic search accuracy and NER boundary precision.""")

add_slide(prs, "Tools & Datasets", """* AI/ML Technologies: PyTorch, Hugging Face Transformers, TorchCRF, Sentence-Transformers
* Backend & Frontend: Python, FastAPI, React, Node.js, Vite
* Database: MongoDB, FAISS/NumPy for vectors
* Datasets (Indian Context):
  * OpenNyAI InLegalNER Dataset: 15,000+ examples for training the Hybrid NER.
  * Joel Niklaus / Indian Legal Cases: 7,875 real, structured Indian court judgments.""")

add_slide(prs, "Work Plan (7th & 8th Semesters)", """* Review 1 (Sept 2026): Finalized problem, literature review, architecture, and 20% implementation (Backend API setup, Indian datasets).
* Review 2 (Oct 2026): 60% implementation; Hybrid NER model fully trained and integrated into the pipeline.
* Review 3 (Oct 2026): 100% Phase 1 completion; End-to-end working system and paper drafted.
* Review 4 & 5 (8th Sem): Advanced Version enhancements, final validation, and comparative performance analysis.""")

add_slide(prs, "20% Implementation Demonstration", """1. The Backend Server: FastAPI server running with active endpoints.
2. The New Datasets: The database containing the 7,875 real Indian judgments.
3. The Similarity Engine: Successful generation of 384-dimensional dense vectors (embeddings.npy) for semantic search.
4. The Training Pipeline: The Hybrid NER training process active on the OpenNyAI dataset.""")

output_path = os.path.join(os.getcwd(), "Capstone_Review_1.pptx")
prs.save(output_path)
print(f"Presentation saved successfully to: {output_path}")
