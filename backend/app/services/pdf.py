import pdfplumber
import cv2
import numpy as np
import easyocr
import fitz
from PIL import Image

import os

# Ensure the models directory exists
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(model_dir, exist_ok=True)

reader = easyocr.Reader(['en'], model_storage_directory=model_dir, user_network_directory=model_dir)

def preprocess_image_for_ocr(image):
    # Convert PIL Image to cv2 format (numpy array)
    img = np.array(image)
    # Convert RGB to BGR
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_text_from_pdf(file_path: str):
    """
    Extract all readable text from a PDF.
    """
    pages = []
    metadata = {"methods": []}

    pdf_document = fitz.open(file_path)
    pdf_images = []
    for page in pdf_document:
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_images.append(img)

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text and len(text.strip()) >= 50:
                pages.append(text.strip())
                metadata["methods"].append("pdfplumber")
            else:
                if i < len(pdf_images):
                    image = pdf_images[i]
                    processed_img = preprocess_image_for_ocr(image)
                    ocr_text = " ".join(reader.readtext(processed_img, detail=0))
                    
                    pages.append(ocr_text.strip())
                    metadata["methods"].append("ocr")
                else:
                    if text:
                        pages.append(text.strip())
                        metadata["methods"].append("pdfplumber")
                    else:
                        pages.append("")
                        metadata["methods"].append("unknown")

    return "\n\n".join(pages).strip(), metadata