import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all readable text from a PDF.
    """

    pages = []

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

    return "\n\n".join(pages).strip()