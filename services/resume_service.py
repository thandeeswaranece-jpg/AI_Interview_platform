"""
services/resume_service.py

PDF resume text extraction, split out of main.py so it can raise clear,
catchable errors instead of letting a bad PDF crash the request.
"""

import fitz  # PyMuPDF


class ResumeExtractionError(Exception):
    pass


def extract_resume_text(pdf_bytes: bytes) -> str:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ResumeExtractionError(f"Could not open the file as a PDF: {e}")

    text = ""
    try:
        for page in pdf:
            text += page.get_text()
    finally:
        pdf.close()

    text = text.strip()
    if len(text) < 30:
        raise ResumeExtractionError(
            "Extracted resume text is empty or too short. The PDF might be a "
            "scanned image with no selectable text (this project does not do OCR)."
        )
    return text