from PyPDF2 import PdfReader

def load_pdf_chunks(path, chunk_size=500):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
