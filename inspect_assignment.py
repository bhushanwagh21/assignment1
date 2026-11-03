from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(r"c:\Users\DELL\Downloads\assignment1\UST - Assignment - 1 - Diabetes Prediction using ML.pdf")
print('exists', pdf_path.exists(), 'size', pdf_path.stat().st_size if pdf_path.exists() else 'missing')
reader = PdfReader(str(pdf_path))
print('pages', len(reader.pages))
for i, page in enumerate(reader.pages[:5], 1):
    text = page.extract_text() or ''
    print(f'--- PAGE {i} START ---')
    print(text[:5000])
    print(f'--- PAGE {i} END ---')
