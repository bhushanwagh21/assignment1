from pathlib import Path
from pypdf import PdfReader

base = Path(r'c:\Users\DELL\Downloads\assignment1')
pdf_path = next(p for p in base.iterdir() if p.suffix.lower() == '.pdf')
print('pdf_path', repr(str(pdf_path.name)))
reader = PdfReader(str(pdf_path))
print('pages', len(reader.pages))
for i, page in enumerate(reader.pages[:10], 1):
    text = page.extract_text() or ''
    print(f'--- PAGE {i} START ---')
    print(text[:5000])
    print(f'--- PAGE {i} END ---')
