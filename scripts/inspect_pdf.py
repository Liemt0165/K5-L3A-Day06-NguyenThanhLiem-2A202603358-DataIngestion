import os
import sys
from pypdf import PdfReader

pdf_path = "data/AI-Agents-Chapter3.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

reader = PdfReader(pdf_path)
total_pages = len(reader.pages)
print(f"Total physical pages: {total_pages}")

valid_pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    text_clean = text.strip()
    char_count = len(text_clean)
    if char_count > 100:
        valid_pages.append((i + 1, char_count, text_clean[:120].replace('\n', ' ')))

print(f"Pages with text (>100 chars): {len(valid_pages)}")
print("\nSample first 5 pages with content:")
for p_num, count, preview in valid_pages[:5]:
    print(f"Page {p_num:2d} ({count:4d} chars): {preview}...")
