import os
import re
import base64
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv(".env")
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


print(f"API Key loaded: {api_key[:10]}..." if api_key else "API Key NOT found")
if not api_key:
    print("MISTRAL_API_KEY not set")
    exit(1)

pdf_path = "data/raw/pastpaper/24-pp-ict-en-part-1.pdf"
try:
    uploaded_pdf = client.files.upload(
        file={"file_name": os.path.basename(pdf_path), "content": open(pdf_path, "rb")},
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        table_format="html",
        include_image_base64=True
    )
except Exception as e:
    print(f"Error: {e}")
    exit(1)

processed_pages = []
for page in ocr_response.pages:
    md = page.markdown
    
    if hasattr(page, 'images') and page.images:
        for img in page.images:
            img_data = img.image_base64
            if "," in img_data:
                img_data = img_data.split(",")[1]
            image_bytes = base64.b64decode(img_data)
            os.makedirs("data/extracted/assets/images", exist_ok=True)
            with open(f"images/{img.id}", "wb") as f:
                f.write(image_bytes)
            md = md.replace(f"![{img.id}]({img.id})", f"![{img.id}](images/{img.id})")
    
    if hasattr(page, 'tables') and page.tables:
        for tbl in page.tables:
            md = md.replace(f"[{tbl.id}]({tbl.id})", f"\n{tbl.content}\n")
    
    processed_pages.append(md)

full_markdown = "\n\n---\n\n".join(processed_pages)

questions = []
lines = full_markdown.split('\n')
current_q = None
current_text = []
current_images = []
current_tables = []

img_pattern = re.compile(r'!\[(.*?)\]\((images/.*?)\)')

for line in lines:

    line = line.strip()

    img_match = img_pattern.search(line)
    if img_match:
        current_images.append(img_match.group(2))

    match = re.match(r'^(?:Q)?(\d+)[.\s]+(.*)', line)

    if match:

        if current_q:
            questions.append({
                "question_id": f"2024_P1_Q{str(current_q).zfill(2)}",
                "year": 2024,
                "paper": 1,
                "q_num": current_q,
                "text": ' '.join(current_text).strip(),
                "marks": 2,
                "images": current_images,
                "tables": current_tables
            })

        current_q = int(match.group(1))
        current_text = [match.group(2)]
        current_images = []
        current_tables = []

    elif current_q and line:
        current_text.append(line)

print(f"Extracted {len(questions)} questions")


os.makedirs("data/extracted/processed", exist_ok=True)

with open("data/extracted/processed/2024_P1_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2)

with open("ocr_output_mapped.md", "w", encoding="utf-8") as f:
    f.write(full_markdown)

print("Saved: questions, schemes, markdown")
print("Files created:")
print(f"  - data/extracted/2024_P1_questions.json ({len(questions)} Q)")
print("  - ocr_output_mapped.md")