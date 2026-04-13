import os
import re
import sys
from mistralai.client import Mistral
from dotenv import load_dotenv
import json

load_dotenv(".env")
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

scheme_pdf_path = "data/raw/schemes/24-ppms-ict-en-part-1.pdf"

try:
    uploaded_pdf = client.files.upload(
        file={"file_name": os.path.basename(scheme_pdf_path), "content": open(scheme_pdf_path, "rb")},
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        table_format="html",
        
    )
    
    processed_pages = []
    for page in ocr_response.pages:

        md = page.markdown
        
        if hasattr(page, 'tables') and page.tables:
            for tbl in page.tables:
                md = md.replace(f"[{tbl.id}]({tbl.id})", f"\n{tbl.content}\n")
        
        processed_pages.append(md)

    full_markdown = "\n\n---\n\n".join(processed_pages)
    table_match = re.search(r'<table>(.*?)</table>', full_markdown, re.DOTALL)
    if not table_match:
        print(" No table found")
        exit(1)

    table_html = table_match.group(1)

    # Parse table rows
    rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
    scheme_data = {}

    # Skip header row (first row)
    for row in rows[1:]:
        cells = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
        
        # Process pairs: Q number + answer
        for i in range(0, len(cells), 2):
            if i + 1 < len(cells):
                q_num_raw = cells[i].strip()
                answer_raw = cells[i + 1].strip()
                
                # Extract number from "01." format
                q_match = re.search(r'(\d+)', q_num_raw)
                if not q_match:
                    continue
                
                q_num = int(q_match.group(1))
                
                # Extract answers - handle ALL cases
                answers = []
                
                # Remove HTML breaks first
                clean_answer = answer_raw.replace('<br/>', ' ')
                
                # Find all numbers in the answer
                nums = re.findall(r'\d+', clean_answer)
                answers = [int(n) for n in nums]
                
                # Special case: if "ALL" or "S - ALL" found, mark as special
                if 'ALL' in clean_answer:
                    answers = ['ALL']  # Mark it as special
                elif not answers:  # No numbers found, keep raw
                    answers = None
                
                scheme_data[q_num] = {
                    "question_id": f"2024_P1_Q{str(q_num).zfill(2)}",
                    "answers": answers,
                    "raw": answer_raw  # Debug: keep raw for inspection
                }

    # Print to verify
    print(f"✅ Extracted {len(scheme_data)} answers")
    for q_num in sorted(scheme_data.keys()):
        print(f"  Q{q_num:02d}: {scheme_data[q_num]['answers']}")

    # Save for next step (remove raw field)
    clean_data = {k: {'question_id': v['question_id'], 'answers': v['answers']} 
                for k, v in scheme_data.items()}

    with open("scheme_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)

    print("\n✅ Saved to scheme_data.json")

except Exception as e:
    print(f"Error: {e}")
    exit(1)
