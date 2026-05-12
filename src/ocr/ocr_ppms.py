import os
import base64
import json
from io import BytesIO
from pdf2image import convert_from_path
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv(".env")

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# model = "mistral-small-latest"

# pdf_path = "data/raw/schemes/p1/22-ppms-ict-en-part-1.pdf"

# pages = convert_from_path(pdf_path, first_page=1, last_page=1)
# buffered = BytesIO()
# pages[0].save(buffered, format="JPEG")
# base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

# prompt = """
# Extract the MCQ answers from this marking scheme image into a structured JSON format.
# The output must be a JSON object where the keys are question numbers (1-50) 
# and the values are the correct option numbers.

# Requirements:
# 1. If a question has multiple answers (e.g., '4,5'), represent them as an array: [4, 5].
# 2. If the answer is 'ALL', represent it as ["ALL"].
# 3. Ensure all 50 questions are present.

# Return ONLY the raw JSON.
# """

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
#         ]
#     }
# ]

# response = client.chat.complete(model=model, messages=messages)

raw_content = """{
  "1": 3,
  "2": 5,
  "3": 4,
  "4": 1,
  "5": 4,
  "6": 5,
  "7": 3,
  "8": 3,
  "9": 2,
  "10": 5,
  "11": [2, 4],
  "12": 1,
  "13": 4,
  "14": 4,
  "15": 1,
  "16": 5,
  "17": 5,
  "18": 3,
  "19": 4,
  "20": 4,
  "21": 5,
  "22": 3,
  "23": 1,
  "24": 4,
  "25": 1,
  "26": 2,
  "27": 1,
  "28": 2,
  "29": 5,
  "30": 3,
  "31": 1,
  "32": 3,
  "33": 4,
  "34": 3,
  "35": 1,
  "36": 1,
  "37": 2,
  "38": 2,
  "39": 1,
  "40": 3,
  "41": 1,
  "42": 5,
  "43": 2,
  "44": 2,
  "45": 5,
  "46": 2,
  "47": 3,
  "48": 4,
  "49": 5,
  "50": 4
}"""
clean_json_str = raw_content.replace("```json", "").replace("```", "").strip()

try:
    initial_data = json.loads(clean_json_str)
    
    restructured_data = {}
    
    for q_num_str in sorted(initial_data.keys(), key=lambda x: int(x)):
        val = initial_data[q_num_str]
        
        if isinstance(val, list):
            answers = val
        else:
            answers = [val]
            
        q_num_padded = q_num_str.zfill(2)
        restructured_data[q_num_str] = {
            "question_id": f"2022_P1_Q{q_num_padded}",
            "answers": answers
        }

    with open("data/extracted/processed/2022_P1_scheme.json", "w", encoding="utf-8") as f:
        json.dump(restructured_data, f, indent=2)

    print(f"Restructured {len(restructured_data)} questions. Saved to scheme_data_vision.json")

except json.JSONDecodeError as e:
    print(f"Failed to parse LLM output: {e}")