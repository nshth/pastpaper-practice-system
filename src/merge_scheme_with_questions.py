import json

with open("data/extracted/processed/2022_P1_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("data/extracted/processed/2022_P1_scheme.json", "r", encoding="utf-8") as f:
    scheme_raw = json.load(f)

scheme = {data["question_id"]: data["answers"] for _, data in scheme_raw.items()}

merged_count = 0
missing_count = 0

for question in questions:
    question_id = question["question_id"]
    
    if question_id in scheme:
        question["answer"] = scheme[question_id]
        merged_count += 1
    else:
        question["answer"] = None
        missing_count += 1
        print(f"No scheme for {question_id}")


with open("data/extracted/merged/2022_P1_QA.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"\n Merged {merged_count}/{len(questions)} questions")
if missing_count > 0:
    print(f" Missing {missing_count} answers")
print("Saved to 2022_P1_QA.json")
