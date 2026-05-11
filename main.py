import json

def main():
    questions = json.load(open("questions.json", encoding="utf-8"))
    for q in questions:
        q_num = q['q_num'] 
        unit = q['unit'] 
        unit_confidence = q['unit_confidence']

        print(f"Q:{q_num} -> unit:{unit}")
        
if __name__ == "__main__":
    main()
