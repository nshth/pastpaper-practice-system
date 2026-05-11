import json
import sqlite3
import argparse


def resolve_unit(raw) -> int | None:
    if raw is None:
        return None
    
    # If it's a list (e.g., [1]), take the first element
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    
    # Try to convert to integer; if it's "ALL" or invalid text, return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None

def ingest(args):
    conn = sqlite3.connect("pastpapers.db")
    cur = conn.cursor()

    cur.execute(
        "INSERT OR IGNORE INTO papers (paper_id, subject, year, paper_number, source_pdf, marking_pdf) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (args.paper_id, args.subject, args.year, args.paper_num, args.source_pdf, args.marking_pdf),
    )

    questions = json.load(open(args.questions))
    inserted = 0
    for q in questions:
        question_id = f"{args.paper_id}_Q{q['q_num']:02d}"
        unit = resolve_unit(q.get("unit"))
        cur.execute(
            "INSERT OR REPLACE INTO questions "
            "(question_id, paper_id, question_number, text, correct_option, unit, unit_confidence, unit_source, has_image) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question_id,
                args.paper_id,
                q["q_num"],
                q["text"],
                resolve_unit(q.get("answer")), # Use resolve_unit here as well
                unit,
                q.get("unit_confidence"),
                q.get("unit_source", "retrieval"),
                int(bool(q.get("has_image", False))),
            ),
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} questions for {args.paper_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",   required=True)
    parser.add_argument("--paper_id",   required=True)
    parser.add_argument("--year",        type=int, required=True)
    parser.add_argument("--paper_num",   type=int, required=True)
    parser.add_argument("--subject",     default="ICT")
    parser.add_argument("--source_pdf",  default=None)
    parser.add_argument("--marking_pdf", default=None)
    ingest(parser.parse_args())