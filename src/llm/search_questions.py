import sqlite3

DB = "pastpapers.db"


def search_questions(unit: int | None = None, year: int | None = None, limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT q.question_id, q.question_number, q.text, q.correct_option,
               q.unit, u.unit_name, p.year, p.paper_number
        FROM questions q
        JOIN papers p ON q.paper_id = p.paper_id
        LEFT JOIN units u ON q.unit = u.unit_number
        WHERE 1=1
    """
    params = []

    if unit:
        query += " AND q.unit = ?"
        params.append(unit)
    if year:
        query += " AND p.year = ?"
        params.append(year)

    query += " ORDER BY p.year DESC, q.question_number ASC LIMIT ?"
    params.append(limit)

    rows = cur.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_unit_descriptions() -> str:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    rows = cur.execute("SELECT unit_number, unit_name, description FROM units ORDER BY unit_number").fetchall()
    conn.close()
    return "\n".join(f"{r[0]}. {r[1]}: {r[2]}" for r in rows)