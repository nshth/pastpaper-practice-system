import sqlite3

UNIT_NAMES = {
    1: ("Concept of ICT", "Fundamental concepts of data, information, and the role of technology."),
    2: ("Introduction to Computer", "Evolution of computing devices and computer architecture."),
    3: ("Data Representation", "How instructions and data are represented in computers."),
    4: ("Fundamental of Digital Circuits", "Logic gates and basic digital circuit design."),
    5: ("Computer Operating System", "OS concepts, process and memory management."),
    6: ("Data Communication and Networking", "Signals, protocols, and network architecture."),
    7: ("System Analysis and Design", "Systems concept, SSADM, and information system development."),
    8: ("Database Management", "Database concepts, SQL, and data modelling."),
    9: ("Programming", "Programming concepts, algorithms, and Python."),
    10: ("Web Development", "HTML, CSS, JavaScript, and web technologies."),
    11: ("Internet of Things", "IoT architecture, sensors, and applications."),
    12: ("ICT in Business", "ICT applications in business and e-commerce."),
    13: ("New Trends and Future Directions of ICT", "AI, cloud, cybersecurity, and emerging technologies."),
    14: ("Project", "Practical project work."),
}

def init_db(db_path: str = "pastpapers.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id        TEXT PRIMARY KEY,
            subject         TEXT NOT NULL,
            year            INTEGER NOT NULL,
            paper_number    INTEGER NOT NULL,
            source_pdf      TEXT,
            marking_pdf     TEXT
        );

        CREATE TABLE IF NOT EXISTS questions (
            question_id     TEXT PRIMARY KEY,
            paper_id        TEXT NOT NULL REFERENCES papers(paper_id),
            question_number INTEGER NOT NULL,
            text            TEXT NOT NULL,
            correct_option  TEXT,
            unit            INTEGER REFERENCES units(unit_number),
            unit_confidence REAL,
            unit_source     TEXT,
            has_image       INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS units (
            unit_number     INTEGER PRIMARY KEY,
            unit_name       TEXT NOT NULL,
            description     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_questions_paper   ON questions(paper_id);
        CREATE INDEX IF NOT EXISTS idx_questions_unit    ON questions(unit);
        CREATE INDEX IF NOT EXISTS idx_questions_paper_unit ON questions(paper_id, unit);
    """)

    cur.executemany(
        "INSERT OR IGNORE INTO units (unit_number, unit_name, description) VALUES (?, ?, ?)",
        [(k, v[0], v[1]) for k, v in UNIT_NAMES.items()]
    )

    conn.commit()
    conn.close()
    print("DB initialized.")

if __name__ == "__main__":
    init_db()