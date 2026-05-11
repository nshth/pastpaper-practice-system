"""
Run this once to extract and process the PDF(s) into units_chunks.json.
Usage:
    python extract_pdf.py --pdf1 eGr12TG_ICT.pdf --pdf2 <second_pdf_path>

Output: units_chunks.json  — list of chunks, each with:
    {
        "unit_id":    "ICT_001",
        "unit_number": 1,
        "unit_name":  "Concept of ICT",
        "competency_level": "1.1",          # sub-unit label from PDF
        "text":       "<combined syllabus + lesson guide text>",
        "keywords":   [...]                 # extracted from learning outcomes
    }

Strategy:
- One chunk per competency level (e.g. 1.1, 1.2, ..., 7.7)
- Each chunk = syllabus table content + lesson guide content combined
- Unit number derived from competency prefix (1.x -> unit 1, etc.)
- For units 8-14 (from pdf2): same logic applied
- Falls back to units.json description for any unit with no PDF coverage
"""

import re
import json
import argparse
import subprocess
from pathlib import Path


UNIT_NAMES = {
    1:  "Concept of ICT",
    2:  "Introduction to Computer",
    3:  "Data Representation",
    4:  "Fundamental of Digital Circuits",
    5:  "Computer Operating System",
    6:  "Data Communication and Networking",
    7:  "System Analysis and Design",
    8:  "Database Management",
    9:  "Programming",
    10: "Web Development",
    11: "Internet of Things",
    12: "ICT in Business",
    13: "New Trends and Future Directions of ICT",
    14: "Project",
}

# competency number -> unit number mapping
# (handles cases where competency numbers don't match unit numbers)
COMPETENCY_TO_UNIT = {i: i for i in range(1, 15)}


def pdf_to_text(pdf_path: str) -> list[str]:
    result = subprocess.run(
        ["pdftotext", pdf_path, "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.stdout.splitlines()

def extract_chunks_from_lines(lines: list[str]) -> list[dict]:
    """
    Splits extracted PDF text into per-competency-level chunks.
    Combines both the syllabus table section and the lesson guide section
    for each competency level into a single rich text block.
    """
    comp_pattern = re.compile(
        r"^Competency Level (\d+)\.(\d+)[:\s]*(.*)", re.IGNORECASE
    )

    # --- Pass 1: find all competency level boundaries ---
    boundaries = []  # (line_index, comp_level_str, heading_text)
    for i, line in enumerate(lines):
        m = comp_pattern.match(line.strip())
        if m:
            comp_major = int(m.group(1))
            comp_minor = m.group(2)
            heading = m.group(3).strip()
            comp_level = f"{comp_major}.{comp_minor}"
            boundaries.append((i, comp_level, comp_major, heading))

    # --- Pass 2: slice text between boundaries ---
    raw_chunks: dict[str, list[str]] = {}
    for idx, (start_line, comp_level, comp_major, heading) in enumerate(boundaries):
        end_line = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        chunk_lines = lines[start_line:end_line]
        text_block = "\n".join(chunk_lines).strip()

        if comp_level not in raw_chunks:
            raw_chunks[comp_level] = []
        raw_chunks[comp_level].append(text_block)

    # --- Pass 3: merge duplicate comp_level blocks (syllabus + guide) ---
    chunks = []
    for comp_level, blocks in raw_chunks.items():
        comp_major = int(comp_level.split(".")[0])
        unit_num = COMPETENCY_TO_UNIT.get(comp_major, comp_major)
        unit_name = UNIT_NAMES.get(unit_num, f"Unit {unit_num}")
        combined_text = "\n\n".join(blocks)
        keywords = extract_keywords(combined_text)

        chunks.append({
            "unit_id": f"ICT_{unit_num:03d}",
            "unit_number": unit_num,
            "unit_name": unit_name,
            "competency_level": comp_level,
            "text": combined_text,
            "keywords": keywords,
        })

    return chunks


def extract_keywords(text: str) -> list[str]:
    """
    Pull candidate keywords from learning outcome lines and content bullets.
    Simple heuristic: noun phrases after action verbs, capitalised terms.
    """
    keywords = set()

    # capitalised multi-word terms
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        kw = m.group(1).strip()
        if len(kw) > 4:
            keywords.add(kw.lower())

    # acronyms
    for m in re.finditer(r'\b([A-Z]{2,6})\b', text):
        keywords.add(m.group(1))

    # lines that look like content bullets (start with known keywords)
    topic_words = re.findall(
        r'(?:Definition|Concept|Types?|Introduction|Overview|'
        r'Characteristics?|Functions?|Structure|Protocol|Algorithm|'
        r'Architecture|Design|Analysis|Model|System|Network|Database|'
        r'Programming|Language|Web|Internet|IoT|Security)\w*',
        text, re.IGNORECASE
    )
    keywords.update(w.lower() for w in topic_words)

    return sorted(keywords)[:40]


def load_fallback_units(units_json_path: str) -> dict[int, dict]:
    if not Path(units_json_path).exists():
        return {}
    units = json.load(open(units_json_path))
    return {int(u["unit_id"].split("_")[1]): u for u in units}


def build_fallback_chunk(unit_num: int, unit_data: dict) -> dict:
    return {
        "unit_id": unit_data["unit_id"],
        "unit_number": unit_num,
        "unit_name": unit_data["unit_name"],
        "competency_level": None,
        "text": unit_data.get("combined_text", "") + " " + unit_data.get("description", ""),
        "keywords": unit_data.get("keywords", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf1", default="./data/raw/eGr12TG_ICT.pdf", help="First PDF (units 1-7)")
    parser.add_argument("--pdf2", default="./data/raw/eGr13TG_ICT.pdf", help="Second PDF (units 8-14)")
    parser.add_argument("--units_json", default="./units.json", help="Fallback units.json")
    parser.add_argument("--out", default="units_chunks.json")
    args = parser.parse_args()

    all_chunks = []
    covered_units = set()

    for pdf_path in filter(None, [args.pdf1, args.pdf2]):
        if not Path(pdf_path).exists():
            print(f"WARNING: {pdf_path} not found, skipping.")
            continue
        print(f"Processing {pdf_path}...")
        lines = pdf_to_text(pdf_path)
        chunks = extract_chunks_from_lines(lines)
        all_chunks.extend(chunks)
        covered_units.update(c["unit_number"] for c in chunks)
        print(f"  -> {len(chunks)} competency-level chunks from {len(set(c['unit_number'] for c in chunks))} units")

    # Fallback: units not covered by any PDF use units.json
    fallback = load_fallback_units(args.units_json)
    missing_units = set(range(1, 15)) - covered_units
    if missing_units:
        print(f"Falling back to units.json for units: {sorted(missing_units)}")
        for unit_num in sorted(missing_units):
            if unit_num in fallback:
                all_chunks.append(build_fallback_chunk(unit_num, fallback[unit_num]))
            else:
                print(f"  WARNING: unit {unit_num} has no data anywhere, skipping.")

    all_chunks.sort(key=lambda c: (c["unit_number"], c["competency_level"] or ""))
    json.dump(all_chunks, open(args.out, "w"), indent=2)
    print(f"\nDone. {len(all_chunks)} total chunks -> {args.out}")
    print("Unit coverage:", sorted(set(c['unit_number'] for c in all_chunks)))


if __name__ == "__main__":
    main()