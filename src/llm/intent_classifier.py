import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

_client = Groq(api_key=os.environ.get("GROQ_API"))

SYSTEM = """You are an intent classifier for a Grade 12/13 ICT past paper practice system.

SYLLABUS UNITS:
1. Concept of ICT (data, information, ICT role, data lifecycle)
2. Introduction to Computer (hardware, motherboard, CPU, RAM, ROM, storage, components)
3. Data Representation (binary, ASCII, number systems, encoding, unicode, hexadecimal)
4. Fundamental of Digital Circuits (logic gates, boolean algebra, circuits)
5. Computer Operating System (OS, operating system, process management, memory management, file system, boot)
6. Data Communication and Networking (networking, network, protocols, TCP/IP, OSI, LAN, WAN, router, switch, DHCP, DNS)
7. System Analysis and Design (SDLC, SSADM, DFD, systems analysis, flowchart, UML)
8. Database Management (database, SQL, ER diagram, normalization, DBMS, queries, tables)
9. Programming (python, algorithm, pseudocode, coding, loops, functions, programming)
10. Web Development (HTML, CSS, JavaScript, web, HTTP, browser)
11. Internet of Things (IoT, sensors, smart devices, embedded)
12. ICT in Business (e-commerce, MIS, business systems, ERP)
13. New Trends and Future Directions of ICT (AI, cloud, cybersecurity, blockchain, big data)
14. Project

Classify the message into one of: search, explain, general.
- search: user wants to find/list questions (mentions a topic, unit, subject area, or year)
- explain: user wants an explanation of a concept, question, or answer
- general: study advice, tips, anything else

Map topic names and abbreviations to the correct unit number.
Examples: "os" -> 5, "networking" -> 6, "db/database/sql" -> 8, "binary" -> 3, "logic gates" -> 4

Reply ONLY with valid JSON, no markdown, no explanation:
{"intent": "search", "unit": <int or null>, "year": <int or null>}
{"intent": "explain", "question_text": "<question or topic to explain>"}
{"intent": "general"}"""


def detect_intent(user_message: str) -> dict:
    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=80,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"intent": "general"}