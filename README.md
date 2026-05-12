# 📚 Past Paper Practice System – AI-Powered Study Assistant

An intelligent past paper practice platform that helps students learn faster by combining **hybrid retrieval (BM25 + embeddings)** with **LLM-powered explanations and marking scheme matching**.

It transforms static past papers into an interactive learning system with instant feedback, model answers, and smart question understanding.

---

## 🚀 What it does

- **Question Practice**  
  Students attempt past paper questions classified by unit, year and topic through a simple interface 

- **Instant explanation**  
  Retrieves relevant marking schemes and generates AI-powered model answers.

- **Hybrid Search (BM25 + Embeddings)**  
  Combines keyword-based + semantic search for accurate retrieval.

- **Intent Classification**  
  Detects what the student wants:
  - Search based on unit, year 
  - Explain questions 
  - General 

- **OCR Pipeline**  
  Extracts questions and marking schemes from PDF past papers automatically.

---

## 📖 Why this exists

Past papers are usually static, repetitive, and boring. This system turns them into an interactive learning loop where students understand why an answer is correct instead of memorizing it. It also lets them filter questions by unit or topic across different years to analyze exam patterns, something that would normally take a lot of time if done manually.

---

## 🛠️ Installation

### Install dependencies
```bash
pip install -r requirements.txt
````

or (if using uv)

```bash
uv sync
```

---

### Environment setup

```bash
cp .env.example .env
```

Add your API keys and config:

* Groq api
* HF api
* Mistral api

---

## ▶️ Run the application

```bash
streamlit run app.py
```
or 

simply go the URL
https://pastpaper-practice-system.streamlit.app/

---

## 🧠 Required Services

* **LLM Provider (Groq / OpenAI)**
  Used for answer generation and intent classification.

* **Chroma DB**
  Vector database for semantic retrieval.

* **SQLite**
  Stores questions, papers, and mappings.

---

## 📖 Usage

### 👩‍🎓 Students

* Open the app
* Attempt past paper questions
* Get:

  * Model answers
  * Marking scheme references
  * AI explanations

---

### 🧑‍🏫 Teachers / Admins

* Run `pdf_splitter.py` to split needed content from pdf
* Run `mistral_ocr.py` to extract data from PDFs
* Manage data via:

  * `questions.json`
  * `pastpapers.db`
  * /data

---

## 🏗️ Project Structure

```text
pastpaper-practice-system/
├── src/               # Core modules (retrieval, OCR, classification, llm)
├── db/                # SQLite schema & migrations
├── chroma_db/        # Vector database storage
├── data/             # Raw PDFs + extracted content
├── app.py            # Main streamlit application entry
├── mistral_ocr.py    # OCR + extraction logic for paper
├── ocr_ppms.py       # OCR + extraction logic for schemes
├── requirements.txt  # Dependencies
├── pastpapers.db     # SQLite database
```

---

## 🤝 Contributing

Pull requests are welcome.

Before contributing:

* Ensure OCR extraction works correctly
* Test hybrid retrieval performance
* Validate question-to-scheme mapping accuracy

---

## ⚡ Notes

This system is modular by design:

* Retrieval layer can be swapped (BM25 / embeddings / hybrid)
* LLM provider is configurable
* OCR pipeline can evolve independently

---

Built to make past paper practice less painful and slightly more intelligent.

```
