# Physics AI Tutor

Physics AI Tutor is a subject-specific AI assistant designed to answer physics-related questions using OpenAI’s API.  
The repository contains two versions:

1) **tutor_v1.4.py** – stable version for direct text-based Q&A.  
2) **tutor_v1.6.py** – extended version with OCR support (Tesseract + Poppler) for processing questions from images or PDFs. While functional, it requires additional setup and is less suited for quick demonstration.

---
## Tech Stack

- **Programming Language:** Python 3.8+
- **AI Model:** OpenAI API (GPT-based)
- **Environment Management:** python-dotenv
- **OCR (optional, v1.6):** Tesseract OCR, Poppler
- **Data Handling:** OS, pathlib, and standard Python libraries
- **Version Control:** Git + GitHub


## Features

- Physics-focused Q&A powered by OpenAI.
- Simple retrieval mechanism for referencing local knowledge sources.
- Optional OCR integration for image/PDF input (v1.6).
- 3 modes for students usage. Doubt clarification. Learning Assistant and Quiz Generator

---

## Setup and Usage

Clone the repository and ensure Python 3.8+ is installed.  
Install dependencies from either `requirements_v1.4.txt` or `requirements_v1.6.txt` depending on the version you want to run.  
Create a `.env` file in the project directory containing your `OPENAI_API_KEY` (required for both versions). For v1.6, also install Tesseract and Poppler locally, adding their paths to the `.env` file.  
Once environment variables are configured, run the chosen version with:

```bash
python tutor_v1.4.py
# or
python tutor_v1.6.py
