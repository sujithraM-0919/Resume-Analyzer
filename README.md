
---

# Smart Resume Matcher

**AI-powered Resume ↔ Job Description Alignment Engine**

A full-stack NLP project that evaluates how well a candidate’s resume aligns with a job description using statistical text modeling and skill-based analysis.

This project is designed to simulate a lightweight Applicant Tracking System (ATS), focusing on semantic similarity and skill coverage to generate actionable insights.

---

## Key Features

* Match score (0–100) using TF-IDF and cosine similarity
* Skill extraction engine based on a curated taxonomy
* Identification of matched and missing skills
* Hireability classification with reasoning
* REST API built with Flask
* React-based frontend for interaction
* CLI support for quick testing

---

## Architecture Overview

```
Resume + Job Description
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Skill Extraction
        ↓
Scoring + Insights
        ↓
API / UI Output
```

---

## Project Structure

```
smart-resume-matcher/
├── backend/
│   ├── matcher.py          # Core NLP pipeline
│   ├── app.py              # Flask API
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.jsx         # React UI
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_job_description.txt
├── docs/
│   └── architecture.md
└── README.md
```

---

## Tech Stack

* Python
* Scikit-learn
* Flask
* React.js
* Regex-based NLP

---

## Running the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

API runs at:
[http://localhost:5000](http://localhost:5000)

---

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at:
[http://localhost:3000](http://localhost:3000)

---

### CLI Mode

```bash
python matcher.py sample_data/sample_resume.txt sample_data/sample_job_description.txt
```

---

## API Reference

### POST /api/match

**Request**

```json
{
  "resume": "Full resume text...",
  "job_description": "Job description text..."
}
```

**Response**

```json
{
  "match_score": 82.5,
  "hireability": "High",
  "hireability_reason": "Strong semantic alignment and good skill coverage",
  "matched_skills": ["python", "machine learning", "aws"],
  "missing_skills": ["pytorch"],
  "recommendation": "Strong match. Adding missing skills can further improve alignment."
}
```

---

## Methodology

**Text Preprocessing**
Lowercasing, punctuation removal, and basic normalization.

**TF-IDF Vectorization**
Transforms text into weighted vectors based on term importance.

**Cosine Similarity**
Measures similarity between resume and job description vectors and scales it to a percentage score.

**Skill Extraction**
Rule-based matching against a predefined skill set across domains.

**Scoring Logic**
Combines similarity score with skill overlap to produce final output.

---

## Design Decisions

* TF-IDF chosen initially for simplicity, speed, and interpretability
* Rule-based skill extraction for transparency and control
* API-first structure to separate backend and frontend concerns

---

## Future Improvements

* Replace TF-IDF with embedding models (e.g., SBERT)
* Add PDF/DOCX resume parsing
* Expand and refine skill taxonomy
* Deploy as a hosted web application
* Add user history and dashboard features

---

## What This Project Demonstrates

* Practical NLP application in a real-world use case
* End-to-end system design (backend + frontend)
* Understanding of similarity modeling and feature engineering
* Clean and modular project structure

---

## Limitations

* Rule-based skill extraction may miss context-specific skills
* TF-IDF has limited semantic understanding
* Results depend on input quality

---


