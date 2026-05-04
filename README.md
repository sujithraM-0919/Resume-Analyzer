# Smart Resume Matcher AI

Production-grade AI tool that analyzes resume–job description fit using NLP techniques including TF-IDF, cosine similarity, and skill taxonomy extraction.

## Folder Structure

```
smart-resume-matcher/
├── backend/
│   ├── matcher.py          # Core NLP engine (TF-IDF + skill extraction)
│   ├── app.py              # Flask REST API
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── src/
│       └── App.jsx         # React frontend (Claude AI-powered version)
├── sample_data/
│   ├── sample_resume.txt
│   └── sample_job_description.txt
├── docs/
│   └── architecture.md
└── README.md
```

## Quick Start

### Option A: AI-powered Web UI (Claude API)
Open `frontend/src/App.jsx` as a React artifact in Claude.ai — it runs fully in-browser using the Anthropic API.

### Option B: Python Backend + Flask API

**1. Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**2. Start the API server**
```bash
python app.py
# Server runs at http://localhost:5000
```

**3. Test the API**
```bash
curl -X POST http://localhost:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Python developer with 5 years experience in machine learning...",
    "job_description": "Looking for Python ML engineer with TensorFlow experience..."
  }'
```

**4. CLI usage**
```bash
python matcher.py sample_data/sample_resume.txt sample_data/sample_job_description.txt
```

### Option C: React Frontend + Flask API

```bash
# Terminal 1 – Backend
cd backend && python app.py

# Terminal 2 – Frontend
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

## API Reference

### POST /api/match
**Request body:**
```json
{
  "resume": "Full resume text...",
  "job_description": "Job description text..."
}
```

**Response:**
```json
{
  "match_score": 82.5,
  "hireability": "High",
  "hireability_reason": "Strong semantic alignment with extensive skill overlap.",
  "matched_skills": ["python", "machine learning", "docker", "aws"],
  "missing_skills": ["pytorch", "spark"],
  "bonus_skills": ["react", "postgresql"],
  "shared_keywords": ["pipeline", "deployment", "model", "data"],
  "recommendation": "Excellent match! Your resume aligns strongly...",
  "stats": {
    "resume_word_count": 412,
    "jd_word_count": 287,
    "resume_skills_count": 22,
    "jd_skills_count": 18
  }
}
```

## NLP Methodology

1. **Text Preprocessing**: Lowercasing, punctuation removal, stopword filtering
2. **TF-IDF Vectorization**: Term Frequency × Inverse Document Frequency weighting
3. **Cosine Similarity**: Measures angular distance between document vectors (0–1 scale, normalized to 0–100)
4. **Skill Taxonomy Extraction**: Regex pattern matching against 80+ curated skills across 6 categories
5. **Hireability Scoring**: Composite of similarity score + skill coverage ratio

## Extending with SBERT

To use sentence-transformer embeddings instead of TF-IDF:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def match_with_embeddings(resume_text, job_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)
    score = float(util.cos_sim(emb_resume, emb_job)[0][0])
    return round(score * 100, 1)
```

Embedding-based matching captures semantic meaning (e.g., "LLM engineer" ≈ "large language model developer"), while TF-IDF is faster and requires no model download.
