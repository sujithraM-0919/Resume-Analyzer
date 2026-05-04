"""
Smart Resume Matcher AI - Core NLP Engine
Uses TF-IDF + Cosine Similarity + Keyword Extraction
"""

import re
import json
import math
from collections import Counter
from typing import Dict, List, Tuple


# ─── Skill taxonomy ────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "programming": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "ruby", "scala", "kotlin", "swift", "php", "r", "matlab", "sql", "bash",
        "shell", "powershell", "html", "css", "sass", "graphql"
    ],
    "frameworks": [
        "react", "angular", "vue", "nextjs", "nuxtjs", "svelte", "django", "flask",
        "fastapi", "spring", "springboot", "express", "nodejs", "rails", "laravel",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "pandas", "numpy", "scipy", "huggingface", "langchain"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "github actions", "circleci", "prometheus", "grafana",
        "elasticsearch", "kafka", "rabbitmq", "redis", "nginx", "linux"
    ],
    "databases": [
        "postgresql", "mysql", "sqlite", "mongodb", "cassandra", "dynamodb",
        "bigquery", "snowflake", "redshift", "neo4j", "pinecone", "weaviate"
    ],
    "data_ai": [
        "machine learning", "deep learning", "nlp", "computer vision", "llm",
        "rag", "fine-tuning", "data engineering", "etl", "spark", "airflow",
        "dbt", "tableau", "power bi", "data pipeline", "feature engineering",
        "model deployment", "mlops", "a/b testing", "statistics"
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving", "agile",
        "scrum", "project management", "mentoring", "stakeholder management",
        "cross-functional", "collaboration", "presentation"
    ]
}

ALL_SKILLS = [skill for skills in SKILL_TAXONOMY.values() for skill in skills]


# ─── Text utilities ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "it", "as",
    "our", "your", "their", "its", "my", "his", "her", "we're", "you're",
    "experience", "work", "working", "looking", "seeking", "strong", "good",
    "excellent", "ability", "skills", "knowledge", "understanding", "year",
    "years", "background", "proven", "demonstrated", "must", "required",
    "preferred", "using", "use", "used", "plus", "also", "including"
}


def extract_terms(text: str) -> List[str]:
    tokens = tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


# ─── TF-IDF ─────────────────────────────────────────────────────────────────────
def compute_tf(tokens: List[str]) -> Dict[str, float]:
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: freq / total for word, freq in count.items()}


def compute_tfidf_vector(text: str, corpus_idf: Dict[str, float] = None) -> Dict[str, float]:
    tokens = extract_terms(text)
    tf = compute_tf(tokens)
    if corpus_idf:
        return {word: tf_val * corpus_idf.get(word, 1.0) for word, tf_val in tf.items()}
    return tf


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Skill extraction ───────────────────────────────────────────────────────────
def extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))


def get_skill_category(skill: str) -> str:
    for category, skills in SKILL_TAXONOMY.items():
        if skill in skills:
            return category
    return "other"


# ─── Hireability ────────────────────────────────────────────────────────────────
def compute_hireability(score: float, matched_skills: List[str], missing_skills: List[str]) -> Tuple[str, str]:
    skill_coverage = len(matched_skills) / max(len(matched_skills) + len(missing_skills), 1)

    if score >= 75 and skill_coverage >= 0.65:
        label = "High"
        reason = "Strong semantic alignment with extensive skill overlap."
    elif score >= 50 and skill_coverage >= 0.40:
        label = "Medium"
        reason = "Moderate fit with some skill gaps; upskilling recommended."
    else:
        label = "Low"
        reason = "Significant gaps in required skills and content alignment."

    return label, reason


# ─── Main matching logic ─────────────────────────────────────────────────────────
def match_resume(resume_text: str, job_text: str) -> Dict:
    # TF-IDF similarity
    resume_vec = compute_tfidf_vector(resume_text)
    job_vec = compute_tfidf_vector(job_text)
    raw_sim = cosine_similarity(resume_vec, job_vec)
    match_score = round(min(raw_sim * 200, 100), 1)  # scale to 0-100

    # Skill extraction
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched_skills = [s for s in job_skills if s in resume_skills]
    missing_skills = [s for s in job_skills if s not in resume_skills]
    bonus_skills = [s for s in resume_skills if s not in job_skills]

    # Hireability
    hire_label, hire_reason = compute_hireability(match_score, matched_skills, missing_skills)

    # Keyword highlights (top shared terms)
    resume_terms = set(extract_terms(resume_text))
    job_terms = set(extract_terms(job_text))
    shared_keywords = list(resume_terms & job_terms - STOPWORDS)[:30]

    # Recommendation
    if match_score >= 75:
        recommendation = (
            f"Excellent match! Your resume aligns strongly with this role. "
            f"You have {len(matched_skills)} of the {len(job_skills)} required skills. "
            f"Focus on quantifying your achievements before applying."
        )
    elif match_score >= 50:
        top_missing = ", ".join(missing_skills[:3]) if missing_skills else "none identified"
        recommendation = (
            f"Good potential fit with room to improve. You match {len(matched_skills)} of "
            f"{len(job_skills)} required skills. Consider adding experience with: {top_missing}. "
            f"Tailor your resume language closer to the job description."
        )
    else:
        top_missing = ", ".join(missing_skills[:4]) if missing_skills else "review the JD carefully"
        recommendation = (
            f"Significant skill gaps detected. You meet {len(matched_skills)} of "
            f"{len(job_skills)} required skills. Prioritize acquiring: {top_missing}. "
            f"Consider applying to entry-level roles while building these competencies."
        )

    return {
        "match_score": match_score,
        "hireability": hire_label,
        "hireability_reason": hire_reason,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "bonus_skills": bonus_skills[:10],
        "shared_keywords": shared_keywords,
        "recommendation": recommendation,
        "stats": {
            "resume_word_count": len(resume_text.split()),
            "jd_word_count": len(job_text.split()),
            "resume_skills_count": len(resume_skills),
            "jd_skills_count": len(job_skills),
        }
    }


# ─── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        with open(sys.argv[1]) as f:
            resume = f.read()
        with open(sys.argv[2]) as f:
            jd = f.read()
        result = match_resume(resume, jd)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python matcher.py <resume.txt> <job_description.txt>")
