"""
utils.py
Shared utilities: JSON formatting, data loading, candidate helpers, scoring helpers.

Fix changelog (production hardening):
  FIX-12: load_csv_candidates now uses random_state=42 in df.sample() for reproducible
          results. Without a seed, every run returns a different subset, making
          regression testing and debugging unreliable.
"""

import json
import re
import os
import csv
import random
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

MODEL_DIR  = "models"
GRAPHS_DIR = "static/graphs"

# ─────────────────────────────────────────────
# JSON Helpers
# ─────────────────────────────────────────────

def pretty_json(data: Any) -> str:
    """Pretty-print any JSON-serialisable object."""
    return json.dumps(data, indent=2, default=str)


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────
# Resume / Dataset Helpers
# ─────────────────────────────────────────────

def load_csv_candidates(csv_path: str, max_rows: int = 50) -> List[Dict]:
    """
    Load candidates from CSV and format for HRAgent.rank_candidates.
    Returns list of dicts with id, name, resume_text, experience_years.

    FIX-12: Uses random_state=42 for reproducible sampling across runs.
    If the dataset is smaller than max_rows, all rows are used.
    """
    df = pd.read_csv(csv_path)

    # FIX-11 (partial): guard against empty CSV before sampling
    if df.empty:
        return []

    n  = min(max_rows, len(df))
    # FIX-12: added random_state=42 — reproducible, consistent results
    df = df.sample(n=n, random_state=42)
    candidates = []
    for i, row in df.iterrows():
        skills = str(row.get("Skills", ""))
        edu    = str(row.get("Education_Level", ""))
        exp    = float(row.get("Experience_Years", 0) or 0)
        certs  = str(row.get("Certifications", ""))
        job    = str(row.get("Current_Job_Title", ""))
        jd     = str(row.get("Target_Job_Description", ""))

        resume_text = (
            f"Name: {row.get('Name','')}. "
            f"Education: {edu} in {row.get('Field_of_Study','')}. "
            f"Experience: {exp} years. "
            f"Current role: {job}. "
            f"Skills: {skills}. "
            f"Certifications: {certs}. "
            f"Objective: {jd}"
        )
        candidates.append({
            "id":               i + 1,
            "name":             str(row.get("Name", f"Candidate_{i+1}")),
            "resume_text":      resume_text,
            "experience_years": exp,
            "raw_skills":       skills,
            "education":        edu,
            "current_job":      job,
            "certifications":   certs,
        })
    return candidates


def generate_demo_candidates(n: int = 5) -> List[Dict]:
    """Generate synthetic candidate dicts for demo purposes."""
    names  = ["Priya Sharma", "Arjun Mehta", "Neha Gupta", "Rahul Das", "Zara Khan",
               "Dev Patel", "Ananya Singh", "Kiran Rao", "Aditya Verma", "Meera Nair"]
    skills = [
        "Python Machine Learning Deep Learning TensorFlow Docker AWS CI/CD",
        "JavaScript React Node.js REST APIs MongoDB Agile",
        "SQL Data Analysis Power BI Statistics Python R",
        "Java Spring Boot Microservices Docker Kubernetes Jenkins",
        "Cybersecurity Network Security Linux Ansible Terraform",
    ]
    return [
        {
            "id":               i + 1,
            "name":             names[i % len(names)],
            "resume_text":      random.choice(skills) + f" {random.randint(1, 10)} years experience",
            "experience_years": random.randint(1, 10),
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────
# Score Formatting
# ─────────────────────────────────────────────

def score_badge(score: float) -> str:
    """Return emoji badge for match score."""
    if score >= 0.75:
        return "🟢 Excellent"
    if score >= 0.50:
        return "🟡 Good"
    if score >= 0.30:
        return "🟠 Fair"
    return "🔴 Poor"


def priority_badge(priority: str) -> str:
    icons = {"high": "🚨 HIGH", "medium": "⚠️ MEDIUM", "low": "📋 LOW", "none": "✅ NONE"}
    return icons.get(priority, priority)


# ─────────────────────────────────────────────
# Graph Helpers
# ─────────────────────────────────────────────

GRAPH_FILES = {
    "train_vs_test_accuracy": "train_vs_test_accuracy.png",
    "precision_recall_f1":    "precision_recall_f1.png",
    "fpr_fnr_comparison":     "fpr_fnr_comparison.png",
    "roc_auc_comparison":     "roc_auc_comparison.png",
    "confusion_matrix":       "confusion_matrix.png",
    "cv_comparison":          "cv_comparison.png",
    "error_metrics":          "error_metrics.png",
    "roc_curves":             "roc_curves.png",
    "feature_importance":     "feature_importance.png",
    "actual_vs_predicted":    "actual_vs_predicted.png",
    "metric_dashboard":       "metric_dashboard.png",
    "f1_score_comparison":    "f1_score_comparison.png",
    "fpr_comparison":         "fpr_comparison.png",
    "fnr_comparison":         "fnr_comparison.png",
    "precision_comparison":   "precision_comparison.png",
    "recall_comparison":      "recall_comparison.png",
}


def graphs_exist() -> bool:
    return all(
        os.path.exists(f"{GRAPHS_DIR}/{f}") for f in GRAPH_FILES.values()
    )


def get_graph_path(key: str) -> Optional[str]:
    fname = GRAPH_FILES.get(key)
    if fname is None:
        return None
    full = f"{GRAPHS_DIR}/{fname}"
    return full if os.path.exists(full) else None


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{level}] {msg}")


# ─────────────────────────────────────────────
# Model result summariser (for dashboard)
# ─────────────────────────────────────────────

def summarise_results(results: Dict) -> pd.DataFrame:
    """Convert training results dict to a tidy DataFrame for display."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":         name,
            "Test Accuracy": round(r["test_accuracy"],  4),
            "Train Accuracy": round(r["train_accuracy"], 4),
            "ROC-AUC":       round(r["roc_auc"],        4),
            "Precision":     round(r["precision"],      4),
            "Recall":        round(r["recall"],         4),
            "F1 Score":      round(r["f1_score"],       4),
            "FNR":           round(r["fnr"],            4),
            "FPR":           round(r["fpr"],            4),
            "RMSE":          round(r["rmse"],           4),
            "MSE":           round(r["mse"],            4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Test Accuracy", ascending=False)
        .reset_index(drop=True)
    )