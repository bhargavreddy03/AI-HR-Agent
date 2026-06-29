# 🤖 AI-Based Intelligent HR Agent for Automated Resume Screening and Recruitment Optimization

An AI-powered Human Resource (HR) Agent that automates resume screening, candidate ranking, and recruitment workflow using Machine Learning. The system intelligently evaluates resumes, predicts candidate suitability, ranks applicants based on multiple criteria, schedules interviews, and generates skill-based interview questions.

---

## 📌 Project Overview

Recruiters often receive hundreds or thousands of resumes for a single job opening. Manual screening is time-consuming, inconsistent, and prone to bias.

This project introduces an **AI-Based HR Agent** that automates the recruitment process using Machine Learning algorithms and intelligent candidate ranking techniques.

The system performs:

- Resume Screening
- Candidate Classification
- Candidate Ranking
- Recruitment Pipeline Management
- Interview Scheduling
- AI-based Interview Question Generation

---

## 🚀 Features

✅ Automated Resume Screening

- Accepts candidate resume information
- Extracts important candidate attributes
- Predicts candidate suitability

---

✅ Intelligent Candidate Ranking

Ranks candidates based on:

- Skills Match
- Experience
- Education
- Certifications
- Keyword Matching
- Skill Diversity
- Job Description Similarity

---

✅ Recruitment Pipeline Management

Tracks candidates through different recruitment stages:

- Applied
- Shortlisted
- Interview Scheduled
- Selected
- Rejected

---

✅ Interview Scheduler

Automatically assigns interview slots without conflicts.

---

✅ AI Interview Question Generator

Generates technical and scenario-based interview questions according to candidate skills.

---

## 📂 Dataset

The project uses approximately **1200 candidate profiles**.

Each profile contains:

- Candidate Name
- Skills
- Years of Experience
- Education
- Field of Study
- Certifications
- Current Job Role

Dataset Split:

- Training Data → 75%
- Testing Data → 25%

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Text Feature Extraction
- Skill Clustering
- Feature Engineering
- Feature Selection
- Robust Scaling
- Power Transformation (Yeo-Johnson)
- Winsorization
- SMOTE (Class Balancing)
- Noise Injection
- Data Normalization

---

## 🧠 Machine Learning Models

The following models were trained and evaluated:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Multi-Layer Perceptron (MLP)

---

## 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Mean Squared Error (MSE)
- Cross Validation Accuracy

---

## 🏆 Best Performing Models

The best performance was achieved by:

- Random Forest
- Gradient Boosting

These models produced high accuracy with excellent precision and recall.

---

## 🔄 Project Workflow

```
Resume Upload
      │
      ▼
Resume Parsing
      │
      ▼
Feature Extraction
      │
      ▼
Data Preprocessing
      │
      ▼
Machine Learning Model
      │
      ▼
Candidate Classification
      │
      ▼
Candidate Ranking
      │
      ▼
Recruitment Pipeline
      │
      ▼
Interview Scheduling
      │
      ▼
Question Generation
      │
      ▼
Final Candidate Selection
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- NumPy
- Pandas

### Data Visualization

- Matplotlib
- Seaborn

### Data Processing

- SMOTE
- Feature Engineering
- RobustScaler
- PowerTransformer

### Model Storage

- Joblib

---

## 📁 Project Structure

```
AI-HR-Agent/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│
├── preprocessing/
│
├── notebooks/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-HR-Agent.git
```

Move into project folder

```bash
cd AI-HR-Agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python app.py
```

---

## 📈 Future Enhancements

- Resume Parsing using NLP
- BERT-based Resume Matching
- Large Language Model Integration
- PDF Resume Parsing
- Cloud Deployment
- Explainable AI (XAI)
- Real-time Dashboard
- ATS Integration
- Multi-language Resume Support

---

## 🎯 Applications

- HR Departments
- Recruitment Agencies
- Corporate Hiring
- Campus Recruitment
- Talent Acquisition Platforms
- Job Portals

---

## 👨‍💻 Authors

**N. Bhargav Reddy**

Computer Science & Engineering

Lovely Professional University

---

**M. Jagan Mohan Reddy**

Computer Science & Engineering

Lovely Professional University

---

## 📚 References

This project is based on Machine Learning, Recruitment Automation, Resume Screening, Recommendation Systems, and HR Analytics techniques using established research in Artificial Intelligence and Data Mining.

---

## 📄 License

This project is developed for educational and research purposes.

---

## ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
