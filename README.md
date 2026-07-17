# bootcamp-ace-26-team-10

Bootcamp ACE 2026 Team 10 project for a Chronic Kidney Disease (CKD) risk prediction system.

## Overview
This repository contains a Streamlit-based application that predicts CKD stage from patient clinical data. The workflow includes data preprocessing, model training, evaluation, prediction explainability, PDF report generation, and a user-facing dashboard for clinicians and patients.

## What the app includes
- An interactive prediction form for clinical input values
- A training pipeline that evaluates multiple classifiers
- Best-model selection based on weighted F1-score
- Prediction explanation and report-generation support
- PDF export and email-delivery support for generated reports
- Authentication, patient management, appointment scheduling, feedback collection, and audit logging

## Project layout
```text
project/
├── app.py
├── auth.py
├── train_model.py
├── requirements.txt
├── ckd_utils/
│   ├── prediction.py
│   ├── preprocessing.py
│   └── report_generator.py
├── dataset/
├── model/
├── notebooks/
└── tests/
```

## Getting started

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r project/requirements.txt
```

### 3. Train the model
```bash
python project/train_model.py
```

### 4. Launch the app
```bash
streamlit run project/app.py
```

## Model details
The training workflow evaluates several classifiers:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The most suitable model is selected using weighted F1-score to improve multiclass performance.

## Testing
Run the test suite from the repository root:
```bash
pytest project/tests
```

## Disclaimer
This application is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for clinical decisions.
