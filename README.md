bootcamp-ace-26-team-10

Bootcamp by ACE students — Team 10

🩺 Chronic Kidney Disease Prediction System

📋 Objective

Develop a Machine Learning–based web application to predict Chronic Kidney Disease stages using clinical parameters, with an interactive Streamlit interface for data entry, prediction, and reporting.


🧠 Overview

This system takes patient clinical data — either entered manually or uploaded as a CSV — and predicts their CKD risk/stage using a trained classification model. The app includes user login, a prediction dashboard, and downloadable result reports.

Domain: Healthcare / Clinical Machine Learning
Bootcamp: ACE Bootcamp — Team 10


🛠️ Technologies


Python
Pandas
NumPy
Scikit-learn
Streamlit



📊 Dataset


Training_CKD_dataset.csv
Testing_CKD_dataset.csv
Source: Chronic Kidney Disease (CKD) Clinical Dataset — Kaggle


DetailValueTarget VariableCKD Stage / Risk Level (confirm exact class labels once finalized)Number of Records(fill in once confirmed from the dataset)Number of Features(fill in — clinical parameters such as Age, Blood Pressure, Serum Creatinine, eGFR, Hemoglobin, etc.)Missing Values(note if the dataset required imputation)


📌 To finalize this section: confirm the exact number of rows/features and the list of CKD stages/classes from your actual Training_CKD_dataset.csv once EDA is complete, and replace the placeholders above.




🔬 Pipeline

Raw Dataset (Training_CKD_dataset.csv / Testing_CKD_dataset.csv)
       │
       ▼
Data Cleaning (missing values, type fixes)
       │
       ▼
Exploratory Data Analysis (EDA)
       │
       ▼
Feature Engineering (encoding, scaling)
       │
       ▼
Model Training (Logistic Regression / Random Forest)
       │
       ▼
Model Evaluation (Accuracy, Precision, Recall, F1)
       │
       ▼
Streamlit App (Login → Input/Upload → Prediction → Report)


🏋️ Model Performance

MetricValueAccuracy(add your final test-set accuracy)Precision(add)Recall(add)F1 Score(add)


📌 Replace with your real evaluation numbers once training is finalized — specific metrics make this section credible to reviewers.




🏛️ Project Structure

bootcamp-ace-26-team-10/
├── app.py                      # Streamlit entry point
├── pages/                      # Login, upload/predict, result pages
├── src/
│   ├── config.py
│   ├── preprocessing.py         #   Missing values, encoding, scaling
│   ├── train.py                 #   Model training
│   ├── evaluate.py              #   Metrics & evaluation
│   └── predict.py               #   Load model, run inference
├── data/
│   ├── Training_CKD_dataset.csv
│   └── Testing_CKD_dataset.csv
├── models/
│   └── ckd_model.pkl
├── notebooks/                   # EDA & experiments
├── requirements.txt
├── README.md                    # ← You are here
└── PLAN.md                      # Sprint/technical plan


🚀 Quick Start

bash# 1. Clone the repository
git clone https://github.com/<your-org-or-username>/bootcamp-ace-26-team-10.git
cd bootcamp-ace-26-team-10

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

# 5. (Optional) Train the model from scratch
python src/train.py
python src/evaluate.py

Login Credentials (development/demo only):
Username: admin
Password: admin123


⚠️ Replace with real hashed credentials before sharing this publicly or submitting for evaluation.




🌐 Deployment

(fill in if you deploy — e.g., Streamlit Community Cloud, Hugging Face Spaces)

PlatformSetupStreamlit Community CloudConnect GitHub repo → deploy app.pyLocal onlystreamlit run app.py


👥 Team — ACE Bootcamp Team 10

RoleNameScrum MasterGeethaTeam LeadD. ShravanthiProduct OwnerK. RohitBusiness OwnerE. Varun


Built with ❤️ by Team 10 — ACE Bootcamp
