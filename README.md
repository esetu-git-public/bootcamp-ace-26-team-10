# bootcamp-ace-26-team-10
Bootcamp by ACE students — Team 10

# 🩺 Chronic Kidney Disease Prediction System

## 📋 Objective
Develop a Machine Learning–based web application to predict Chronic Kidney Disease stages using clinical parameters, with an interactive Streamlit interface for data entry, prediction, and reporting.

---

## 🧠 Overview
This system takes patient clinical data — either entered manually or uploaded as a CSV — and predicts their CKD risk/stage using a trained classification model. The app includes user login, a prediction dashboard, and downloadable result reports.

**Domain:** Healthcare / Clinical Machine Learning
**Bootcamp:** ACE Bootcamp — Team 10

---

## 🛠️ Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📊 Dataset
- `Training_CKD_dataset.csv`
- `Testing_CKD_dataset.csv`
- **Source:** [Chronic Kidney Disease (CKD) Clinical Dataset — Kaggle](https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset)

| Detail | Value |
|---|---|
| Target Variable | CKD Stage / Risk Level *(confirm exact class labels once finalized)* |
| Number of Records | *(fill in once confirmed from the dataset)* |
| Number of Features | *(fill in — clinical parameters such as Age, Blood Pressure, Serum Creatinine, eGFR, Hemoglobin, etc.)* |
| Missing Values | *(note if the dataset required imputation)* |

> 📌 **To finalize this section:** confirm the exact number of rows/features and the list of CKD stages/classes from your actual `Training_CKD_dataset.csv` once EDA is complete, and replace the placeholders above.

---

## 🔬 Pipeline

```
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
```

---

## 🏋️ Model Performance
| Metric | Value |
|---|---|
| Accuracy | *(add your final test-set accuracy)* |
| Precision | *(add)* |
| Recall | *(add)* |
| F1 Score | *(add)* |

> 📌 Replace with your real evaluation numbers once training is finalized — specific metrics make this section credible to reviewers.

---

## 🏛️ Project Structure

```
## 📁 Project Structure


project/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/
│     └── kidney_model.pkl          # Trained model bundle (generated)
│
├── dataset/
│     ├── Training_CKD_dataset.csv  # Training data (21,001 rows)
│     └── Testing_CKD_dataset.csv   # Testing data
│
├── notebooks/
│     ├── 01_Data_Cleaning.ipynb    # Data quality & cleaning
│     ├── 02_EDA.ipynb              # Exploratory Data Analysis
│     └── 03_Model_Training.ipynb   # Model training & evaluation
│
├── utils/
│     ├── __init__.py
│     ├── preprocessing.py          # Data loading, encoding, scaling
│     └── prediction.py             # Model loading & inference
│
└── assets/
      ├── confusion_matrix.png       # Generated after training
      └── model_comparison.png       # Generated after training


---       

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
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
```

**Login Credentials (development/demo only):**
Username: `admin`
Password: `admin123`
> ⚠️ Replace with real hashed credentials before sharing this publicly or submitting for evaluation.

---

## 🌐 Deployment
*(fill in if you deploy — e.g., Streamlit Community Cloud, Hugging Face Spaces)*

| Platform | Setup |
|---|---|
| Streamlit Community Cloud | Connect GitHub repo → deploy `app.py` |
| Local only | `streamlit run app.py` |

---

## 👥 Team — ACE Bootcamp Team 10

| Role | Name |
|---|---|
| Scrum Master | Geetha |
| Team Lead | D. Shravanthi |
| Product Owner | K. Rohit |
| Business Owner | E. Varun |

---

Built with ❤️ by **Team 10 — ACE Bootcamp**
