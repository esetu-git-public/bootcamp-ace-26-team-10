# Chronic Kidney Disease Risk Prediction System

An AI-powered clinical decision support web application that predicts the stage of Chronic Kidney Disease (CKD) using 35 clinical biomarkers and a trained machine learning model.

Built by **ACE Bootcamp Team 10** as part of the GEN AI Bootcamp program.

---

## 🫁 Overview

Chronic Kidney Disease (CKD) often goes undetected until advanced stages. This project provides an early-warning risk classification tool that predicts a patient's CKD stage based on routine clinical and lab data, enabling earlier intervention and better-informed care decisions.

---

## ✨ Features

- Streamlit web interface with dedicated pages for prediction, analytics, and education
- 35-feature clinical input pipeline covering demographics, vitals, kidney function, electrolytes, blood panel, glucose/lipids, and risk factors
- Four trained classifiers compared automatically, with the best model selected by weighted F1-score
- Confidence score returned alongside each prediction
- Auto-generated performance visualizations (confusion matrix, model comparison)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Model Training | scikit-learn |
| Model Serialization | joblib (`.pkl` bundle) |
| Data | Kaggle CKD Dataset (21,001 training rows) |
| Project Management | ClickUp |
| Version Control | GitHub |

---

## 📁 Project Structure

```
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
      ├── confusion_matrix.png      # Generated after training
      └── model_comparison.png      # Generated after training
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<org-or-username>/ckd-risk-prediction.git
   cd ckd-risk-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess the training & testing datasets
- Train 4 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Select the best model by weighted F1-score
- Save the model bundle to `model/kidney_model.pkl`
- Generate performance charts in `assets/`

### 3. Launch the Web App
```bash
streamlit run app.py
```

---

## 🎯 Target Classes

| Class | CKD Stage | eGFR Range |
|-------|-----------|------------|
| Healthy Kidney | — | ≥ 90 mL/min |
| Mild CKD (Stage 1–2) | Stage 1–2 | 60–89 mL/min |
| Moderate CKD (Stage 3) | Stage 3 | 30–59 mL/min |
| Severe CKD (Stage 4) | Stage 4 | 15–29 mL/min |
| Kidney Failure (Stage 5) | Stage 5 | < 15 mL/min |

---

## 🔬 Input Features (35 total)

| Category | Features |
|----------|---------|
| Demographics | Age, Gender, BMI |
| Vital Signs | Systolic BP, Diastolic BP, Heart Rate |
| Kidney Function | Serum Creatinine, BUN, eGFR, Urine Albumin, Urine Protein, ACR, Urine Specific Gravity |
| Electrolytes | Sodium, Potassium, Calcium, Phosphorus, Chloride, Bicarbonate |
| Blood Panel | Hemoglobin, RBC Count, WBC Count, Platelet Count, Packed Cell Volume |
| Glucose & Lipids | Blood Glucose Random, Fasting Glucose, HbA1c, Cholesterol, Triglycerides |
| Serum Proteins | Serum Albumin, Total Protein |
| Risk Factors | Diabetes, Hypertension, Smoking Status, Family History of Kidney Disease |

---

## 🤖 Machine Learning Models Trained

- **Logistic Regression** – Baseline linear classifier
- **Decision Tree** – Non-linear tree-based classifier
- **Random Forest** – Ensemble bagging classifier ⭐ (typically best)
- **Gradient Boosting** – Ensemble boosting classifier

The best model is selected based on **weighted F1-score** on the test set.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix
- Full Classification Report (per-class)

---

## 🖥️ Web Application Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview, CKD stages, key stats, how it works |
| 🔬 Prediction | Patient input form → CKD stage prediction + confidence |
| 📊 Analytics | Dataset summary, class distribution, feature charts, confusion matrix |
| ℹ️ About | CKD explanation, prediction workflow, algorithm details |

---

## 👥 Team — ACE Bootcamp Team 10

| Role | Responsibility |
|---|---|
| Team Lead | Oversees project execution, monitors progress and structure |
| Scrum Master | Facilitates ceremonies, tracks sprint tasks and GitHub issues |
| Product Owner | Defines requirements, builds core application features |
| Business Owner | Provides requirements and validates deliverables |
| Developer | Implements data pipeline, model training, and core logic |

---

## 🤝 Contributing

This is a bootcamp learning project. Team members should:
1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Commit changes with clear messages
3. Open a Pull Request for review before merging to `main`

---

## ⚠️ Medical Disclaimer

This application is intended for **educational and research purposes only**.
It is **not a substitute** for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is developed for educational purposes as part of the GEN AI Bootcamp. Add a license here if you plan to open-source it (e.g., MIT License).

---

## 📬 Contact

For questions about this project, reach out to the ACE Bootcamp Team 10 project channel.
