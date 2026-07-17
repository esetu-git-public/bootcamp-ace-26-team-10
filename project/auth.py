import os
import sqlite3
import hashlib
import re
import streamlit as st
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "users.db")

def init_db():
    create_users_table()
    create_predictions_table()

def create_users_table():
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            ''')
    finally:
        conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            if c.fetchone():
                return False, "Username already exists."
                
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            if c.fetchone():
                return False, "Email already exists."
                
            hashed_pwd = hash_password(password)
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                      (username, email, hashed_pwd))
        log_audit_action(username, "User Registered", f"Email: {email}")
        return True, "Registration successful! Please login."
    finally:
        conn.close()

def login_user(username, password):
    hashed_pwd = hash_password(password)
    conn = sqlite3.connect(DB_FILE)
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pwd))
        if c.fetchone():
            log_audit_action(username, "User Logged In", "Successfully authenticated")
            return True
        # Log failed login attempt
        log_audit_action(username, "Login Attempt Failed", "Incorrect credentials")
        return False
    finally:
        conn.close()

def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["auth_page"] = "login"

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def create_predictions_table():
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    features_json TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    height REAL NOT NULL,
                    weight REAL NOT NULL,
                    diabetes TEXT NOT NULL,
                    hypertension TEXT NOT NULL,
                    smoking_status TEXT NOT NULL,
                    family_history TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    prediction_id INTEGER,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
    finally:
        conn.close()

def normalize_prediction_row(row):
    """Return a prediction row with both canonical and legacy field aliases available."""
    normalized = dict(row or {})
    aliases = {
        "Age": ("Age", "age"),
        "Gender": ("Gender", "gender"),
        "BMI": ("BMI", "bmi"),
        "Diabetes": ("Diabetes", "diabetes"),
        "Hypertension": ("Hypertension", "hypertension"),
        "Family_History_Kidney": ("Family_History_Kidney", "family_history"),
    }

    for canonical, names in aliases.items():
        value = None
        for name in names:
            if name in normalized:
                value = normalized[name]
                break
        if value is not None:
            normalized[canonical] = value
            for name in names:
                if name != canonical:
                    normalized[name] = value

    return normalized


def save_prediction(username, patient_name, prediction_label, confidence, features_dict, created_at=None):
    import json
    if created_at is None:
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_json = json.dumps(features_dict)
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions_v2 (
                    username, patient_name, prediction_label, confidence, created_at, features_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                username, patient_name, prediction_label, confidence, created_at, features_json
            ))
            pred_id = c.lastrowid
        log_audit_action(username, "Prediction Generated", f"Patient: {patient_name}, Stage: {prediction_label} (ID: {pred_id})")
        return pred_id
    finally:
        conn.close()

def get_predictions(username, search_query=None, start_date=None, end_date=None):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = "SELECT * FROM predictions_v2 WHERE username = ?"
        params = [username]
        
        if search_query:
            query += " AND patient_name LIKE ?"
            params.append(f"%{search_query}%")
            
        if start_date:
            query += " AND date(created_at) >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND date(created_at) <= ?"
            params.append(end_date)
            
        query += " ORDER BY created_at DESC"
        
        c.execute(query, params)
        rows = c.fetchall()
        
        import json
        result = []
        for r in rows:
            d = dict(r)
            try:
                features = json.loads(d['features_json'])
                d.update(features)
            except:
                pass
            result.append(normalize_prediction_row(d))
        return result
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Patient Profile Management
# ---------------------------------------------------------------------------
def create_patient(username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO patients (
                    username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, created_at))
            patient_id = c.lastrowid
        log_audit_action(username, "Create Patient", f"Created patient: {name} (ID: {patient_id})")
        return patient_id
    finally:
        conn.close()

def get_patients(username):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM patients WHERE username = ? ORDER BY name ASC', (username,))
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def update_patient(username, patient_id, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history):
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                UPDATE patients SET
                    name = ?, age = ?, gender = ?, height = ?, weight = ?,
                    diabetes = ?, hypertension = ?, smoking_status = ?, family_history = ?
                WHERE id = ? AND username = ?
            ''', (name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, patient_id, username))
        log_audit_action(username, "Update Patient", f"Updated patient: {name} (ID: {patient_id})")
        return True
    finally:
        conn.close()

def delete_patient(username, patient_id):
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('DELETE FROM patients WHERE id = ? AND username = ?', (patient_id, username))
        log_audit_action(username, "Delete Patient", f"Deleted patient ID: {patient_id}")
        return True
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Appointment Reminders
# ---------------------------------------------------------------------------
def schedule_appointment(username, patient_name, appointment_date, notes):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO appointments (username, patient_name, appointment_date, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, patient_name, appointment_date, notes, created_at))
            appointment_id = c.lastrowid
        log_audit_action(username, "Schedule Appointment", f"Scheduled appointment for {patient_name} on {appointment_date}")
        return appointment_id
    finally:
        conn.close()

def get_appointments(username):
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM appointments WHERE username = ? ORDER BY appointment_date ASC', (username,))
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def delete_appointment(username, appointment_id):
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('DELETE FROM appointments WHERE id = ? AND username = ?', (appointment_id, username))
        log_audit_action(username, "Cancel Appointment", f"Cancelled appointment ID: {appointment_id}")
        return True
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Feedback Module
# ---------------------------------------------------------------------------
def submit_feedback(username, rating, comments, prediction_id=None):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO feedback (username, prediction_id, rating, comments, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, prediction_id, rating, comments, created_at))
            conn.commit()
        log_audit_action(username, "Submit Feedback", f"Rated prediction: {rating} stars")
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_all_feedback():
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM feedback ORDER BY created_at DESC')
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------
def log_audit_action(username, action, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO audit_logs (username, action, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (username, action, details, timestamp))
        return True
    finally:
        conn.close()

def get_audit_logs():
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 500')
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Admin Panel Metrics
# ---------------------------------------------------------------------------
def get_admin_metrics():
    conn = sqlite3.connect(DB_FILE)
    metrics = {}
    try:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        metrics["total_users"] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions_v2')
        metrics["total_predictions"] = c.fetchone()[0]
        
        c.execute('SELECT username, email FROM users ORDER BY username ASC')
        metrics["users_list"] = [{"username": r[0], "email": r[1]} for r in c.fetchall()]
        return metrics
    finally:
        conn.close()

