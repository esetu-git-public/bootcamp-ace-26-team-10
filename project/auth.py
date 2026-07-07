import sqlite3
import hashlib
import re
import streamlit as st

DB_FILE = "users.db"

def create_users_table():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    with sqlite3.connect(DB_FILE) as conn:
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
        conn.commit()
    return True, "Registration successful! Please login."

def login_user(username, password):
    hashed_pwd = hash_password(password)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pwd))
        if c.fetchone():
            return True
    return False

def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["auth_page"] = "login"

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None
