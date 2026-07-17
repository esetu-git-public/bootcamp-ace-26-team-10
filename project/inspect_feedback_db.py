import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
import auth

print('DB_FILE =', auth.DB_FILE)
print('exists =', os.path.exists(auth.DB_FILE))

conn = sqlite3.connect(auth.DB_FILE)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('tables =', cur.fetchall())
cur.execute('SELECT COUNT(*) FROM feedback')
print('feedback_count =', cur.fetchone()[0])
cur.execute('SELECT * FROM feedback ORDER BY id DESC LIMIT 5')
print('rows =', cur.fetchall())
conn.close()
