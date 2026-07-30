import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)
    
    # Predictions history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            score INTEGER,
            result TEXT,
            credit_amount INTEGER,
            duration INTEGER,
            created_at TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    conn.commit()
    conn.close()


def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists"


def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return False
    
    stored_hash = result[0]
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)


def save_prediction(username, score, result, credit_amount, duration):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO predictions (username, score, result, credit_amount, duration, created_at) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (username, score, result, credit_amount, duration, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_user_history(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT score, result, credit_amount, duration, created_at FROM predictions WHERE username = ? ORDER BY id ASC",
        (username,)
    )
    results = cursor.fetchall()
    conn.close()
    
    return results