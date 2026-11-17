import sqlite3
import os
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(os.path.dirname(Config.DATABASE)):
        os.makedirs(os.path.dirname(Config.DATABASE))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin'
        )
    ''')
    
    # Models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            accuracy REAL,
            memory_usage REAL,
            inference_time REAL,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL,
            voc_percentage REAL,
            explanation TEXT,
            advice TEXT,
            patient_name TEXT DEFAULT 'Anonymous',
            patient_age INTEGER DEFAULT 0,
            patient_gender TEXT DEFAULT 'Not Specified',
            smoking_status TEXT DEFAULT 'Unknown',
            symptoms TEXT DEFAULT 'None',
            prob_benign REAL DEFAULT 0.0,
            prob_malignant REAL DEFAULT 0.0,
            prob_normal REAL DEFAULT 0.0,
            heatmap_filename TEXT DEFAULT '',
            risk_level TEXT DEFAULT 'Moderate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Auto-migration for existing databases
    existing_cols = [col[1] for col in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
    new_columns = [
        ('patient_name', 'TEXT DEFAULT "Anonymous"'),
        ('patient_age', 'INTEGER DEFAULT 0'),
        ('patient_gender', 'TEXT DEFAULT "Not Specified"'),
        ('smoking_status', 'TEXT DEFAULT "Unknown"'),
        ('symptoms', 'TEXT DEFAULT "None"'),
        ('prob_benign', 'REAL DEFAULT 0.0'),
        ('prob_malignant', 'REAL DEFAULT 0.0'),
        ('prob_normal', 'REAL DEFAULT 0.0'),
        ('heatmap_filename', 'TEXT DEFAULT ""'),
        ('risk_level', 'TEXT DEFAULT "Moderate"')
    ]
    for col_name, col_def in new_columns:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def}")
    
    # Dataset Stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            count INTEGER NOT NULL
        )
    ''')

    # Add default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
