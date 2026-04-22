"""
database.py
-----------
Phase 7 — Database Integration

Stores and retrieves prediction history using SQLite.

All database connections use the 'with' context manager:

    with sqlite3.connect(DB_PATH) as conn:
        ...

This guarantees the connection is always committed and closed
automatically — even if an error occurs inside the block.
No resource leaks are possible.

Functions
---------
init_db()
    Creates the database file and table on first run.
    Safely migrates existing databases to add new columns.
    Safe to call every time the app starts.

insert_prediction(record: dict)
    Saves one prediction session to the database.
    Accepts a flat dictionary. Missing keys default to None.
    Returns the new row ID on success, None on failure.

fetch_history()
    Returns all saved records as a list of plain dicts,
    ordered newest first (most recent at index 0).

clear_history()
    Deletes all rows from the table.
    Use only during development and testing.

Database file
-------------
Location is set in config.py as DB_PATH.
Default: health_history.db in the project root folder.
SQLite creates the file automatically on first use.

Table: prediction_history
Columns
-------
    id                      auto-incremented row number
    timestamp               date and time the record was saved
    age                     patient age in years
    bmi                     body mass index
    blood_pressure          systolic blood pressure in mmHg
    glucose                 blood glucose in mg/dL
    cholesterol             total cholesterol in mg/dL
    heart_rate              resting heart rate in bpm
    smoking_status          1 = smoker, 0 = non-smoker
    activity_level          0=sedentary 1=light 2=moderate 3=active
    diabetes_risk           'High Risk', 'Medium Risk', or 'Low Risk'
    diabetes_confidence     model confidence percentage (0-100)
    heart_risk              'High Risk', 'Medium Risk', or 'Low Risk'
    heart_confidence        model confidence percentage (0-100)
    hypertension_risk       'High Risk', 'Medium Risk', or 'Low Risk'
    hypertension_confidence model confidence percentage (0-100)
    kidney_risk             'High Risk', 'Medium Risk', or 'Low Risk'
    kidney_confidence       model confidence percentage (0-100)
    lung_risk               'High Risk', 'Medium Risk', or 'Low Risk'
    lung_confidence         model confidence percentage (0-100)
    recommendations_summary plain text summary of all suggestions
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path so config.py can be imported
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_PATH

# SQL to create the table on first run.
# IF NOT EXISTS means this is safe to run on every app startup —
# it does nothing if the table already exists.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    age                     REAL,
    bmi                     REAL,
    blood_pressure          REAL,
    glucose                 REAL,
    cholesterol             REAL,
    heart_rate              REAL,
    smoking_status          INTEGER,
    activity_level          INTEGER,
    diabetes_risk           TEXT,
    diabetes_confidence     REAL,
    heart_risk              TEXT,
    heart_confidence        REAL,
    hypertension_risk       TEXT,
    hypertension_confidence REAL,
    kidney_risk             TEXT,
    kidney_confidence       REAL,
    lung_risk               TEXT,
    lung_confidence         REAL,
    recommendations_summary TEXT
);
"""

# New columns added for kidney and lung disease.
# Used by init_db() to migrate existing databases that were created
# before these columns existed. Each entry is (column_name, sql_type).
# ALTER TABLE in SQLite does not support IF NOT EXISTS, so we catch
# the OperationalError that is raised when the column already exists.
_NEW_COLUMNS = [
    ("kidney_risk",       "TEXT"),
    ("kidney_confidence", "REAL"),
    ("lung_risk",         "TEXT"),
    ("lung_confidence",   "REAL"),
]


# =============================================================
# FUNCTION 1 — init_db
# =============================================================

def init_db():
    """
    Create the database file and prediction_history table if they
    do not already exist. Also safely adds any new columns to an
    existing database without losing old records.

    Migration approach
    ------------------
    SQLite's ALTER TABLE does not support IF NOT EXISTS, so we attempt
    each ALTER TABLE and silently ignore the OperationalError that
    SQLite raises when the column already exists. This is the safest
    approach for beginners and works on all SQLite versions.

    This function is safe to call every time the app starts.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # Step 1: Create the table if it does not exist yet
        conn.execute(_CREATE_TABLE_SQL)

        # Step 2: Add new columns to existing databases.
        # If the column already exists, SQLite raises OperationalError.
        # We catch it and move on — existing data is never touched.
        for col_name, col_type in _NEW_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE prediction_history "
                    f"ADD COLUMN {col_name} {col_type}"
                )
                print(f"[database] Migration: added column '{col_name}'")
            except sqlite3.OperationalError:
                pass  # Column already exists — nothing to do

    print(f"[database] Ready: {Path(DB_PATH).name}")


# =============================================================
# FUNCTION 2 — insert_prediction
# =============================================================

def insert_prediction(record: dict):
    """
    Save one prediction session to the database.

    Parameters
    ----------
    record : dict
        A flat dictionary. Recognised keys:
            age, bmi, blood_pressure, glucose, cholesterol,
            heart_rate, smoking_status, activity_level,
            diabetes_risk, diabetes_confidence,
            heart_risk, heart_confidence,
            hypertension_risk, hypertension_confidence,
            kidney_risk, kidney_confidence,
            lung_risk, lung_confidence,
            recommendations_summary

        Any key not provided defaults to None — no KeyError is raised.

    Returns
    -------
    int   row ID of the newly inserted record (starts at 1)
    None  if validation failed or the database raised an error

    Example
    -------
        record = {
            "age": 45, "bmi": 27.5, "blood_pressure": 130,
            "glucose": 110, "cholesterol": 210, "heart_rate": 78,
            "smoking_status": 0, "activity_level": 2,
            "diabetes_risk": "Low Risk",  "diabetes_confidence": 91.0,
            "heart_risk": "High Risk",    "heart_confidence": 82.4,
            "hypertension_risk": "Low Risk", "hypertension_confidence": 88.0,
            "kidney_risk": "High Risk",   "kidney_confidence": 79.5,
            "lung_risk": "Low Risk",      "lung_confidence": 93.2,
            "recommendations_summary": "DIABETES - Low Risk\n  diet: ...",
        }
        row_id = insert_prediction(record)
    """
    # --- Validate input ---
    if not isinstance(record, dict):
        print(
            f"[database] insert_prediction skipped: "
            f"record must be a dict. Got: {type(record).__name__}"
        )
        return None

    if not record:
        print("[database] insert_prediction skipped: record dict is empty.")
        return None

    # --- Build the row values in the exact column order of the table ---
    # record.get(key) returns None if the key is missing — safe for all fields
    row = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        record.get("age"),
        record.get("bmi"),
        record.get("blood_pressure"),
        record.get("glucose"),
        record.get("cholesterol"),
        record.get("heart_rate"),
        record.get("smoking_status"),
        record.get("activity_level"),
        record.get("diabetes_risk"),
        record.get("diabetes_confidence"),
        record.get("heart_risk"),
        record.get("heart_confidence"),
        record.get("hypertension_risk"),
        record.get("hypertension_confidence"),
        record.get("kidney_risk"),
        record.get("kidney_confidence"),
        record.get("lung_risk"),
        record.get("lung_confidence"),
        record.get("recommendations_summary"),
    )

    # Parameterised query — the ? placeholders prevent SQL injection
    sql = """
        INSERT INTO prediction_history (
            timestamp,
            age, bmi, blood_pressure, glucose, cholesterol,
            heart_rate, smoking_status, activity_level,
            diabetes_risk, diabetes_confidence,
            heart_risk, heart_confidence,
            hypertension_risk, hypertension_confidence,
            kidney_risk, kidney_confidence,
            lung_risk, lung_confidence,
            recommendations_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    try:
        # 'with' commits the INSERT and closes the connection automatically
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(sql, row)
            row_id = cursor.lastrowid
        print(f"[database] Record saved (id={row_id})")
        return row_id
    except sqlite3.Error as e:
        print(f"[database] Insert failed: {e}")
        return None


# =============================================================
# FUNCTION 3 — fetch_history
# =============================================================

def fetch_history() -> list:
    """
    Return all saved prediction records as a list of plain dicts,
    ordered newest first.

    Returns
    -------
    list of dict
        Each dict is one saved prediction session.
        All column values are accessible by name, e.g. record["age"].
        Returns an empty list [] if the table is empty or a read
        error occurs — never raises an exception.

    Example single record
    ---------------------
        {
            "id":                      1,
            "timestamp":               "2024-06-01 14:32:10",
            "age":                     45.0,
            "bmi":                     27.5,
            "blood_pressure":          130.0,
            "glucose":                 110.0,
            "cholesterol":             210.0,
            "heart_rate":              78.0,
            "smoking_status":          0,
            "activity_level":          2,
            "diabetes_risk":           "Low Risk",
            "diabetes_confidence":     91.0,
            "heart_risk":              "High Risk",
            "heart_confidence":        82.4,
            "hypertension_risk":       "Low Risk",
            "hypertension_confidence": 88.0,
            "kidney_risk":             "High Risk",
            "kidney_confidence":       79.5,
            "lung_risk":               "Low Risk",
            "lung_confidence":         93.2,
            "recommendations_summary": "DIABETES - Low Risk\n  diet: ...",
        }
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # row_factory allows column access by name: row["age"]
            # instead of by index: row[2]
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM prediction_history ORDER BY id DESC"
            ).fetchall()

        # Convert sqlite3.Row objects to plain Python dicts
        # so app.py can use standard dict operations like .get()
        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        print(f"[database] Fetch failed: {e}")
        return []


# =============================================================
# FUNCTION 4 — clear_history
# =============================================================

def clear_history():
    """
    Delete all rows from prediction_history.

    The table structure is kept — only the data rows are removed.
    The auto-increment counter resets on next insert.

    Use this only during development and testing.
    Do not call this from app.py.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_history")
    print("[database] All records cleared.")
