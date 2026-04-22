"""
config.py
---------
Central configuration for the healthcare AI project.

All file paths use pathlib so the project works on any machine
regardless of where the folder is located.

This file is the single source of truth for:
    - dataset file paths  (DATA_FILES)
    - target column names (TARGET_COLUMNS)
    - model file paths    (MODEL_FILES)
    - training settings   (TEST_SIZE, RANDOM_STATE)

Supported diseases
------------------
    diabetes     — Pima Indians format       target: Outcome
    heart        — UCI Heart Disease format  target: target
    hypertension — Framingham-style format   target: target
    stroke       — Stroke risk format        target: stroke
    kidney       — CKD-style format          target: target  (0=healthy, 1=ckd)
    lung         — Lung disease format       target: target  (0=healthy, 1=disease)
"""

from pathlib import Path

# ── Folder locations ──────────────────────────────────────────────
# BASE_DIR is always the folder that contains this config.py file.
# All other paths are built relative to it.
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH    = str(BASE_DIR / "health_history.db")

# ── Dataset file paths ────────────────────────────────────────────
# Keys must match exactly across all modules.
# Do not rename these keys without updating every module that uses them.
DATA_FILES = {
    "diabetes":     str(DATA_DIR / "diabetes.csv"),
    "heart":        str(DATA_DIR / "heart.csv"),
    "hypertension": str(DATA_DIR / "hypertension.csv"),
    "stroke":       str(DATA_DIR / "stroke.csv"),
    "kidney":       str(DATA_DIR / "kidney.csv"),
    "lung":         str(DATA_DIR / "lung.csv"),
}

# ── Target column names ───────────────────────────────────────────
# The label column that the model learns to predict.
# 0 = Low Risk / negative class
# 1 = High Risk / positive class
TARGET_COLUMNS = {
    "diabetes":     "Outcome",  # Pima Indians Diabetes format
    "heart":        "target",   # UCI Heart Disease format
    "hypertension": "target",   # Framingham-style format
    "stroke":       "stroke",   # Stroke risk format
    "kidney":       "target",   # CKD format — 0=healthy, 1=ckd
    "lung":         "target",   # Lung disease format — 0=healthy, 1=disease
}

# ── Model bundle file paths ───────────────────────────────────────
# Each .pkl file stores: model + scaler + feature names + metrics
MODEL_FILES = {
    "diabetes":     str(MODELS_DIR / "diabetes_model.pkl"),
    "heart":        str(MODELS_DIR / "heart_model.pkl"),
    "hypertension": str(MODELS_DIR / "hypertension_model.pkl"),
    "stroke":       str(MODELS_DIR / "stroke_model.pkl"),
    "kidney":       str(MODELS_DIR / "kidney_model.pkl"),
    "lung":         str(MODELS_DIR / "lung_model.pkl"),
}

# ── Training settings ─────────────────────────────────────────────
TEST_SIZE    = 0.2   # 20% of data used for testing
RANDOM_STATE = 42    # fixed seed for reproducible results
