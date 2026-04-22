"""
generate_data.py
----------------
Generates realistic synthetic datasets for the healthcare AI project.
Run this ONCE before training:

    python generate_data.py

Creates:
    data/diabetes.csv      (800 rows, Pima Indians format)
    data/heart.csv         (800 rows, UCI Heart Disease format)
    data/hypertension.csv  (800 rows, Framingham-style format)
    data/stroke.csv        (800 rows, stroke risk format)
    data/kidney.csv        (800 rows, UCI CKD-style format)
    data/lung.csv          (800 rows, lung disease symptom format)
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N   = 800

os.makedirs("data", exist_ok=True)

# ── diabetes.csv ─────────────────────────────────────────────────
diab = pd.DataFrame({
    "Pregnancies":              rng.integers(0, 17, N).tolist(),
    "Glucose":                  rng.integers(70, 200, N).tolist(),
    "BloodPressure":            rng.integers(40, 122, N).tolist(),
    "SkinThickness":            rng.integers(0, 99, N).tolist(),
    "Insulin":                  rng.integers(0, 846, N).tolist(),
    "BMI":                      [round(x, 1) for x in rng.uniform(18.0, 67.0, N).tolist()],
    "DiabetesPedigreeFunction": [round(x, 3) for x in rng.uniform(0.08, 2.42, N).tolist()],
    "Age":                      rng.integers(21, 81, N).tolist(),
})
diab["Outcome"] = (
    (diab["Glucose"] > 140) | (diab["BMI"] > 35)
).astype(int)
diab.to_csv("data/diabetes.csv", index=False)
print(f"data/diabetes.csv    : {diab.shape}  Outcome={diab['Outcome'].value_counts().to_dict()}")

# ── heart.csv ────────────────────────────────────────────────────
heart = pd.DataFrame({
    "age":      rng.integers(29, 77, N).tolist(),
    "sex":      rng.integers(0, 2, N).tolist(),
    "cp":       rng.integers(0, 4, N).tolist(),
    "trestbps": rng.integers(94, 200, N).tolist(),
    "chol":     rng.integers(126, 564, N).tolist(),
    "fbs":      rng.integers(0, 2, N).tolist(),
    "restecg":  rng.integers(0, 3, N).tolist(),
    "thalach":  rng.integers(71, 202, N).tolist(),
    "exang":    rng.integers(0, 2, N).tolist(),
    "oldpeak":  [round(x, 1) for x in rng.uniform(0.0, 6.2, N).tolist()],
    "slope":    rng.integers(0, 3, N).tolist(),
    "ca":       rng.integers(0, 5, N).tolist(),
    "thal":     rng.integers(0, 4, N).tolist(),
})
heart["target"] = (
    (heart["trestbps"] > 140) | (heart["chol"] > 250)
).astype(int)
heart.to_csv("data/heart.csv", index=False)
print(f"data/heart.csv       : {heart.shape}  target={heart['target'].value_counts().to_dict()}")

# ── hypertension.csv ─────────────────────────────────────────────
hyp = pd.DataFrame({
    "male":            rng.integers(0, 2, N).tolist(),
    "age":             rng.integers(32, 70, N).tolist(),
    "education":       rng.integers(1, 5, N).tolist(),
    "currentSmoker":   rng.integers(0, 2, N).tolist(),
    "cigsPerDay":      rng.integers(0, 70, N).tolist(),
    "BPMeds":          rng.integers(0, 2, N).tolist(),
    "prevalentStroke": rng.integers(0, 2, N).tolist(),
    "prevalentHyp":    rng.integers(0, 2, N).tolist(),
    "diabetes":        rng.integers(0, 2, N).tolist(),
    "totChol":         rng.integers(107, 696, N).tolist(),
    "sysBP":           [round(x, 1) for x in rng.uniform(83.5, 295.0, N).tolist()],
    "diaBP":           [round(x, 1) for x in rng.uniform(48.0, 142.5, N).tolist()],
    "BMI":             [round(x, 2) for x in rng.uniform(15.5, 56.8, N).tolist()],
    "heartRate":       rng.integers(44, 143, N).tolist(),
    "glucose":         rng.integers(40, 394, N).tolist(),
})
hyp["target"] = (
    (hyp["sysBP"] > 140) | (hyp["diaBP"] > 90)
).astype(int)
hyp.to_csv("data/hypertension.csv", index=False)
print(f"data/hypertension.csv: {hyp.shape}  target={hyp['target'].value_counts().to_dict()}")

# ── stroke.csv ───────────────────────────────────────────────────
# Stroke shares key risk factors with the other three diseases:
# hypertension, heart disease, high glucose, age, BMI, smoking.
# This makes it a natural fourth condition for this project.
age_vals  = rng.integers(20, 82, N).tolist()
bmi_vals  = [round(x, 1) for x in rng.uniform(15.0, 55.0, N).tolist()]
gluc_vals = rng.integers(55, 280, N).tolist()
sbp_vals  = rng.integers(80, 200, N).tolist()

stroke = pd.DataFrame({
    "age":                age_vals,
    "hypertension":       rng.integers(0, 2, N).tolist(),
    "heart_disease":      rng.integers(0, 2, N).tolist(),
    "ever_married":       rng.integers(0, 2, N).tolist(),
    "avg_glucose_level":  gluc_vals,
    "bmi":                bmi_vals,
    "smoking_status":     rng.integers(0, 3, N).tolist(),  # 0=never,1=formerly,2=smokes,3=unknown
    "systolic_bp":        sbp_vals,
})
# Stroke risk increases with age, high glucose, high BP, and high BMI
stroke["stroke"] = (
    (pd.Series(age_vals) > 60) |
    (pd.Series(gluc_vals) > 180) |
    (pd.Series(sbp_vals) > 160)
).astype(int)
stroke.to_csv("data/stroke.csv", index=False)
print(f"data/stroke.csv      : {stroke.shape}  stroke={stroke['stroke'].value_counts().to_dict()}")

# ── kidney.csv ─────────────────────────────────────────────────────
# UCI Chronic Kidney Disease format.
# Column names must match _build_feature_row("kidney") in predict.py exactly.
kidney_age  = rng.integers(2,  90,  N).tolist()
kidney_bp   = rng.integers(50, 180, N).tolist()
kidney_bgr  = rng.integers(70, 490, N).tolist()   # blood glucose random
kidney_bu   = rng.integers(10, 200, N).tolist()   # blood urea
kidney_sc   = [round(x, 1) for x in rng.uniform(0.4, 15.0, N).tolist()]  # serum creatinine
kidney_hemo = [round(x, 1) for x in rng.uniform(3.1, 17.8, N).tolist()]  # haemoglobin

kidney = pd.DataFrame({
    "age":  kidney_age,
    "bp":   kidney_bp,
    "sg":   [round(x, 3) for x in rng.uniform(1.005, 1.025, N).tolist()],  # specific gravity
    "al":   rng.integers(0, 6, N).tolist(),    # albumin (0-5 scale)
    "su":   rng.integers(0, 6, N).tolist(),    # sugar (0-5 scale)
    "bgr":  kidney_bgr,
    "bu":   kidney_bu,
    "sc":   kidney_sc,
    "sod":  rng.integers(111, 163, N).tolist(),  # sodium (mEq/L)
    "pot":  [round(x, 1) for x in rng.uniform(2.5, 7.5, N).tolist()],  # potassium
    "hemo": kidney_hemo,
    "pcv":  rng.integers(9, 54, N).tolist(),   # packed cell volume (%)
    "wc":   rng.integers(2200, 26400, N).tolist(),  # white blood cell count
    "rc":   [round(x, 2) for x in rng.uniform(2.1, 8.0, N).tolist()],  # red blood cell count
})
# CKD risk: high creatinine, low haemoglobin, high blood urea, or high BP
kidney["target"] = (
    (pd.Series(kidney_sc)   > 5.0)  |
    (pd.Series(kidney_hemo) < 9.0)  |
    (pd.Series(kidney_bu)   > 100)  |
    (pd.Series(kidney_bp)   > 140)
).astype(int)
kidney.to_csv("data/kidney.csv", index=False)
print(f"data/kidney.csv      : {kidney.shape}  target={kidney['target'].value_counts().to_dict()}")

# ── lung.csv ─────────────────────────────────────────────────────────
# Lung disease symptom format.
# Binary encoded: 1 = No symptom, 2 = Symptom present.
# GENDER: 1 = Male, 2 = Female.
# Column names must match _build_feature_row("lung") in predict.py exactly.
def _binary(n):  # helper: random 1 or 2 values
    return rng.integers(1, 3, n).tolist()

lung_smoking = _binary(N)
lung_age     = rng.integers(21, 87, N).tolist()

lung = pd.DataFrame({
    "GENDER":                _binary(N),
    "AGE":                   lung_age,
    "SMOKING":               lung_smoking,
    "YELLOW_FINGERS":        _binary(N),
    "ANXIETY":               _binary(N),
    "PEER_PRESSURE":         _binary(N),
    "CHRONIC_DISEASE":       _binary(N),
    "FATIGUE":               _binary(N),
    "ALLERGY":               _binary(N),
    "WHEEZING":              _binary(N),
    "ALCOHOL_CONSUMING":     _binary(N),
    "COUGHING":              _binary(N),
    "SHORTNESS_OF_BREATH":   _binary(N),
    "SWALLOWING_DIFFICULTY": _binary(N),
    "CHEST_PAIN":            _binary(N),
})
# Lung disease risk: smoker + wheezing, or chronic disease + shortness of breath
lung["target"] = (
    ((lung["SMOKING"] == 2) & (lung["WHEEZING"] == 2)) |
    ((lung["CHRONIC_DISEASE"] == 2) & (lung["SHORTNESS_OF_BREATH"] == 2)) |
    ((lung["COUGHING"] == 2) & (lung["CHEST_PAIN"] == 2))
).astype(int)
lung.to_csv("data/lung.csv", index=False)
print(f"data/lung.csv        : {lung.shape}  target={lung['target'].value_counts().to_dict()}")

print("\nAll datasets ready. Run: python src/train_model.py")
