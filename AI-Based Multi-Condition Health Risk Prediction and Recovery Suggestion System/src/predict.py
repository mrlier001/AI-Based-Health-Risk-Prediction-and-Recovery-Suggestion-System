"""
predict.py
----------
Loads trained model bundles and generates health risk predictions
for diabetes, heart disease, hypertension, stroke, kidney disease,
and lung disease.

Exports required by app.py
---------------------------
    predict_all(user_input)     - run all disease predictions, return nested dict
    load_model(model_path)      - load one .pkl bundle by file path (cached)
    SUPPORTED_DISEASES          - dict mapping disease key -> model file path

Return format of predict_all()
-------------------------------
    {
        "diabetes":     {"risk": "High Risk", "confidence": 82.4, "error": None},
        "heart":        {"risk": "Low Risk",  "confidence": 91.0, "error": None},
        "hypertension": {"risk": "High Risk", "confidence": 76.3, "error": None},
        "stroke":       {"risk": "Low Risk",  "confidence": 88.1, "error": None},
        "kidney":       {"risk": "High Risk", "confidence": 79.5, "error": None},
        "lung":         {"risk": "Low Risk",  "confidence": 93.2, "error": None},
    }

    On failure for any disease:
        {"risk": None, "confidence": None, "error": "reason string"}
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to sys.path so config can be imported
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MODEL_FILES, MODELS_DIR
from src.validators import FIELD_RULES

# In-memory cache keyed by resolved path string.
# Avoids reloading .pkl files on every Streamlit rerun.
_model_cache: dict = {}

# SUPPORTED_DISEASES — dict mapping disease key -> model file path.
# app.py imports this and iterates its keys.
# Keys must match DISEASE_LABELS in app.py exactly.
SUPPORTED_DISEASES: dict = {
    "diabetes":     MODEL_FILES["diabetes"],
    "heart":        MODEL_FILES["heart"],
    "hypertension": MODEL_FILES["hypertension"],
    "stroke":       MODEL_FILES["stroke"],
    "kidney":       MODEL_FILES["kidney"],
    "lung":         MODEL_FILES["lung"],
}

# Required input fields — all must be present and numeric
_REQUIRED_FIELDS = (
    "age", "bmi", "blood_pressure", "glucose",
    "cholesterol", "heart_rate", "smoking_status", "activity_level",
)


# =============================================================
# load_model(model_path)
# Called by app.py as: load_model(MODEL_FILES[disease])
# =============================================================

def load_model(model_path: str) -> dict:
    """
    Load a saved model bundle from disk by file path.
    Cached after first load — each file is only read once per session.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to a .pkl model bundle file.
        Typically passed as MODEL_FILES[disease] from config.py.

    Returns
    -------
    dict with keys: model, scaler, features, all_metrics, best_model_name

    Raises
    ------
    TypeError         if model_path is not a string
    ValueError        if model_path escapes the models/ directory (CWE-22)
    FileNotFoundError if the .pkl file does not exist
    """
    if not isinstance(model_path, str):
        raise TypeError(
            f"model_path must be a string. Got: {type(model_path).__name__}"
        )

    # Resolve and apply CWE-22 path traversal guard
    resolved = Path(model_path).resolve()
    if not str(resolved).startswith(str(MODELS_DIR.resolve())):
        raise ValueError(
            f"model_path must be inside the models/ folder. "
            f"Got: '{model_path}'"
        )

    # Return from cache if already loaded
    cache_key = str(resolved)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not resolved.exists():
        raise FileNotFoundError(
            f"Model file not found: '{resolved.name}'\n"
            f"Please run: python src/train_model.py"
        )

    bundle = joblib.load(resolved)
    _model_cache[cache_key] = bundle
    print(f"[predict] Loaded '{resolved.name}'  "
          f"(best: {bundle.get('best_model_name', 'unknown')})")
    return bundle


# =============================================================
# Internal helpers
# =============================================================

def _validate_input(user_input: dict):
    """
    Raise TypeError or ValueError if user_input is invalid.

    Checks:
        1. user_input must be a non-empty dict
        2. All 8 required fields must be present
        3. All fields must be numeric (int or float)
        4. All fields in FIELD_RULES must be within their allowed range
    """
    if not isinstance(user_input, dict):
        raise TypeError(
            f"user_input must be a dict. Got: {type(user_input).__name__}"
        )
    if not user_input:
        raise ValueError("user_input must not be empty.")

    for field in _REQUIRED_FIELDS:
        if field not in user_input:
            raise ValueError(
                f"Required field '{field}' is missing. "
                f"Required: {list(_REQUIRED_FIELDS)}"
            )
        value = user_input[field]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Field '{field}' must be numeric. "
                f"Got: {type(value).__name__}"
            )

    # Range validation using the shared FIELD_RULES
    for field, (lo, hi, label, unit, _hint) in FIELD_RULES.items():
        if field not in user_input:
            continue
        value = user_input[field]
        if isinstance(value, (int, float)) and not (lo <= value <= hi):
            raise ValueError(
                f"{label} value {value} is outside the valid range "
                f"({lo}-{hi} {unit})."
            )


def _build_feature_row(user_input: dict, disease: str) -> dict:
    """
    Translate the 8 generic form keys into the exact column names
    each disease model was trained on.
    """
    age   = user_input["age"]
    bmi   = user_input["bmi"]
    bp    = user_input["blood_pressure"]
    gluc  = user_input["glucose"]
    chol  = user_input["cholesterol"]
    hr    = user_input["heart_rate"]
    smoke = user_input["smoking_status"]

    if disease == "diabetes":
        return {
            "Pregnancies":              user_input.get("Pregnancies", 0),
            "Glucose":                  user_input.get("Glucose", gluc),
            "BloodPressure":            user_input.get("BloodPressure", bp),
            "SkinThickness":            user_input.get("SkinThickness", 20),
            "Insulin":                  user_input.get("Insulin", 80),
            "BMI":                      user_input.get("BMI", bmi),
            "DiabetesPedigreeFunction": user_input.get("DiabetesPedigreeFunction", 0.5),
            "Age":                      user_input.get("Age", age),
        }

    if disease == "heart":
        return {
            "age":      user_input.get("age", age),
            "sex":      user_input.get("sex", 1),
            "cp":       user_input.get("cp", 0),
            "trestbps": user_input.get("trestbps", bp),
            "chol":     user_input.get("chol", chol),
            "fbs":      user_input.get("fbs", 1 if gluc > 120 else 0),
            "restecg":  user_input.get("restecg", 0),
            "thalach":  user_input.get("thalach", hr),
            "exang":    user_input.get("exang", 0),
            "oldpeak":  user_input.get("oldpeak", 1.0),
            "slope":    user_input.get("slope", 1),
            "ca":       user_input.get("ca", 0),
            "thal":     user_input.get("thal", 2),
        }

    if disease == "hypertension":
        return {
            "male":            user_input.get("male", 1),
            "age":             age,
            "education":       user_input.get("education", 2),
            "currentSmoker":   user_input.get("currentSmoker", smoke),
            "cigsPerDay":      user_input.get("cigsPerDay", 10 if smoke else 0),
            "BPMeds":          user_input.get("BPMeds", 0),
            "prevalentStroke": user_input.get("prevalentStroke", 0),
            "prevalentHyp":    user_input.get("prevalentHyp", 1 if bp > 140 else 0),
            "diabetes":        user_input.get("diabetes", 1 if gluc > 126 else 0),
            "totChol":         user_input.get("totChol", chol),
            "sysBP":           user_input.get("sysBP", bp),
            "diaBP":           user_input.get("diaBP", int(bp * 0.65)),
            "BMI":             user_input.get("BMI", bmi),
            "heartRate":       user_input.get("heartRate", hr),
            "glucose":         user_input.get("glucose", gluc),
        }

    if disease == "stroke":
        return {
            "age":               age,
            "hypertension":      user_input.get("hypertension", 1 if bp > 140 else 0),
            "heart_disease":     user_input.get("heart_disease", 0),
            "ever_married":      user_input.get("ever_married", 1 if age > 30 else 0),
            "avg_glucose_level": user_input.get("avg_glucose_level", gluc),
            "bmi":               bmi,
            "smoking_status":    smoke,
            "systolic_bp":       user_input.get("systolic_bp", bp),
        }

    if disease == "kidney":
        # UCI Chronic Kidney Disease dataset column names.
        # Defaults are population medians for each feature.
        # bgr = blood glucose random, bu = blood urea, sc = serum creatinine
        # sod = sodium, pot = potassium, hemo = haemoglobin
        # pcv = packed cell volume, wc = white blood cell count
        # rc  = red blood cell count
        return {
            "age":  age,
            "bp":   bp,
            "sg":   user_input.get("sg",   1.020),   # specific gravity
            "al":   user_input.get("al",   0),        # albumin (0-5 scale)
            "su":   user_input.get("su",   0),        # sugar (0-5 scale)
            "bgr":  user_input.get("bgr",  gluc),     # blood glucose random
            "bu":   user_input.get("bu",   40),       # blood urea (mg/dL)
            "sc":   user_input.get("sc",   1.2),      # serum creatinine (mg/dL)
            "sod":  user_input.get("sod",  137),      # sodium (mEq/L)
            "pot":  user_input.get("pot",  4.5),      # potassium (mEq/L)
            "hemo": user_input.get("hemo", 13.5),     # haemoglobin (g/dL)
            "pcv":  user_input.get("pcv",  44),       # packed cell volume (%)
            "wc":   user_input.get("wc",   8000),     # white blood cell count
            "rc":   user_input.get("rc",   5.0),      # red blood cell count
        }

    if disease == "lung":
        # Lung disease dataset column names (binary 1/2 encoded: 1=No, 2=Yes).
        # The form collects smoking_status (0/1) and sex (0/1);
        # all other symptoms default to 1 (No symptom present).
        # GENDER: 1=Male, 2=Female — mapped from sex (0=Female, 1=Male)
        sex = user_input.get("sex", 1)
        return {
            "GENDER":               2 - sex,                          # 1=Male,2=Female
            "AGE":                  age,
            "SMOKING":              1 + smoke,                        # 0->1(No), 1->2(Yes)
            "YELLOW_FINGERS":       user_input.get("YELLOW_FINGERS",       1),
            "ANXIETY":              user_input.get("ANXIETY",              1),
            "PEER_PRESSURE":        user_input.get("PEER_PRESSURE",        1),
            "CHRONIC_DISEASE":      user_input.get("CHRONIC_DISEASE",      1),
            "FATIGUE":              user_input.get("FATIGUE",              1),
            "ALLERGY":              user_input.get("ALLERGY",              1),
            "WHEEZING":             user_input.get("WHEEZING",             1),
            "ALCOHOL_CONSUMING":    user_input.get("ALCOHOL_CONSUMING",    1),
            "COUGHING":             user_input.get("COUGHING",             1),
            "SHORTNESS_OF_BREATH":  user_input.get("SHORTNESS_OF_BREATH",  1),
            "SWALLOWING_DIFFICULTY":user_input.get("SWALLOWING_DIFFICULTY",1),
            "CHEST_PAIN":           user_input.get("CHEST_PAIN",           1),
        }

    raise ValueError(f"No feature mapping for disease: '{disease}'")


def _confidence_to_label(prediction: int, confidence: float) -> str:
    """
    Convert a binary prediction and its confidence score into a
    three-tier risk label.

    Tiers
    -----
    High Risk   : model predicted 1 (positive class) with confidence >= 60%
    Medium Risk : model predicted 1 but confidence is 40-59%  (borderline)
    Low Risk    : model predicted 0 (negative class)

    The medium tier exists because a confidence of, say, 52% on a
    High Risk prediction is not the same as 95% — the model is
    uncertain and the patient should be monitored rather than alarmed.

    Parameters
    ----------
    prediction : int    0 or 1 from model.predict()
    confidence : float  probability of the predicted class (0-100)

    Returns
    -------
    str  one of: 'High Risk', 'Medium Risk', 'Low Risk'
    """
    if prediction == 0:
        return "Low Risk"
    # prediction == 1
    if confidence >= 60.0:
        return "High Risk"
    return "Medium Risk"


def _predict_one(disease: str, model_path: str, user_input: dict) -> dict:
    """
    Run prediction for one disease.

    Returns a result dict with keys:
        label      : 'High Risk', 'Medium Risk', or 'Low Risk'
        risk       : same as label (backward-compatible alias)
        confidence : probability of the predicted class as a percentage
                     (0.0-100.0), or None if model has no predict_proba
        error      : None on success, error message string on failure

    Never raises — all exceptions are captured into the error key.
    """
    result = {"label": None, "risk": None, "confidence": None, "error": None}
    try:
        bundle        = load_model(model_path)
        model         = bundle["model"]
        scaler        = bundle["scaler"]
        feature_names = bundle["features"]

        feature_row = _build_feature_row(user_input, disease)
        input_df    = pd.DataFrame([feature_row], columns=feature_names)
        input_df    = input_df.fillna(0)
        X           = scaler.transform(input_df)

        prediction = int(model.predict(X)[0])

        # Get confidence from predict_proba if available
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X)[0]
            confidence = round(float(proba[prediction]) * 100, 2)
        else:
            # Model does not support probabilities (e.g. SVM without probability=True)
            # Fall back to a binary label with no confidence score
            confidence = None

        label = _confidence_to_label(prediction, confidence if confidence is not None else 100.0)

        result["label"]      = label
        result["risk"]       = label   # alias kept for backward compatibility
        result["confidence"] = confidence

    except FileNotFoundError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Prediction failed: {e}"

    return result


# =============================================================
# predict_all(user_input)
# Main function called by app.py
# =============================================================

def get_feature_importance(disease: str) -> list:
    """
    Return the top feature importances for a trained disease model.

    Only works for tree-based models (Decision Tree, Random Forest)
    which expose feature_importances_. Returns an empty list for
    Logistic Regression or if the model file is missing.

    Parameters
    ----------
    disease : str
        One of: 'diabetes', 'heart', 'hypertension', 'stroke', 'kidney', 'lung'

    Returns
    -------
    list of (feature_name, importance_score) tuples
        Sorted by importance descending. Empty list on failure.
    """
    model_path = MODEL_FILES.get(disease)
    if not model_path:
        return []
    try:
        bundle = load_model(model_path)
        model  = bundle["model"]
        names  = bundle["features"]
        if not hasattr(model, "feature_importances_"):
            return []
        scores = model.feature_importances_
        pairs  = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return pairs
    except Exception:
        return []


def predict_all(user_input: dict) -> dict:
    """
    Run predictions for all diseases from a single user input dict.

    Parameters
    ----------
    user_input : dict
        Must contain the 8 required fields plus any dataset-specific
        aliases that app.py includes (Glucose, trestbps, sysBP, etc.).

    Returns
    -------
    dict  {disease_key: {label, risk, confidence, error}}

        disease_key values: "diabetes", "heart", "hypertension", "stroke", "kidney", "lung"

        On success:
            {
                "label":      "High Risk",   # three-tier label
                "risk":       "High Risk",   # alias of label
                "confidence": 82.4,          # probability % of predicted class
                "error":      None
            }

        label values:
            "High Risk"   — model predicted positive with confidence >= 60%
            "Medium Risk" — model predicted positive with confidence 40-59%
            "Low Risk"    — model predicted negative class

        On failure:
            {"label": None, "risk": None, "confidence": None, "error": "reason"}

    Raises
    ------
    TypeError  if user_input is not a dict
    ValueError if a required field is missing or non-numeric
    """
    _validate_input(user_input)

    return {
        disease: _predict_one(disease, model_path, user_input)
        for disease, model_path in SUPPORTED_DISEASES.items()
    }
