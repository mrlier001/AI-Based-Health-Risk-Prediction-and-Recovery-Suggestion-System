"""
test_project.py
---------------
Complete testing checklist for the Healthcare AI project.

Run from the project root folder:
    python test_project.py

Covers:
    1. preprocess.py  — load_dataset, preprocess_data, scale_user_input
    2. train_model.py — get_candidate_models, evaluate_model, train_and_save
    3. predict.py     — load_model, predict_all
    4. recommendations.py — get_recommendations, get_all_recommendations
    5. database.py    — init_db, insert_prediction, fetch_history
    6. End-to-end flow
    7. Edge cases and invalid inputs

Each test prints PASS or FAIL with a short description.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_FILES, TARGET_COLUMNS, MODEL_FILES
from src.preprocess     import load_dataset, preprocess_data, scale_user_input
from src.predict        import predict_all, load_model
from src.recommendations import get_recommendations, get_all_recommendations
from src.database       import init_db, insert_prediction, fetch_history, clear_history

# ── Helpers ───────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ================================================================
# 1. PREPROCESS.PY
# ================================================================
section("1. preprocess.py")

# 1a. load_dataset — valid file
try:
    df = load_dataset(DATA_FILES["diabetes"])
    check("load_dataset returns DataFrame",
          isinstance(df, pd.DataFrame))
    check("load_dataset has correct columns",
          "Outcome" in df.columns)
    check("load_dataset has 800 rows",
          len(df) == 800)
except Exception as e:
    check("load_dataset valid file", False, str(e))

# 1b. load_dataset — type validation
try:
    load_dataset(123)
    check("load_dataset rejects int", False, "no error raised")
except TypeError:
    check("load_dataset rejects int", True)

# 1c. load_dataset — empty string
try:
    load_dataset("")
    check("load_dataset rejects empty string", False, "no error raised")
except ValueError:
    check("load_dataset rejects empty string", True)

# 1d. load_dataset — wrong extension
try:
    load_dataset("data/file.txt")
    check("load_dataset rejects .txt extension", False, "no error raised")
except ValueError:
    check("load_dataset rejects .txt extension", True)

# 1e. load_dataset — path traversal
try:
    load_dataset("data/../config.py")
    check("load_dataset blocks path traversal", False, "no error raised")
except ValueError:
    check("load_dataset blocks path traversal", True)

# 1f. load_dataset — missing file
try:
    load_dataset("data/missing.csv")
    check("load_dataset raises FileNotFoundError", False, "no error raised")
except FileNotFoundError:
    check("load_dataset raises FileNotFoundError", True)

# 1g. preprocess_data — valid input
try:
    df = load_dataset(DATA_FILES["diabetes"])
    result = preprocess_data(df, "Outcome")
    check("preprocess_data returns dict",
          isinstance(result, dict))
    check("preprocess_data has all 6 keys",
          set(result.keys()) == {"X_train","X_test","y_train","y_test","scaler","feature_names"})
    check("preprocess_data X_train shape correct",
          result["X_train"].shape == (640, 8))
    check("preprocess_data X_test shape correct",
          result["X_test"].shape == (160, 8))
    check("preprocess_data scaler is fitted",
          hasattr(result["scaler"], "mean_"))
    check("preprocess_data feature_names is list of 8",
          isinstance(result["feature_names"], list) and len(result["feature_names"]) == 8)
except Exception as e:
    check("preprocess_data valid input", False, str(e))

# 1h. preprocess_data — wrong df type
try:
    preprocess_data("not a dataframe", "Outcome")
    check("preprocess_data rejects non-DataFrame", False, "no error raised")
except TypeError:
    check("preprocess_data rejects non-DataFrame", True)

# 1i. preprocess_data — empty DataFrame
try:
    preprocess_data(pd.DataFrame(), "Outcome")
    check("preprocess_data rejects empty DataFrame", False, "no error raised")
except ValueError:
    check("preprocess_data rejects empty DataFrame", True)

# 1j. preprocess_data — missing target column
try:
    df = load_dataset(DATA_FILES["diabetes"])
    preprocess_data(df, "NonExistentColumn")
    check("preprocess_data rejects missing target", False, "no error raised")
except ValueError:
    check("preprocess_data rejects missing target", True)

# 1k. scale_user_input — valid
try:
    df = load_dataset(DATA_FILES["diabetes"])
    result = preprocess_data(df, "Outcome")
    sample = {f: 0.0 for f in result["feature_names"]}
    scaled = scale_user_input(sample, result["feature_names"], result["scaler"])
    check("scale_user_input returns array shape (1, 8)",
          scaled.shape == (1, 8))
except Exception as e:
    check("scale_user_input valid", False, str(e))

# 1l. scale_user_input — wrong scaler type
try:
    scale_user_input({"a": 1}, ["a"], "not_a_scaler")
    check("scale_user_input rejects wrong scaler type", False, "no error raised")
except TypeError:
    check("scale_user_input rejects wrong scaler type", True)


# ================================================================
# 2. TRAIN_MODEL.PY
# ================================================================
section("2. train_model.py")

from src.train_model import get_candidate_models, evaluate_model, DISEASES
import joblib

# 2a. get_candidate_models
models = get_candidate_models()
check("get_candidate_models returns dict with 3 entries",
      isinstance(models, dict) and len(models) == 3)
check("get_candidate_models has correct keys",
      set(models.keys()) == {"Logistic Regression", "Decision Tree", "Random Forest"})

# 2b. DISEASES list
check("DISEASES contains 3 entries",
      len(DISEASES) == 3)
check("DISEASES contains correct keys",
      set(DISEASES) == {"diabetes", "heart", "hypertension"})

# 2c. evaluate_model — returns correct keys
try:
    df = load_dataset(DATA_FILES["diabetes"])
    result = preprocess_data(df, "Outcome")
    model = list(models.values())[0]
    model.fit(result["X_train"], result["y_train"])
    metrics = evaluate_model(model, result["X_test"], result["y_test"])
    check("evaluate_model returns dict with 4 keys",
          set(metrics.keys()) == {"accuracy","precision","recall","f1"})
    check("evaluate_model accuracy is float between 0 and 1",
          isinstance(metrics["accuracy"], float) and 0 <= metrics["accuracy"] <= 1)
    check("evaluate_model f1 is float between 0 and 1",
          isinstance(metrics["f1"], float) and 0 <= metrics["f1"] <= 1)
except Exception as e:
    check("evaluate_model", False, str(e))

# 2d. Saved bundles exist and have correct keys
for disease in ["diabetes", "heart", "hypertension"]:
    bundle_path = Path(MODEL_FILES[disease])
    check(f"models/{disease}_model.pkl exists",
          bundle_path.exists())
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        check(f"{disease} bundle has all required keys",
              set(bundle.keys()) >= {"model","scaler","features","all_metrics","best_model_name"})
        check(f"{disease} scaler is fitted",
              hasattr(bundle["scaler"], "mean_"))
        check(f"{disease} model is trained",
              hasattr(bundle["model"], "classes_"))


# ================================================================
# 3. PREDICT.PY
# ================================================================
section("3. predict.py")

# Minimal valid input — only the 8 required fields
_MIN_INPUT = {
    "age": 45, "bmi": 27.5, "blood_pressure": 130,
    "glucose": 110, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 0, "activity_level": 2,
}

# 3a. predict_all — valid input returns correct shape
try:
    result = predict_all(_MIN_INPUT)
    check("predict_all returns dict",
          isinstance(result, dict))
    check("predict_all has 4 disease keys",
          set(result.keys()) == {"diabetes","heart","hypertension","stroke"})
    for d in ["diabetes","heart","hypertension"]:
        check(f"predict_all {d} has risk/confidence/error keys",
              set(result[d].keys()) >= {"risk","confidence","error"})
        check(f"predict_all {d} error is None",
              result[d]["error"] is None,
              str(result[d].get("error")))
        check(f"predict_all {d} risk is High/Low Risk",
              result[d]["risk"] in ("High Risk","Low Risk"))
        check(f"predict_all {d} confidence is 0-100",
              result[d]["confidence"] is None or 0 <= result[d]["confidence"] <= 100)
except Exception as e:
    check("predict_all valid input", False, str(e))

# 3b. predict_all — not a dict
try:
    predict_all("not a dict")
    check("predict_all rejects string input", False, "no error raised")
except TypeError:
    check("predict_all rejects string input", True)

# 3c. predict_all — empty dict
try:
    predict_all({})
    check("predict_all rejects empty dict", False, "no error raised")
except ValueError:
    check("predict_all rejects empty dict", True)

# 3d. predict_all — missing required field
try:
    predict_all({"age": 45})
    check("predict_all rejects missing fields", False, "no error raised")
except ValueError:
    check("predict_all rejects missing fields", True)

# 3e. predict_all — non-numeric field
try:
    bad = {**_MIN_INPUT, "age": "forty-five"}
    predict_all(bad)
    check("predict_all rejects non-numeric field", False, "no error raised")
except ValueError:
    check("predict_all rejects non-numeric field", True)

# 3f. Prediction is meaningful — low-risk profile
low_risk = {
    "age": 22, "bmi": 20.0, "blood_pressure": 105,
    "glucose": 80, "cholesterol": 150, "heart_rate": 62,
    "smoking_status": 0, "activity_level": 3,
}
high_risk = {
    "age": 68, "bmi": 40.0, "blood_pressure": 175,
    "glucose": 210, "cholesterol": 290, "heart_rate": 98,
    "smoking_status": 1, "activity_level": 0,
}
try:
    r_low  = predict_all(low_risk)
    r_high = predict_all(high_risk)
    check("low-risk profile returns Low Risk for diabetes",
          r_low["diabetes"]["risk"] == "Low Risk",
          r_low["diabetes"]["risk"])
    check("high-risk profile returns High Risk for diabetes",
          r_high["diabetes"]["risk"] == "High Risk",
          r_high["diabetes"]["risk"])
except Exception as e:
    check("prediction meaningfulness", False, str(e))

# 3g. load_model — wrong type
try:
    load_model(123)
    check("load_model rejects non-string path", False, "no error raised")
except TypeError:
    check("load_model rejects non-string path", True)


# ================================================================
# 4. RECOMMENDATIONS.PY
# ================================================================
section("4. recommendations.py")

# 4a. get_recommendations — valid
r = get_recommendations("diabetes", "High Risk")
check("get_recommendations returns dict with 4 keys",
      set(r.keys()) == {"disease","risk","suggestions","error"})
check("get_recommendations error is None for valid input",
      r["error"] is None)
check("get_recommendations has 6 suggestion categories",
      len(r["suggestions"]) == 6)
check("get_recommendations categories are correct",
      set(r["suggestions"].keys()) == {"diet","exercise","hydration","sleep","stress","doctor"})
check("get_recommendations each category has at least 1 tip",
      all(len(tips) >= 1 for tips in r["suggestions"].values()))

# 4b. get_recommendations — all three diseases and both risk levels
for disease in ["diabetes", "heart_disease", "hypertension"]:
    for risk in ["High Risk", "Low Risk"]:
        r = get_recommendations(disease, risk)
        check(f"get_recommendations {disease} {risk} has no error",
              r["error"] is None)

# 4c. get_recommendations — invalid disease
r = get_recommendations("cancer", "High Risk")
check("get_recommendations returns error for unknown disease",
      r["error"] is not None)
check("get_recommendations suggestions empty for unknown disease",
      r["suggestions"] == {})

# 4d. get_recommendations — invalid risk level
r = get_recommendations("diabetes", "Medium Risk")
check("get_recommendations returns error for unknown risk level",
      r["error"] is not None)

# 4e. get_recommendations — empty string
r = get_recommendations("", "High Risk")
check("get_recommendations returns error for empty disease",
      r["error"] is not None)

# 4f. get_all_recommendations — valid
preds = predict_all(_MIN_INPUT)
flat = {
    "diabetes":     preds["diabetes"]["risk"],
    "heart_disease": preds["heart"]["risk"],
    "hypertension": preds["hypertension"]["risk"],
}
all_recs = get_all_recommendations(flat)
check("get_all_recommendations returns dict with 3 keys",
      set(all_recs.keys()) == {"diabetes","heart_disease","hypertension"})
check("get_all_recommendations all errors are None",
      all(r["error"] is None for r in all_recs.values()))

# 4g. get_all_recommendations — not a dict
result = get_all_recommendations("not a dict")
check("get_all_recommendations handles non-dict input",
      all(r["error"] is not None for r in result.values()))


# ================================================================
# 5. DATABASE.PY
# ================================================================
section("5. database.py")

init_db()
clear_history()

# 5a. insert_prediction — valid record
record = {
    "age": 45, "bmi": 27.5, "blood_pressure": 130,
    "glucose": 110, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 0, "activity_level": 2,
    "diabetes_risk": "Low Risk",    "diabetes_confidence": 91.0,
    "heart_risk": "High Risk",      "heart_confidence": 82.4,
    "hypertension_risk": "Low Risk","hypertension_confidence": 88.0,
    "recommendations_summary": "DIABETES - Low Risk\n  diet: Eat vegetables",
}
row_id = insert_prediction(record)
check("insert_prediction returns integer row ID",
      isinstance(row_id, int) and row_id > 0)

# 5b. fetch_history — returns list of dicts
rows = fetch_history()
check("fetch_history returns a list",
      isinstance(rows, list))
check("fetch_history returns at least 1 record",
      len(rows) >= 1)
check("fetch_history records are dicts",
      all(isinstance(r, dict) for r in rows))

# 5c. fetch_history — all 17 columns present
expected_cols = [
    "id","timestamp","age","bmi","blood_pressure","glucose","cholesterol",
    "heart_rate","smoking_status","activity_level",
    "diabetes_risk","diabetes_confidence",
    "heart_risk","heart_confidence",
    "hypertension_risk","hypertension_confidence",
    "recommendations_summary",
]
r = rows[0]
check("fetch_history record has all 17 columns",
      all(k in r for k in expected_cols))

# 5d. fetch_history — values stored correctly
check("fetch_history age stored correctly",
      r["age"] == 45.0)
check("fetch_history diabetes_risk stored correctly",
      r["diabetes_risk"] == "Low Risk")
check("fetch_history heart_risk stored correctly",
      r["heart_risk"] == "High Risk")
check("fetch_history confidence stored correctly",
      r["diabetes_confidence"] == 91.0)

# 5e. fetch_history — newest first
insert_prediction({**record, "age": 60})
rows = fetch_history()
check("fetch_history newest record is first",
      rows[0]["age"] == 60.0)

# 5f. insert_prediction — bad input rejected
check("insert_prediction rejects string input",
      insert_prediction("not a dict") is None)
check("insert_prediction rejects empty dict",
      insert_prediction({}) is None)
check("insert_prediction rejects None",
      insert_prediction(None) is None)

# 5g. fetch_history — returns empty list when table is empty
clear_history()
check("fetch_history returns [] when table is empty",
      fetch_history() == [])


# ================================================================
# 6. END-TO-END FLOW
# ================================================================
section("6. End-to-End Flow")

init_db()
clear_history()

try:
    # Step 1: predict
    preds = predict_all(_MIN_INPUT)
    check("E2E step 1: predict_all succeeds",
          all(r["error"] is None for r in preds.values()))

    # Step 2: recommendations
    flat = {
        "diabetes":      preds["diabetes"]["risk"],
        "heart_disease": preds["heart"]["risk"],
        "hypertension":  preds["hypertension"]["risk"],
    }
    recs = get_all_recommendations(flat)
    check("E2E step 2: get_all_recommendations succeeds",
          all(r["error"] is None for r in recs.values()))

    # Step 3: database save
    rid = insert_prediction({
        **_MIN_INPUT,
        "diabetes_risk":           preds["diabetes"]["risk"],
        "diabetes_confidence":     preds["diabetes"]["confidence"],
        "heart_risk":              preds["heart"]["risk"],
        "heart_confidence":        preds["heart"]["confidence"],
        "hypertension_risk":       preds["hypertension"]["risk"],
        "hypertension_confidence": preds["hypertension"]["confidence"],
        "recommendations_summary": "test",
    })
    check("E2E step 3: insert_prediction returns row ID",
          isinstance(rid, int) and rid > 0)

    # Step 4: history display
    history = fetch_history()
    check("E2E step 4: fetch_history returns the saved record",
          len(history) >= 1 and history[0]["diabetes_risk"] == preds["diabetes"]["risk"])

    check("E2E full flow completed without errors", True)

except Exception as e:
    check("E2E full flow", False, str(e))


# ================================================================
# 7. EDGE CASES
# ================================================================
section("7. Edge Cases")

# 7a. Boundary values — minimum valid inputs
min_vals = {
    "age": 1, "bmi": 10.0, "blood_pressure": 50,
    "glucose": 50, "cholesterol": 100, "heart_rate": 30,
    "smoking_status": 0, "activity_level": 0,
}
try:
    r = predict_all(min_vals)
    check("Edge: minimum boundary values accepted",
          all(v["error"] is None for v in r.values()))
except Exception as e:
    check("Edge: minimum boundary values", False, str(e))

# 7b. Boundary values — maximum valid inputs
max_vals = {
    "age": 120, "bmi": 60.0, "blood_pressure": 250,
    "glucose": 400, "cholesterol": 600, "heart_rate": 220,
    "smoking_status": 1, "activity_level": 3,
}
try:
    r = predict_all(max_vals)
    check("Edge: maximum boundary values accepted",
          all(v["error"] is None for v in r.values()))
except Exception as e:
    check("Edge: maximum boundary values", False, str(e))

# 7c. Float values for integer fields
float_vals = {**_MIN_INPUT, "age": 45.9, "smoking_status": 0.0}
try:
    r = predict_all(float_vals)
    check("Edge: float values for integer fields accepted",
          all(v["error"] is None for v in r.values()))
except Exception as e:
    check("Edge: float values for integer fields", False, str(e))

# 7d. Extra keys in input are ignored
extra = {**_MIN_INPUT, "unknown_field": 999, "another_field": "abc"}
try:
    r = predict_all(extra)
    check("Edge: extra keys in input are ignored",
          all(v["error"] is None for v in r.values()))
except Exception as e:
    check("Edge: extra keys ignored", False, str(e))

# 7e. Database insert with all None values
rid = insert_prediction({
    "age": None, "bmi": None, "blood_pressure": None,
    "glucose": None, "cholesterol": None, "heart_rate": None,
    "smoking_status": None, "activity_level": None,
    "diabetes_risk": None, "diabetes_confidence": None,
    "heart_risk": None, "heart_confidence": None,
    "hypertension_risk": None, "hypertension_confidence": None,
    "recommendations_summary": None,
})
check("Edge: insert_prediction accepts all-None values",
      isinstance(rid, int))

# 7f. Recommendations with error passthrough from predict
error_preds = {
    "diabetes":      "Error: Model not found.",
    "heart_disease": "High Risk",
    "hypertension":  "Low Risk",
}
recs = get_all_recommendations(error_preds)
check("Edge: error passthrough — diabetes has error",
      recs["diabetes"]["error"] is not None)
check("Edge: error passthrough — heart_disease still works",
      recs["heart_disease"]["error"] is None)

# 7g. preprocess_data with DataFrame that has all nulls in one column
try:
    df = load_dataset(DATA_FILES["diabetes"]).copy()
    df["Glucose"] = None
    result = preprocess_data(df, "Outcome")
    check("Edge: DataFrame with all-null column drops rows and still works",
          result["X_train"].shape[0] > 0)
except ValueError:
    check("Edge: DataFrame with all-null column raises ValueError", True)
except Exception as e:
    check("Edge: all-null column", False, str(e))


# ================================================================
# FINAL SUMMARY
# ================================================================
total = _passed + _failed
print(f"\n{'=' * 60}")
print(f"  RESULTS: {_passed} passed  |  {_failed} failed  |  {total} total")
print(f"{'=' * 60}")
if _failed == 0:
    print("  All tests passed. Project is ready for submission.")
else:
    print(f"  {_failed} test(s) failed. Review the FAIL lines above.")
print(f"{'=' * 60}\n")
