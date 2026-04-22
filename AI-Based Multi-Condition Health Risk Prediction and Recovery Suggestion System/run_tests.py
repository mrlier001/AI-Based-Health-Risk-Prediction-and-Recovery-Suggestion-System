"""
run_tests.py
------------
Testing checklist for the Healthcare AI project.
Covers all 5 modules with normal, edge, and invalid cases.

Run from project root:
    python run_tests.py

Each test prints PASS or FAIL with the expected output.
No external test framework needed.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_FILES, TARGET_COLUMNS
from src.preprocess      import load_dataset, preprocess_data, scale_user_input
from src.train_model     import get_candidate_models, evaluate_model
from src.predict         import predict_all, load_model, _validate_input
from src.recommendations import get_recommendations, get_all_recommendations
from src.database        import init_db, insert_prediction, fetch_history, clear_history
from sklearn.preprocessing import StandardScaler

# ── Test runner ───────────────────────────────────────────────────
passed = failed = 0

def check(name, condition, expected="", actual=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        if expected or actual:
            print(f"         expected : {expected}")
            print(f"         actual   : {actual}")
        failed += 1

def raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False

def section(title):
    print(f"\n{'='*58}\n  {title}\n{'='*58}")


# ================================================================
# MODULE 1 — PREPROCESS
# ================================================================
section("MODULE 1: preprocess.py")

# --- load_dataset: normal cases ---
print("\n  load_dataset — normal cases")
df = load_dataset(DATA_FILES["diabetes"])
check("loads diabetes.csv as DataFrame",     isinstance(df, pd.DataFrame))
check("diabetes has 800 rows",               df.shape[0] == 800)
check("diabetes has 9 columns",              df.shape[1] == 9)
check("Outcome column exists",               "Outcome" in df.columns)

df_h = load_dataset(DATA_FILES["heart"])
check("loads heart.csv",                     isinstance(df_h, pd.DataFrame))
check("heart has target column",             "target" in df_h.columns)

# --- load_dataset: invalid input ---
print("\n  load_dataset — invalid input")
check("integer path raises TypeError",       raises(lambda: load_dataset(123),           TypeError))
check("empty string raises ValueError",      raises(lambda: load_dataset(""),            ValueError))
check("whitespace string raises ValueError", raises(lambda: load_dataset("   "),         ValueError))
check(".txt extension raises ValueError",    raises(lambda: load_dataset("data/f.txt"),  ValueError))
check("path traversal raises ValueError",    raises(lambda: load_dataset("../app.py"),   ValueError))
check("missing file raises FileNotFoundError",
      raises(lambda: load_dataset("data/missing.csv"), FileNotFoundError))

# --- preprocess_data: normal cases ---
print("\n  preprocess_data — normal cases")
result = preprocess_data(df, "Outcome")
check("returns dict",                        isinstance(result, dict))
check("has all 6 keys", set(result.keys()) == {
    "X_train","X_test","y_train","y_test","scaler","feature_names"})
check("X_train is 80% of data",              result["X_train"].shape[0] == 640)
check("X_test is 20% of data",               result["X_test"].shape[0]  == 160)
check("feature_names is a list",             isinstance(result["feature_names"], list))
check("feature_names has 8 entries",         len(result["feature_names"]) == 8)
check("scaler is StandardScaler",
      isinstance(result["scaler"], StandardScaler))
check("X_train columns match feature count",
      result["X_train"].shape[1] == len(result["feature_names"]))

# --- preprocess_data: missing values dropped ---
print("\n  preprocess_data — missing value handling")
df_mv = pd.DataFrame({
    "A": [1.0]*10 + [np.nan]*2 + [1.0]*8,
    "B": [2.0]*20,
    "label": [0]*10 + [1]*10,
})
r_mv = preprocess_data(df_mv, "label")
total = r_mv["X_train"].shape[0] + r_mv["X_test"].shape[0]
check("NaN rows dropped before split",       total == 18)

# --- preprocess_data: categorical encoding ---
print("\n  preprocess_data — categorical encoding")
df_cat = pd.DataFrame({
    "age":    list(range(20, 40)),
    "gender": ["Male","Female"] * 10,
    "label":  [0]*10 + [1]*10,
})
r_cat = preprocess_data(df_cat, "label")
check("raw 'gender' column removed",         "gender" not in r_cat["feature_names"])
check("encoded gender column present",
      any("gender" in f for f in r_cat["feature_names"]))

# --- preprocess_data: invalid input ---
print("\n  preprocess_data — invalid input")
check("non-DataFrame raises TypeError",      raises(lambda: preprocess_data("x","y"),    TypeError))
check("empty DataFrame raises ValueError",   raises(lambda: preprocess_data(pd.DataFrame(),"y"), ValueError))
check("integer target raises TypeError",     raises(lambda: preprocess_data(df, 99),     TypeError))
check("empty target raises ValueError",      raises(lambda: preprocess_data(df, ""),     ValueError))
check("missing target raises ValueError",    raises(lambda: preprocess_data(df,"missing"),ValueError))

# --- scale_user_input ---
print("\n  scale_user_input — normal and invalid")
scaler = result["scaler"]
fnames = result["feature_names"]
user   = {f: 0.0 for f in fnames}
scaled = scale_user_input(user, fnames, scaler)
check("returns numpy array shape (1, 8)",    scaled.shape == (1, 8))
check("non-dict raises TypeError",           raises(lambda: scale_user_input("x", fnames, scaler), TypeError))
check("empty dict raises ValueError",        raises(lambda: scale_user_input({}, fnames, scaler),  ValueError))
check("empty feature list raises ValueError",raises(lambda: scale_user_input(user, [], scaler),    ValueError))


# ================================================================
# MODULE 2 — TRAINING
# ================================================================
section("MODULE 2: train_model.py")

print("\n  get_candidate_models")
models = get_candidate_models()
check("returns dict",                        isinstance(models, dict))
check("has 3 models",                        len(models) == 3)
check("Logistic Regression present",         "Logistic Regression" in models)
check("Decision Tree present",               "Decision Tree" in models)
check("Random Forest present",               "Random Forest" in models)

print("\n  evaluate_model — normal cases")
from sklearn.tree import DecisionTreeClassifier
X_tr = result["X_train"]
y_tr = result["y_train"]
X_te = result["X_test"]
y_te = result["y_test"]
dt   = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr, y_tr)
metrics = evaluate_model(dt, X_te, y_te)
check("returns dict",                        isinstance(metrics, dict))
check("has accuracy key",                    "accuracy"  in metrics)
check("has precision key",                   "precision" in metrics)
check("has recall key",                      "recall"    in metrics)
check("has f1 key",                          "f1"        in metrics)
check("accuracy between 0 and 1",            0.0 <= metrics["accuracy"]  <= 1.0)
check("precision between 0 and 1",           0.0 <= metrics["precision"] <= 1.0)
check("recall between 0 and 1",              0.0 <= metrics["recall"]    <= 1.0)
check("f1 between 0 and 1",                  0.0 <= metrics["f1"]        <= 1.0)
check("all values are floats",
      all(isinstance(v, float) for v in metrics.values()))

print("\n  saved model bundles — all 4 diseases")
from config import MODEL_FILES
for disease in ["diabetes", "heart", "hypertension", "stroke"]:
    bundle = load_model(MODEL_FILES[disease])
    check(f"{disease}: bundle has 'model' key",           "model"           in bundle)
    check(f"{disease}: bundle has 'scaler' key",          "scaler"          in bundle)
    check(f"{disease}: bundle has 'features' key",        "features"        in bundle)
    check(f"{disease}: bundle has 'all_metrics' key",     "all_metrics"     in bundle)
    check(f"{disease}: bundle has 'best_model_name' key", "best_model_name" in bundle)
    check(f"{disease}: features is non-empty list",
          isinstance(bundle["features"], list) and len(bundle["features"]) > 0)
    check(f"{disease}: all_metrics has 3 entries",        len(bundle["all_metrics"]) == 3)


# ================================================================
# MODULE 3 — PREDICTION
# ================================================================
section("MODULE 3: predict.py")

VALID = {
    "age": 45, "bmi": 28.5, "blood_pressure": 130,
    "glucose": 148, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 1, "activity_level": 1,
}

print("\n  predict_all — normal cases")
preds = predict_all(VALID)
check("returns dict",                        isinstance(preds, dict))
check("has 4 disease keys",                  set(preds.keys()) == {"diabetes","heart","hypertension","stroke"})
for d in ["diabetes", "heart", "hypertension", "stroke"]:
    r = preds[d]
    check(f"{d}: has risk key",              "risk"       in r)
    check(f"{d}: has confidence key",        "confidence" in r)
    check(f"{d}: has error key",             "error"      in r)
    check(f"{d}: risk is High/Low Risk",     r["risk"] in ("High Risk","Low Risk"))
    check(f"{d}: confidence 0-100",          0.0 <= r["confidence"] <= 100.0)
    check(f"{d}: error is None",             r["error"] is None)

print("\n  predict_all — low risk profile")
LOW = {
    "age": 25, "bmi": 21.0, "blood_pressure": 110,
    "glucose": 85, "cholesterol": 160, "heart_rate": 65,
    "smoking_status": 0, "activity_level": 3,
}
preds_low = predict_all(LOW)
check("low profile returns results for all 4 diseases",
      all(preds_low[d]["error"] is None for d in preds_low))

print("\n  predict_all — extreme values")
EXTREME_HIGH = {
    "age": 120, "bmi": 60.0, "blood_pressure": 250,
    "glucose": 400, "cholesterol": 600, "heart_rate": 220,
    "smoking_status": 1, "activity_level": 0,
}
EXTREME_LOW = {
    "age": 1, "bmi": 10.0, "blood_pressure": 50,
    "glucose": 50, "cholesterol": 100, "heart_rate": 30,
    "smoking_status": 0, "activity_level": 0,
}
p_hi = predict_all(EXTREME_HIGH)
p_lo = predict_all(EXTREME_LOW)
check("extreme high values: no errors",
      all(p_hi[d]["error"] is None for d in p_hi))
check("extreme low values: no errors",
      all(p_lo[d]["error"] is None for d in p_lo))

print("\n  _validate_input — invalid input")
check("empty dict raises ValueError",
      raises(lambda: _validate_input({}), ValueError))
check("non-dict raises TypeError",
      raises(lambda: _validate_input("bad"), TypeError))
check("missing field raises ValueError",
      raises(lambda: _validate_input({"age": 35}), ValueError))
check("string field raises ValueError",
      raises(lambda: _validate_input({**VALID, "age": "old"}), ValueError))
check("None field raises ValueError",
      raises(lambda: _validate_input({**VALID, "bmi": None}), ValueError))

print("\n  load_model — invalid input")
check("non-string path raises TypeError",
      raises(lambda: load_model(123), TypeError))
check("path traversal raises ValueError",
      raises(lambda: load_model("../config.py"), ValueError))
check("missing file raises FileNotFoundError",
      raises(lambda: load_model(str(_ROOT / "models" / "fake.pkl")), FileNotFoundError))


# ================================================================
# MODULE 4 — RECOMMENDATIONS
# ================================================================
section("MODULE 4: recommendations.py")

EXPECTED_CATS = {"diet","exercise","hydration","sleep","stress","doctor"}

print("\n  get_recommendations — normal cases")
for disease in ["diabetes", "heart_disease", "hypertension"]:
    for risk in ["High Risk", "Low Risk"]:
        r = get_recommendations(disease, risk)
        check(f"{disease} {risk}: error is None",        r["error"] is None)
        check(f"{disease} {risk}: has 6 categories",
              set(r["suggestions"].keys()) == EXPECTED_CATS)
        check(f"{disease} {risk}: each category is a list",
              all(isinstance(v, list) for v in r["suggestions"].values()))
        check(f"{disease} {risk}: each category non-empty",
              all(len(v) > 0 for v in r["suggestions"].values()))

print("\n  get_recommendations — invalid input")
r_bad_d = get_recommendations("cancer", "High Risk")
check("unknown disease: error set, suggestions empty",
      r_bad_d["error"] is not None and r_bad_d["suggestions"] == {})

r_bad_r = get_recommendations("diabetes", "Medium Risk")
check("unknown risk level: error set, suggestions empty",
      r_bad_r["error"] is not None and r_bad_r["suggestions"] == {})

r_empty_d = get_recommendations("", "High Risk")
check("empty disease string: error set",     r_empty_d["error"] is not None)

r_empty_r = get_recommendations("diabetes", "")
check("empty risk string: error set",        r_empty_r["error"] is not None)

r_none_d = get_recommendations(None, "High Risk")
check("None disease: error set",             r_none_d["error"] is not None)

print("\n  get_all_recommendations — normal cases")
flat = {
    "diabetes":      "High Risk",
    "heart_disease": "Low Risk",
    "hypertension":  "High Risk",
}
all_recs = get_all_recommendations(flat)
check("returns dict with 3 keys",
      set(all_recs.keys()) == {"diabetes","heart_disease","hypertension"})
for d in ["diabetes", "heart_disease", "hypertension"]:
    check(f"{d}: error is None",             all_recs[d]["error"] is None)
    check(f"{d}: has 6 categories",
          set(all_recs[d]["suggestions"].keys()) == EXPECTED_CATS)

print("\n  get_all_recommendations — invalid input")
r_non_dict = get_all_recommendations("not a dict")
check("non-dict input: all 3 diseases have errors",
      all(r_non_dict[d]["error"] is not None for d in r_non_dict))

r_missing = get_all_recommendations({})
check("empty dict: all 3 diseases have errors",
      all(r_missing[d]["error"] is not None for d in r_missing))

r_error_str = get_all_recommendations({"diabetes": "Error: model failed"})
check("Error: prefix propagated as error",
      r_error_str["diabetes"]["error"] is not None)


# ================================================================
# MODULE 5 — DATABASE
# ================================================================
section("MODULE 5: database.py")

init_db()

SAMPLE = {
    "age": 45, "bmi": 27.5, "blood_pressure": 130,
    "glucose": 110, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 0, "activity_level": 2,
    "diabetes_risk":           "Low Risk",
    "diabetes_confidence":     91.0,
    "heart_risk":              "High Risk",
    "heart_confidence":        82.4,
    "hypertension_risk":       "Low Risk",
    "hypertension_confidence": 88.0,
    "recommendations_summary": "test summary",
}

print("\n  insert_prediction — normal cases")
row_id = insert_prediction(SAMPLE)
check("returns integer row ID",              isinstance(row_id, int) and row_id >= 1)

partial = {"age": 30, "bmi": 22.0}
row_id2 = insert_prediction(partial)
check("partial record inserts (missing keys default to None)",
      isinstance(row_id2, int) and row_id2 >= 1)

print("\n  insert_prediction — invalid input")
check("string input returns None",           insert_prediction("bad")  is None)
check("integer input returns None",          insert_prediction(42)     is None)
check("empty dict returns None",             insert_prediction({})     is None)
check("list input returns None",             insert_prediction([1,2])  is None)

print("\n  fetch_history — normal cases")
history = fetch_history()
check("returns a list",                      isinstance(history, list))
check("list is non-empty after inserts",     len(history) >= 1)

latest = history[0]
check("each record is a dict",               isinstance(latest, dict))
check("record has 'id' key",                 "id"        in latest)
check("record has 'timestamp' key",          "timestamp" in latest)
check("record has 'age' key",                "age"       in latest)
check("record has 'diabetes_risk' key",      "diabetes_risk" in latest)
check("id is an integer",                    isinstance(latest["id"], int))
check("timestamp is a string",               isinstance(latest["timestamp"], str))
check("age is a float",                      isinstance(latest["age"], (float, type(None))))
check("ordered newest first",                history[0]["id"] >= history[-1]["id"])

print("\n  fetch_history — all 17 columns present")
required = {
    "id","timestamp","age","bmi","blood_pressure","glucose","cholesterol",
    "heart_rate","smoking_status","activity_level",
    "diabetes_risk","diabetes_confidence",
    "heart_risk","heart_confidence",
    "hypertension_risk","hypertension_confidence",
    "recommendations_summary",
}
check("all 17 columns in every record",
      all(required.issubset(set(r.keys())) for r in history))

print("\n  clear_history")
clear_history()
after = fetch_history()
check("returns empty list after clear",      after == [])

print("\n  fetch_history — empty table")
empty = fetch_history()
check("returns [] when table is empty",      empty == [])
check("returns list type even when empty",   isinstance(empty, list))


# ================================================================
# EDGE CASES SUMMARY
# ================================================================
section("EDGE CASES SUMMARY")

print("\n  Empty inputs")
check("predict_all: empty dict raises ValueError",
      raises(lambda: predict_all({}), ValueError))
check("get_recommendations: empty disease returns error dict (no raise)",
      get_recommendations("", "High Risk")["error"] is not None)
check("insert_prediction: empty dict returns None",
      insert_prediction({}) is None)
check("preprocess_data: empty DataFrame raises ValueError",
      raises(lambda: preprocess_data(pd.DataFrame(), "x"), ValueError))

print("\n  Invalid types")
check("predict_all: list input raises TypeError",
      raises(lambda: predict_all([1,2,3]), TypeError))
check("load_dataset: integer raises TypeError",
      raises(lambda: load_dataset(42), TypeError))
check("get_all_recommendations: list input returns error dicts",
      all(v["error"] is not None for v in get_all_recommendations([]).values()))

print("\n  Extreme numeric values")
p = predict_all(EXTREME_HIGH)
check("age=120, glucose=400: all predictions complete without error",
      all(p[d]["error"] is None for d in p))
p2 = predict_all(EXTREME_LOW)
check("age=1, glucose=50: all predictions complete without error",
      all(p2[d]["error"] is None for d in p2))

print("\n  Boundary risk levels")
check("'High Risk' accepted by get_recommendations",
      get_recommendations("diabetes", "High Risk")["error"] is None)
check("'Low Risk' accepted by get_recommendations",
      get_recommendations("diabetes", "Low Risk")["error"] is None)
check("'Medium Risk' rejected by get_recommendations",
      get_recommendations("diabetes", "Medium Risk")["error"] is not None)
check("'high risk' (lowercase) rejected",
      get_recommendations("diabetes", "high risk")["error"] is not None)


# ================================================================
# FINAL RESULTS
# ================================================================
total = passed + failed
print(f"\n{'='*58}")
print(f"  RESULTS: {passed}/{total} tests passed")
if failed:
    print(f"  FAILED : {failed} test(s) — review output above")
else:
    print(f"  ALL TESTS PASSED")
print(f"{'='*58}\n")
