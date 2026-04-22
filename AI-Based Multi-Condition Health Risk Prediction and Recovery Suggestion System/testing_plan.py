"""
testing_plan.py
---------------
Complete testing plan for the Healthcare AI project.
Covers all 6 modules with normal, edge, and invalid cases.

Run from project root:
    python testing_plan.py

Prints:
    - Pass/Fail result for every test case
    - A formatted summary table for the project report
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_FILES, TARGET_COLUMNS, MODEL_FILES
from src.preprocess      import load_dataset, preprocess_data, scale_user_input
from src.train_model     import get_candidate_models, evaluate_model
from src.predict         import predict_all, load_model, _validate_input, _confidence_to_label
from src.recommendations import get_recommendations, get_all_recommendations, CATEGORY_LABELS
from src.database        import init_db, insert_prediction, fetch_history, clear_history
from src.validators      import validate_health_inputs
from sklearn.preprocessing import StandardScaler

# ── Test runner ───────────────────────────────────────────────────
results = []   # (module, test_id, description, input_summary, expected, actual, status)
_id = [0]

def run(module, description, input_summary, expected, fn):
    _id[0] += 1
    tid = f"T{_id[0]:03d}"
    try:
        actual = fn()
        status = "PASS" if actual else "FAIL"
    except Exception as e:
        actual = f"Exception: {type(e).__name__}"
        status = "FAIL"
    results.append((module, tid, description, input_summary, expected, str(actual)[:60], status))
    icon = "✓" if status == "PASS" else "✗"
    print(f"  [{icon}] {tid}  {description}")

def raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ================================================================
# MODULE 1 — preprocess.py
# ================================================================
section("MODULE 1: preprocess.py")

df_diab = load_dataset(DATA_FILES["diabetes"])
result  = preprocess_data(df_diab, TARGET_COLUMNS["diabetes"])

# Normal cases
run("preprocess", "load_dataset returns DataFrame",
    "data/diabetes.csv",
    "pandas DataFrame, 800 rows",
    lambda: isinstance(df_diab, pd.DataFrame) and df_diab.shape[0] == 800)

run("preprocess", "preprocess_data returns dict with 6 keys",
    "diabetes DataFrame, target='Outcome'",
    "dict with X_train, X_test, y_train, y_test, scaler, feature_names",
    lambda: set(result.keys()) == {"X_train","X_test","y_train","y_test","scaler","feature_names"})

run("preprocess", "X_train is 80% of 800 rows",
    "800-row dataset, TEST_SIZE=0.2",
    "X_train.shape[0] == 640",
    lambda: result["X_train"].shape[0] == 640)

run("preprocess", "X_test is 20% of 800 rows",
    "800-row dataset, TEST_SIZE=0.2",
    "X_test.shape[0] == 160",
    lambda: result["X_test"].shape[0] == 160)

run("preprocess", "feature_names has 8 entries for diabetes",
    "diabetes dataset (9 cols - 1 target)",
    "len(feature_names) == 8",
    lambda: len(result["feature_names"]) == 8)

run("preprocess", "scaler shape matches feature count",
    "fitted StandardScaler",
    "scaler.mean_.shape == (8,)",
    lambda: result["scaler"].mean_.shape == (8,))

run("preprocess", "NaN rows are dropped before split",
    "DataFrame with 2 NaN rows injected",
    "total rows = original - NaN count",
    lambda: (lambda df: (
        preprocess_data(df, "label")["X_train"].shape[0] +
        preprocess_data(df, "label")["X_test"].shape[0] == 18
    ))(pd.DataFrame({
        "A": [1.0]*10 + [np.nan]*2 + [1.0]*8,
        "B": [2.0]*20,
        "label": [0]*10 + [1]*10,
    })))

run("preprocess", "categorical columns are one-hot encoded",
    "DataFrame with 'gender' text column",
    "'gender' removed, 'gender_Male' present",
    lambda: (lambda r: "gender" not in r["feature_names"] and
             any("gender" in f for f in r["feature_names"]))(
        preprocess_data(pd.DataFrame({
            "age": list(range(20,40)),
            "gender": ["Male","Female"]*10,
            "label": [0]*10+[1]*10,
        }), "label")))

# Invalid / edge cases
run("preprocess", "integer path raises TypeError",
    "load_dataset(123)",
    "TypeError",
    lambda: raises(lambda: load_dataset(123), TypeError))

run("preprocess", "empty string raises ValueError",
    'load_dataset("")',
    "ValueError",
    lambda: raises(lambda: load_dataset(""), ValueError))

run("preprocess", "wrong extension raises ValueError",
    'load_dataset("data/f.txt")',
    "ValueError",
    lambda: raises(lambda: load_dataset("data/f.txt"), ValueError))

run("preprocess", "path traversal raises ValueError",
    'load_dataset("../app.py")',
    "ValueError",
    lambda: raises(lambda: load_dataset("../app.py"), ValueError))

run("preprocess", "missing file raises FileNotFoundError",
    'load_dataset("data/missing.csv")',
    "FileNotFoundError",
    lambda: raises(lambda: load_dataset("data/missing.csv"), FileNotFoundError))

run("preprocess", "non-DataFrame raises TypeError",
    'preprocess_data("x", "y")',
    "TypeError",
    lambda: raises(lambda: preprocess_data("x", "y"), TypeError))

run("preprocess", "empty DataFrame raises ValueError",
    "preprocess_data(pd.DataFrame(), 'y')",
    "ValueError",
    lambda: raises(lambda: preprocess_data(pd.DataFrame(), "y"), ValueError))

run("preprocess", "missing target column raises ValueError",
    "preprocess_data(df, 'missing')",
    "ValueError",
    lambda: raises(lambda: preprocess_data(df_diab, "missing"), ValueError))

run("preprocess", "scale_user_input returns shape (1, 8)",
    "valid user dict, 8 features",
    "numpy array shape (1, 8)",
    lambda: scale_user_input(
        {f: 0.0 for f in result["feature_names"]},
        result["feature_names"],
        result["scaler"]
    ).shape == (1, 8))

run("preprocess", "scale_user_input: empty dict raises ValueError",
    "scale_user_input({}, ...)",
    "ValueError",
    lambda: raises(
        lambda: scale_user_input({}, result["feature_names"], result["scaler"]),
        ValueError))


# ================================================================
# MODULE 2 — train_model.py
# ================================================================
section("MODULE 2: train_model.py")

from sklearn.tree import DecisionTreeClassifier
models  = get_candidate_models()
dt      = DecisionTreeClassifier(random_state=42)
dt.fit(result["X_train"], result["y_train"])
metrics = evaluate_model(dt, result["X_test"], result["y_test"])

run("train_model", "get_candidate_models returns 3 models",
    "no input",
    "dict with 3 entries",
    lambda: len(models) == 3)

run("train_model", "all 3 algorithm names present",
    "no input",
    "Logistic Regression, Decision Tree, Random Forest",
    lambda: all(k in models for k in
                ["Logistic Regression","Decision Tree","Random Forest"]))

run("train_model", "evaluate_model returns 4 metric keys",
    "fitted DecisionTree, test data",
    "dict with accuracy, precision, recall, f1",
    lambda: set(metrics.keys()) == {"accuracy","precision","recall","f1"})

run("train_model", "all metric values are floats in [0, 1]",
    "fitted DecisionTree, test data",
    "0.0 <= each metric <= 1.0",
    lambda: all(isinstance(v, float) and 0.0 <= v <= 1.0
                for v in metrics.values()))

for disease in ["diabetes","heart","hypertension","stroke"]:
    bundle = load_model(MODEL_FILES[disease])
    run("train_model", f"{disease} bundle has all 5 required keys",
        f"models/{disease}_model.pkl",
        "model, scaler, features, all_metrics, best_model_name",
        lambda b=bundle: all(k in b for k in
            ["model","scaler","features","all_metrics","best_model_name"]))
    run("train_model", f"{disease} all_metrics has 3 algorithm entries",
        f"models/{disease}_model.pkl",
        "len(all_metrics) == 3",
        lambda b=bundle: len(b["all_metrics"]) == 3)


# ================================================================
# MODULE 3 — predict.py
# ================================================================
section("MODULE 3: predict.py")

VALID = {
    "age": 45, "bmi": 28.5, "blood_pressure": 130,
    "glucose": 148, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 1, "activity_level": 1,
}
LOW = {
    "age": 25, "bmi": 21.0, "blood_pressure": 110,
    "glucose": 85, "cholesterol": 160, "heart_rate": 65,
    "smoking_status": 0, "activity_level": 3,
}
preds = predict_all(VALID)

run("predict", "predict_all returns dict with 4 disease keys",
    "valid 8-field input",
    "keys: diabetes, heart, hypertension, stroke",
    lambda: set(preds.keys()) == {"diabetes","heart","hypertension","stroke"})

run("predict", "each result has label, risk, confidence, error keys",
    "valid input",
    "all 4 keys present in each disease result",
    lambda: all(
        all(k in preds[d] for k in ["label","risk","confidence","error"])
        for d in preds))

run("predict", "risk values are valid labels",
    "valid input",
    "High Risk, Medium Risk, or Low Risk",
    lambda: all(
        preds[d]["risk"] in ("High Risk","Medium Risk","Low Risk")
        for d in preds))

run("predict", "confidence is a float between 0 and 100",
    "valid input",
    "0.0 <= confidence <= 100.0",
    lambda: all(
        preds[d]["confidence"] is None or
        0.0 <= preds[d]["confidence"] <= 100.0
        for d in preds))

run("predict", "error is None for all diseases on valid input",
    "valid input",
    "error == None for all 4 diseases",
    lambda: all(preds[d]["error"] is None for d in preds))

run("predict", "low-risk profile returns results without error",
    "age=25, glucose=85, bp=110",
    "all 4 diseases return error=None",
    lambda: all(predict_all(LOW)[d]["error"] is None
                for d in predict_all(LOW)))

run("predict", "_confidence_to_label: pred=0 → Low Risk",
    "prediction=0, confidence=95%",
    "Low Risk",
    lambda: _confidence_to_label(0, 95.0) == "Low Risk")

run("predict", "_confidence_to_label: pred=1, conf>=60 → High Risk",
    "prediction=1, confidence=80%",
    "High Risk",
    lambda: _confidence_to_label(1, 80.0) == "High Risk")

run("predict", "_confidence_to_label: pred=1, conf<60 → Medium Risk",
    "prediction=1, confidence=50%",
    "Medium Risk",
    lambda: _confidence_to_label(1, 50.0) == "Medium Risk")

# Invalid / edge cases
run("predict", "empty dict raises ValueError",
    "predict_all({})",
    "ValueError",
    lambda: raises(lambda: predict_all({}), ValueError))

run("predict", "non-dict raises TypeError",
    'predict_all("bad")',
    "TypeError",
    lambda: raises(lambda: predict_all("bad"), TypeError))

run("predict", "missing field raises ValueError",
    "predict_all({'age': 35}) — 7 fields missing",
    "ValueError",
    lambda: raises(lambda: predict_all({"age": 35}), ValueError))

run("predict", "string field raises ValueError",
    "age='old'",
    "ValueError",
    lambda: raises(
        lambda: predict_all({**VALID, "age": "old"}), ValueError))

run("predict", "age out of range raises ValueError",
    "age=-5",
    "ValueError",
    lambda: raises(
        lambda: predict_all({**VALID, "age": -5}), ValueError))

run("predict", "extreme high values complete without error",
    "age=120, glucose=600, bp=300",
    "all 4 diseases return error=None",
    lambda: all(
        predict_all({**VALID, "age":120,"glucose":600,"blood_pressure":300})[d]["error"] is None
        for d in preds))

run("predict", "path traversal raises ValueError",
    'load_model("../config.py")',
    "ValueError",
    lambda: raises(lambda: load_model("../config.py"), ValueError))

run("predict", "missing model file raises FileNotFoundError",
    'load_model("models/fake.pkl")',
    "FileNotFoundError",
    lambda: raises(
        lambda: load_model(str(ROOT / "models" / "fake.pkl")),
        FileNotFoundError))


# ================================================================
# MODULE 4 — recommendations.py
# ================================================================
section("MODULE 4: recommendations.py")

EXPECTED_CATS = {"diet","exercise","monitoring","doctor"}

for disease in ["diabetes","heart_disease","hypertension"]:
    for risk in ["High Risk","Medium Risk","Low Risk"]:
        r = get_recommendations(disease, risk)
        run("recommendations",
            f"{disease} {risk}: 4 categories returned",
            f"disease='{disease}', risk='{risk}'",
            "dict with diet, exercise, monitoring, doctor",
            lambda r=r: (r["error"] is None and
                         set(r["suggestions"].keys()) == EXPECTED_CATS))
        run("recommendations",
            f"{disease} {risk}: all categories non-empty",
            f"disease='{disease}', risk='{risk}'",
            "each category has at least 1 tip",
            lambda r=r: all(len(v) > 0
                            for v in r["suggestions"].values()))

run("recommendations", "unknown disease returns error dict",
    "disease='cancer'",
    "error set, suggestions={}",
    lambda: (lambda r: r["error"] is not None and r["suggestions"] == {})(
        get_recommendations("cancer","High Risk")))

run("recommendations", "unknown risk level returns error dict",
    "risk='Medium'",
    "error set, suggestions={}",
    lambda: (lambda r: r["error"] is not None and r["suggestions"] == {})(
        get_recommendations("diabetes","Medium")))

run("recommendations", "empty disease string returns error dict",
    "disease=''",
    "error set",
    lambda: get_recommendations("","High Risk")["error"] is not None)

run("recommendations", "None disease returns error dict",
    "disease=None",
    "error set",
    lambda: get_recommendations(None,"High Risk")["error"] is not None)

run("recommendations", "get_all_recommendations: 3 diseases processed",
    "flat dict with 3 risk values",
    "output has 3 keys",
    lambda: len(get_all_recommendations({
        "diabetes":"High Risk",
        "heart_disease":"Low Risk",
        "hypertension":"Medium Risk",
    })) == 3)

run("recommendations", "non-dict input returns error for all 3 diseases",
    "get_all_recommendations('bad')",
    "all 3 diseases have error set",
    lambda: all(
        v["error"] is not None
        for v in get_all_recommendations("bad").values()))

run("recommendations", "empty dict returns error for all 3 diseases",
    "get_all_recommendations({})",
    "all 3 diseases have error set",
    lambda: all(
        v["error"] is not None
        for v in get_all_recommendations({}).values()))

run("recommendations", "CATEGORY_LABELS has all 4 keys",
    "CATEGORY_LABELS dict",
    "diet, exercise, monitoring, doctor",
    lambda: set(CATEGORY_LABELS.keys()) == EXPECTED_CATS)


# ================================================================
# MODULE 5 — database.py
# ================================================================
section("MODULE 5: database.py")

init_db()
clear_history()

SAMPLE = {
    "age": 45, "bmi": 27.5, "blood_pressure": 130,
    "glucose": 110, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 0, "activity_level": 2,
    "diabetes_risk": "Low Risk",     "diabetes_confidence": 91.0,
    "heart_risk": "High Risk",       "heart_confidence": 82.4,
    "hypertension_risk": "Low Risk", "hypertension_confidence": 88.0,
    "recommendations_summary": "testing plan test",
}

row_id = insert_prediction(SAMPLE)

run("database", "insert_prediction returns integer row ID",
    "valid 15-key record dict",
    "int >= 1",
    lambda: isinstance(row_id, int) and row_id >= 1)

run("database", "partial record inserts (missing keys → None)",
    "record with only age and bmi",
    "int row ID returned",
    lambda: isinstance(insert_prediction({"age":30,"bmi":22.0}), int))

run("database", "string input returns None",
    'insert_prediction("bad")',
    "None",
    lambda: insert_prediction("bad") is None)

run("database", "empty dict returns None",
    "insert_prediction({})",
    "None",
    lambda: insert_prediction({}) is None)

run("database", "list input returns None",
    "insert_prediction([1,2])",
    "None",
    lambda: insert_prediction([1,2]) is None)

history = fetch_history()

run("database", "fetch_history returns a list",
    "after 2 inserts",
    "list type",
    lambda: isinstance(history, list))

run("database", "fetch_history returns non-empty list after inserts",
    "after 2 inserts",
    "len >= 1",
    lambda: len(history) >= 1)

run("database", "each record is a dict",
    "fetch_history() result",
    "isinstance(record, dict)",
    lambda: isinstance(history[0], dict))

run("database", "all 17 columns present in each record",
    "fetch_history() result",
    "17 column keys",
    lambda: {
        "id","timestamp","age","bmi","blood_pressure","glucose",
        "cholesterol","heart_rate","smoking_status","activity_level",
        "diabetes_risk","diabetes_confidence","heart_risk","heart_confidence",
        "hypertension_risk","hypertension_confidence","recommendations_summary",
    }.issubset(set(history[0].keys())))

run("database", "records ordered newest first",
    "multiple records",
    "history[0]['id'] >= history[-1]['id']",
    lambda: len(history) < 2 or history[0]["id"] >= history[-1]["id"])

run("database", "clear_history empties the table",
    "clear_history() then fetch",
    "fetch_history() == []",
    lambda: (clear_history() or True) and fetch_history() == [])

run("database", "fetch_history returns [] on empty table",
    "empty table",
    "[]",
    lambda: fetch_history() == [])


# ================================================================
# MODULE 6 — validators.py
# ================================================================
section("MODULE 6: validators.py")

FULL_VALID = {
    "age":35, "bmi":25.0, "blood_pressure":120,
    "glucose":100, "cholesterol":200, "heart_rate":75,
}

run("validators", "valid input returns no errors and no warnings",
    "all 6 fields within normal range",
    "errors=[], warnings=[]",
    lambda: validate_health_inputs(FULL_VALID) == ([], []))

for field, bad_val, label in [
    ("age",            0,    "age=0 below min"),
    ("age",          121,    "age=121 above max"),
    ("bmi",          9.9,    "bmi=9.9 below min"),
    ("bmi",         80.1,    "bmi=80.1 above max"),
    ("blood_pressure", 39,   "bp=39 below min"),
    ("blood_pressure", 301,  "bp=301 above max"),
    ("glucose",       19,    "glucose=19 below min"),
    ("glucose",      601,    "glucose=601 above max"),
    ("cholesterol",   49,    "cholesterol=49 below min"),
    ("cholesterol",  701,    "cholesterol=701 above max"),
    ("heart_rate",    19,    "heart_rate=19 below min"),
    ("heart_rate",   301,    "heart_rate=301 above max"),
]:
    run("validators", f"out-of-range: {label}",
        f"{field}={bad_val}",
        "errors list non-empty",
        lambda f=field, v=bad_val: len(
            validate_health_inputs({**FULL_VALID, f: v})[0]) > 0)

for field, warn_val, label in [
    ("glucose",        30,   "glucose=30 → low glucose warning"),
    ("glucose",       510,   "glucose=510 → high glucose warning"),
    ("blood_pressure",  55,  "bp=55 → low BP warning"),
    ("blood_pressure", 260,  "bp=260 → high BP warning"),
    ("heart_rate",     25,   "heart_rate=25 → low HR warning"),
    ("heart_rate",    225,   "heart_rate=225 → high HR warning"),
    ("bmi",           11.0,  "bmi=11 → low BMI warning"),
    ("bmi",           72.0,  "bmi=72 → high BMI warning"),
]:
    run("validators", f"clinical warning: {label}",
        f"{field}={warn_val}",
        "errors=[], warnings non-empty",
        lambda f=field, v=warn_val: (
            lambda e, w: not e and len(w) > 0
        )(*validate_health_inputs({**FULL_VALID, f: v})))

run("validators", "missing field returns error",
    "dict without 'glucose'",
    "errors non-empty",
    lambda: len(validate_health_inputs(
        {k:v for k,v in FULL_VALID.items() if k != "glucose"})[0]) > 0)

run("validators", "string value returns error",
    "age='thirty-five'",
    "errors non-empty",
    lambda: len(validate_health_inputs(
        {**FULL_VALID, "age": "thirty-five"})[0]) > 0)

run("validators", "non-dict input returns error",
    '"not a dict"',
    "errors non-empty",
    lambda: len(validate_health_inputs("not a dict")[0]) > 0)


# ================================================================
# EDGE CASES SUMMARY
# ================================================================
section("EDGE CASES SUMMARY")

run("edge_cases", "predict_all: all extreme high values complete",
    "age=120, glucose=600, bp=300, bmi=80",
    "all 4 diseases return error=None",
    lambda: all(
        predict_all({
            "age":120,"bmi":80.0,"blood_pressure":300,
            "glucose":600,"cholesterol":700,"heart_rate":300,
            "smoking_status":1,"activity_level":0,
        })[d]["error"] is None for d in preds))

run("edge_cases", "predict_all: all minimum valid values complete",
    "age=1, glucose=20, bp=40, bmi=10",
    "all 4 diseases return error=None",
    lambda: all(
        predict_all({
            "age":1,"bmi":10.0,"blood_pressure":40,
            "glucose":20,"cholesterol":50,"heart_rate":20,
            "smoking_status":0,"activity_level":0,
        })[d]["error"] is None for d in preds))

run("edge_cases", "recommendations: Error: prefix propagated",
    "predictions with 'Error: model failed'",
    "error set in output",
    lambda: get_all_recommendations(
        {"diabetes":"Error: model failed"}
    )["diabetes"]["error"] is not None)

run("edge_cases", "database: insert then fetch round-trip",
    "insert record, fetch, compare age field",
    "fetched age == inserted age",
    lambda: (lambda rid, h: h[0]["age"] == 52.0)(
        insert_prediction({**SAMPLE, "age":52}),
        fetch_history()))


# ================================================================
# PRINT REPORT TABLE
# ================================================================
passed = sum(1 for r in results if r[6] == "PASS")
failed = sum(1 for r in results if r[6] == "FAIL")
total  = len(results)

print(f"\n\n{'='*65}")
print(f"  TESTING PLAN — RESULTS SUMMARY")
print(f"  Project: Hybrid Intelligent Health Risk Prediction System")
print(f"  Date   : {datetime.now().strftime('%d %B %Y')}")
print(f"{'='*65}")
print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}\n")

# Group by module
modules_order = [
    "preprocess","train_model","predict",
    "recommendations","database","validators","edge_cases",
]
module_labels = {
    "preprocess":      "preprocess.py",
    "train_model":     "train_model.py",
    "predict":         "predict.py",
    "recommendations": "recommendations.py",
    "database":        "database.py",
    "validators":      "validators.py",
    "edge_cases":      "Edge Cases",
}

for mod in modules_order:
    mod_results = [r for r in results if r[0] == mod]
    if not mod_results:
        continue
    mp = sum(1 for r in mod_results if r[6] == "PASS")
    mf = sum(1 for r in mod_results if r[6] == "FAIL")
    print(f"\n  {'─'*61}")
    print(f"  {module_labels[mod]}  ({mp}/{len(mod_results)} passed)")
    print(f"  {'─'*61}")
    print(f"  {'ID':<6} {'Test Description':<36} {'Status'}")
    print(f"  {'─'*61}")
    for _, tid, desc, inp, exp, act, status in mod_results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  {tid:<6} {desc[:35]:<36} {icon}")

print(f"\n{'='*65}")
print(f"  FINAL RESULT: {'ALL TESTS PASSED' if failed == 0 else f'{failed} TEST(S) FAILED'}")
print(f"{'='*65}\n")

# ── Report table for project report (plain text) ─────────────────
print("\n" + "="*65)
print("  TABLE FOR PROJECT REPORT (copy into Word/LaTeX)")
print("="*65)
print(f"\n{'No.':<5} {'Module':<18} {'Test Description':<32} {'Expected':<20} {'Result'}")
print("-"*95)
for i, (mod, tid, desc, inp, exp, act, status) in enumerate(results, 1):
    print(f"{i:<5} {module_labels.get(mod, mod):<18} {desc[:31]:<32} {exp[:19]:<20} {status}")
print("-"*95)
print(f"\nTotal: {total}  Passed: {passed}  Failed: {failed}")
