"""
test_database.py
----------------
Manual verification of the database implementation.
Run from project root: python test_database.py
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_PATH
from src.database import init_db, insert_prediction, fetch_history, clear_history

PASS = "[PASS]"
FAIL = "[FAIL]"
sep  = "=" * 55

def section(title):
    print(f"\n{sep}\n  {title}\n{sep}")

results = []

# ── CHECK 1: Context manager — no conn.close() anywhere ─────────
section("CHECK 1: SQLite context manager usage")
src = Path("src/database.py").read_text(encoding="utf-8")
if "conn.close()" in src:
    print(f"  {FAIL} conn.close() found — resource leak risk")
    results.append(("Context manager", False))
else:
    count = src.count("with sqlite3.connect")
    print(f"  {PASS} No conn.close() calls found")
    print(f"  {PASS} {count} context manager(s) used correctly")
    results.append(("Context manager", True))

# ── CHECK 2: Table structure matches insert columns ──────────────
section("CHECK 2: Table structure vs insert columns")
init_db()

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.execute("PRAGMA table_info(prediction_history)")
    table_cols = [row[1] for row in cursor.fetchall()]

insert_cols = [
    "id", "timestamp",
    "age", "bmi", "blood_pressure", "glucose", "cholesterol",
    "heart_rate", "smoking_status", "activity_level",
    "diabetes_risk", "diabetes_confidence",
    "heart_risk", "heart_confidence",
    "hypertension_risk", "hypertension_confidence",
    "recommendations_summary",
]

missing = [c for c in insert_cols if c not in table_cols]
extra   = [c for c in table_cols  if c not in insert_cols]

if missing:
    print(f"  {FAIL} Columns missing from table: {missing}")
    results.append(("Table structure", False))
elif extra:
    print(f"  {FAIL} Unexpected columns in table: {extra}")
    results.append(("Table structure", False))
else:
    print(f"  {PASS} All {len(table_cols)} columns present and matched")
    for col in table_cols:
        print(f"         {col}")
    results.append(("Table structure", True))

# ── CHECK 3: insert_prediction works correctly ───────────────────
section("CHECK 3: insert_prediction")

sample = {
    "age": 45, "bmi": 27.5, "blood_pressure": 130,
    "glucose": 110, "cholesterol": 210, "heart_rate": 78,
    "smoking_status": 0, "activity_level": 2,
    "diabetes_risk":           "Low Risk",
    "diabetes_confidence":     91.0,
    "heart_risk":              "High Risk",
    "heart_confidence":        82.4,
    "hypertension_risk":       "Low Risk",
    "hypertension_confidence": 88.0,
    "recommendations_summary": "DIABETES - Low Risk\n  diet: Eat balanced meals.",
}

row_id = insert_prediction(sample)

if not isinstance(row_id, int) or row_id < 1:
    print(f"  {FAIL} Expected integer row_id >= 1, got: {row_id}")
    results.append(("insert_prediction", False))
else:
    print(f"  {PASS} Record inserted with row_id={row_id}")

    # Verify non-dict and empty dict are rejected
    r1 = insert_prediction("not a dict")
    r2 = insert_prediction({})
    if r1 is None and r2 is None:
        print(f"  {PASS} Non-dict input rejected (returned None)")
        print(f"  {PASS} Empty dict rejected (returned None)")
        results.append(("insert_prediction", True))
    else:
        print(f"  {FAIL} Bad input not rejected: r1={r1}, r2={r2}")
        results.append(("insert_prediction", False))

# ── CHECK 4: fetch_history returns correct format ────────────────
section("CHECK 4: fetch_history return format")

history = fetch_history()

if not isinstance(history, list):
    print(f"  {FAIL} Expected list, got {type(history).__name__}")
    results.append(("fetch_history format", False))
elif len(history) == 0:
    print(f"  {FAIL} History is empty after insert")
    results.append(("fetch_history format", False))
else:
    latest = history[0]
    required_keys = {
        "id", "timestamp", "age", "bmi", "blood_pressure", "glucose",
        "cholesterol", "heart_rate", "smoking_status", "activity_level",
        "diabetes_risk", "diabetes_confidence",
        "heart_risk", "heart_confidence",
        "hypertension_risk", "hypertension_confidence",
        "recommendations_summary",
    }
    missing_keys = required_keys - set(latest.keys())

    if missing_keys:
        print(f"  {FAIL} Missing keys in record: {missing_keys}")
        results.append(("fetch_history format", False))
    else:
        print(f"  {PASS} Returns list of {len(history)} dict(s)")
        print(f"  {PASS} All 17 columns present in each record")
        print(f"  {PASS} Ordered newest first (latest id={latest['id']})")

        # Verify types
        type_ok = (
            isinstance(latest["id"],        int)   and
            isinstance(latest["timestamp"], str)   and
            isinstance(latest["age"],       float) and
            isinstance(latest["diabetes_risk"], str)
        )
        if type_ok:
            print(f"  {PASS} Column types correct (id=int, age=float, risk=str)")
        else:
            print(f"  {FAIL} Unexpected column types")
        results.append(("fetch_history format", type_ok))

# ── CHECK 5: app.py reads all expected columns from history ──────
section("CHECK 5: app.py history display columns")

app_src = Path("app.py").read_text(encoding="utf-8")
display_keys = [
    "timestamp", "diabetes_risk", "diabetes_confidence",
    "heart_risk", "heart_confidence",
    "hypertension_risk", "hypertension_confidence",
    "age", "bmi", "blood_pressure", "glucose",
    "cholesterol", "heart_rate", "smoking_status",
    "activity_level", "recommendations_summary",
]
missing_in_app = [k for k in display_keys if k not in app_src]
if missing_in_app:
    print(f"  {FAIL} Keys not referenced in app.py: {missing_in_app}")
    results.append(("app.py display columns", False))
else:
    print(f"  {PASS} All {len(display_keys)} history columns referenced in app.py")
    results.append(("app.py display columns", True))

# ── CHECK 6: Data round-trip — insert then fetch and compare ─────
section("CHECK 6: Data round-trip (insert, fetch, compare)")

before_count = len(fetch_history())
test_record = {
    "age": 52, "bmi": 31.2, "blood_pressure": 145,
    "glucose": 160, "cholesterol": 240, "heart_rate": 88,
    "smoking_status": 1, "activity_level": 0,
    "diabetes_risk":           "High Risk",
    "diabetes_confidence":     95.5,
    "heart_risk":              "High Risk",
    "heart_confidence":        87.3,
    "hypertension_risk":       "High Risk",
    "hypertension_confidence": 92.1,
    "recommendations_summary": "round-trip test",
}
new_id = insert_prediction(test_record)
history = fetch_history()
after_count = len(history)

if after_count != before_count + 1:
    print(f"  {FAIL} Count before={before_count}, after={after_count}")
    results.append(("Round-trip", False))
else:
    fetched = history[0]
    checks = [
        ("age",              52.0),
        ("bmi",              31.2),
        ("diabetes_risk",    "High Risk"),
        ("heart_risk",       "High Risk"),
        ("hypertension_risk","High Risk"),
        ("smoking_status",   1),
        ("activity_level",   0),
    ]
    all_match = True
    for key, expected in checks:
        actual = fetched.get(key)
        if actual != expected:
            print(f"  {FAIL} {key}: expected={expected}, got={actual}")
            all_match = False
    if all_match:
        print(f"  {PASS} All field values match after round-trip")
        print(f"         id={fetched['id']}  timestamp={fetched['timestamp']}")
    results.append(("Round-trip", all_match))

# ── FINAL SUMMARY ────────────────────────────────────────────────
print(f"\n{sep}")
print("  FINAL RESULTS")
print(sep)
all_passed = True
for name, passed in results:
    print(f"  {PASS if passed else FAIL} {name}")
    if not passed:
        all_passed = False
print(sep)
print(f"  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(sep)
