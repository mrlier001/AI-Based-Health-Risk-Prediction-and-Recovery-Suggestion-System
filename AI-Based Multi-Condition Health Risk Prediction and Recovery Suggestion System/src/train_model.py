"""
train_model.py
--------------
Phase 4 — Model Training and Comparison

Trains three machine learning models for each disease, compares their
performance on a held-out test set, selects the best one, and saves
it as a .pkl bundle.

Diseases trained:
    - diabetes      -> models/diabetes_model.pkl
    - heart         -> models/heart_model.pkl
    - hypertension  -> models/hypertension_model.pkl
    - stroke        -> models/stroke_model.pkl
    - kidney        -> models/kidney_model.pkl
    - lung          -> models/lung_model.pkl

Models compared for each disease:
    - Logistic Regression  (linear, fast, good baseline)
    - Decision Tree        (non-linear, interpretable, can overfit)
    - Random Forest        (ensemble of trees, robust, usually best)

Metrics used for comparison:
    - Accuracy   : overall correct predictions
    - Precision  : of all High Risk predictions, how many were correct
    - Recall     : of all actual High Risk cases, how many were found
    - F1 Score   : balance of Precision and Recall — used to select best

Why F1 and not Accuracy?
    Our datasets are imbalanced (more Low Risk than High Risk cases).
    A model that always predicts Low Risk can score high accuracy but
    will have F1 = 0. F1 penalises this and rewards models that
    correctly identify High Risk cases — which is the clinical goal.

Each saved .pkl bundle contains:
    "model"           : the best trained sklearn classifier
    "scaler"          : the fitted StandardScaler from preprocessing
    "features"        : list of feature column names in training order
    "all_metrics"     : metrics for all three models (used in UI)
    "best_model_name" : name of the model that was selected

Usage — run from the project root folder:
    python src/train_model.py
"""

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Add project root to sys.path so config and src can be imported
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_FILES, TARGET_COLUMNS, MODEL_FILES, MODELS_DIR, RANDOM_STATE
from src.preprocess import load_dataset, preprocess_data

# All six diseases — kidney and lung added; order controls training sequence
DISEASES = ["diabetes", "heart", "hypertension", "stroke", "kidney", "lung"]


# =============================================================
# FUNCTION 1 — get_candidate_models
# =============================================================

def get_candidate_models() -> dict:
    """
    Return one fresh, untrained instance of each candidate model.

    All models use RANDOM_STATE from config.py so results are
    reproducible — running training twice gives the same output.

    Logistic Regression uses max_iter=1000 to ensure it converges
    on all four datasets without printing convergence warnings.

    Returns
    -------
    dict  {model_name: untrained sklearn classifier}
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================
# FUNCTION 2 — evaluate_model
# =============================================================

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on the test set and return four metrics.

    Parameters
    ----------
    model  : trained sklearn classifier
    X_test : numpy array — scaled test features
    y_test : pandas Series — true labels

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
        All values are floats rounded to 4 decimal places.

    Notes
    -----
    zero_division=0 silences warnings when a class has no predicted
    samples — this can happen on very small or imbalanced test sets.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }


# =============================================================
# FUNCTION 3 — print_comparison
# =============================================================

def print_comparison(disease: str, all_metrics: dict, best_name: str):
    """
    Print a detailed, readable comparison of all three models.

    Output includes:
        - A ranked metrics table (best model at the top)
        - A one-line verdict for each model
        - A plain-English explanation of why the best was chosen

    Uses only ASCII characters to avoid encoding errors on Windows
    terminals that use cp1252 instead of UTF-8.

    Parameters
    ----------
    disease     : str   disease name for the header
    all_metrics : dict  {model_name: {accuracy, precision, recall, f1}}
    best_name   : str   name of the model selected as best
    """
    SEP   = "=" * 64
    INNER = "-" * 60

    print(f"\n{SEP}")
    print(f"  DISEASE  : {disease.upper()}")
    print(f"  DATASET  : {Path(DATA_FILES[disease]).name}")
    print(f"  FEATURES : {_feature_count(disease)}")
    print(f"  SPLIT    : 80% train / 20% test  (random_state={RANDOM_STATE})")
    print(SEP)

    # Sort models by F1 score descending so best appears first
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)

    print(f"\n  {'Rank':<6} {'Model':<22} {'Accuracy':>9} {'Precision':>9} "
          f"{'Recall':>7} {'F1 Score':>9}")
    print(f"  {INNER}")

    for rank, (name, m) in enumerate(ranked, start=1):
        marker = "  <-- SELECTED" if name == best_name else ""
        print(
            f"  {rank:<6} {name:<22}"
            f"  {m['accuracy']:.4f}   "
            f"  {m['precision']:.4f}   "
            f"  {m['recall']:.4f}   "
            f"  {m['f1']:.4f}"
            f"{marker}"
        )

    print(f"\n  {INNER}")
    print("  MODEL VERDICTS")
    print(f"  {INNER}")

    for name, m in all_metrics.items():
        verdict = _verdict(name, m, best_name)
        print(f"  {name:<22} : {verdict}")

    print(f"\n  {INNER}")
    print("  SELECTION REASON")
    print(f"  {INNER}")

    best_m = all_metrics[best_name]
    print(
        f"  Winner  : {best_name}\n"
        f"  F1      : {best_m['f1']:.4f}  "
        f"(Precision={best_m['precision']:.4f}, Recall={best_m['recall']:.4f})\n"
        f"  Reason  : F1 Score was used as the selection criterion because\n"
        f"            it balances Precision and Recall. A model that only\n"
        f"            predicts Low Risk would score high Accuracy but F1=0.\n"
        f"            F1 rewards models that correctly identify High Risk cases."
    )
    print(f"{SEP}\n")


def _feature_count(disease: str) -> str:
    """Return feature count string from saved bundle, or 'unknown'."""
    try:
        import joblib as jl
        p = Path(MODEL_FILES[disease])
        if p.exists():
            b = jl.load(p)
            return str(len(b.get("features", [])))
    except Exception:
        pass
    return "unknown"


def _verdict(name: str, m: dict, best_name: str) -> str:
    """Return a one-line plain-English verdict for one model."""
    f1  = m["f1"]
    acc = m["accuracy"]
    rec = m["recall"]

    if name == best_name:
        return f"BEST — F1={f1:.4f}, selected for deployment"

    if f1 >= 0.90:
        return f"Strong — F1={f1:.4f}, close runner-up"
    if f1 >= 0.75:
        return f"Good — F1={f1:.4f}, acceptable performance"
    if rec < 0.50:
        return (f"Weak recall ({rec:.4f}) — misses too many High Risk cases, "
                f"F1={f1:.4f}")
    return f"Below threshold — F1={f1:.4f}, not selected"


# =============================================================
# FUNCTION 4 — train_and_save
# =============================================================

def train_and_save(disease: str) -> dict:
    """
    Run the full training pipeline for one disease.

    Steps:
        1. Load the CSV dataset using load_dataset()
        2. Preprocess using preprocess_data() — returns a dict
        3. Train all three candidate models on the training split
        4. Evaluate each model on the held-out test split
        5. Print a detailed comparison table
        6. Select the best model by F1 score
        7. Save the best model as a .pkl bundle to models/

    Parameters
    ----------
    disease : str
        One of: 'diabetes', 'heart', 'hypertension', 'stroke'

    Returns
    -------
    dict with keys:
        "model"           : best trained sklearn classifier
        "scaler"          : fitted StandardScaler
        "features"        : list of feature column names in training order
        "all_metrics"     : metrics dict for all three models
        "best_model_name" : name of the winning model

    Raises
    ------
    KeyError   if disease is not in DATA_FILES or TARGET_COLUMNS
    ValueError if the resolved save path escapes MODELS_DIR (CWE-22)
    """
    filepath   = DATA_FILES[disease]
    target_col = TARGET_COLUMNS[disease]
    save_path  = MODEL_FILES[disease]

    # Step 1 & 2 — Load and preprocess
    # preprocess_data() returns a dict — never a tuple
    df     = load_dataset(filepath)
    result = preprocess_data(df, target_col)

    X_train       = result["X_train"]
    X_test        = result["X_test"]
    y_train       = result["y_train"]
    y_test        = result["y_test"]
    scaler        = result["scaler"]
    feature_names = result["feature_names"]

    # Step 3 & 4 — Train all models and evaluate on test set
    all_metrics: dict = {}  # model_name -> {accuracy, precision, recall, f1}
    trained:     dict = {}  # model_name -> fitted model object

    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        all_metrics[name] = evaluate_model(model, X_test, y_test)
        trained[name]     = model

    # Step 5 — Select best model by F1 score
    # F1 is preferred over accuracy for imbalanced health datasets.
    # See print_comparison() for the full explanation.
    best_name  = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
    best_model = trained[best_name]

    # Step 6 — Print detailed comparison
    print_comparison(disease, all_metrics, best_name)

    # Step 7 — Save the bundle
    # Resolve path and confirm it stays inside MODELS_DIR (CWE-22 guard).
    resolved_save = Path(save_path).resolve()
    if not str(resolved_save).startswith(str(MODELS_DIR.resolve())):
        raise ValueError(
            f"save_path must be inside the models/ folder. "
            f"Got: '{save_path}'"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Bundle structure — must stay compatible with predict.py
    # predict.py reads: bundle["model"], bundle["scaler"],
    #                   bundle["features"], bundle["all_metrics"],
    #                   bundle["best_model_name"]
    bundle = {
        "model":           best_model,
        "scaler":          scaler,
        "features":        feature_names,
        "all_metrics":     all_metrics,
        "best_model_name": best_name,
    }
    joblib.dump(bundle, resolved_save)
    print(f"  Saved : {resolved_save.name}  ({resolved_save.stat().st_size} bytes)\n")

    return bundle


# =============================================================
# FUNCTION 5 — train_all
# =============================================================

def train_all():
    """
    Train and save models for all six diseases in sequence.

    Prints a comparison table for each disease, then a final
    summary showing which model was selected for each one.

    Called when this script is run directly:
        python src/train_model.py
    """
    print("\n" + "=" * 64)
    print("  HYBRID INTELLIGENT HEALTH RISK PREDICTION SYSTEM")
    print("  Model Training and Comparison")
    print("=" * 64)
    print(f"\n  Training {len(DISEASES)} disease model(s): "
          f"{', '.join(DISEASES)}")
    print("  Algorithms : Logistic Regression, Decision Tree, Random Forest")
    print("  Selection  : Best F1 Score on held-out test set (20%)\n")

    summary = {}  # disease -> best_model_name

    for disease in DISEASES:
        bundle = train_and_save(disease)
        summary[disease] = bundle["best_model_name"]

    # Final summary table
    print("=" * 64)
    print("  TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 64)
    print(f"\n  {'Disease':<16} {'Best Model':<22} {'Saved File'}")
    print(f"  {'-'*58}")
    for disease, best in summary.items():
        fname = Path(MODEL_FILES[disease]).name
        print(f"  {disease:<16} {best:<22} {fname}")
    print(f"\n  All models saved to: {MODELS_DIR}\n")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    train_all()
