"""
evaluate_models.py
------------------
Standalone model evaluation script for the healthcare AI project.

Reads the evaluation metrics that were saved during training and
prints a formatted comparison table for all three diseases.

No retraining is needed — metrics are read directly from the
saved .pkl model bundles.

Usage (run from the project root folder):
    python evaluate_models.py

Output:
    A comparison table for each disease showing Accuracy, Precision,
    Recall, and F1 Score for Logistic Regression, Decision Tree,
    and Random Forest. The selected model is marked.
"""

import sys
import joblib
from pathlib import Path

# Add project root to sys.path so config can be imported
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MODEL_FILES

# Diseases to evaluate — must match what was trained in train_model.py
DISEASES = {
    "diabetes":     "Diabetes",
    "heart":        "Heart Disease",
    "hypertension": "Hypertension",
}

# Metric explanations printed once at the top
METRIC_GUIDE = """
METRIC GUIDE
------------
Accuracy  : Percentage of all predictions that were correct.
Precision : Of all cases predicted as High Risk, how many actually were.
            High precision = fewer false alarms.
Recall    : Of all actual High Risk cases, how many were correctly found.
            High recall = fewer missed cases.
F1 Score  : Balance between Precision and Recall.
            Used to select the best model. 1.0 = perfect, 0.0 = worst.
"""


def print_disease_results(disease_key: str, disease_label: str):
    """
    Load the saved model bundle for one disease and print its
    evaluation metrics in a formatted table.

    Parameters
    ----------
    disease_key   : str  key used in MODEL_FILES (e.g. 'diabetes')
    disease_label : str  human-readable name (e.g. 'Diabetes')
    """
    model_path = Path(MODEL_FILES[disease_key]).resolve()

    # CWE-22 path traversal guard: confirm path stays inside models/
    from config import MODELS_DIR
    if not str(model_path).startswith(str(MODELS_DIR.resolve())):
        print(f"  [{disease_label}] Unsafe model path rejected: '{model_path}'")
        return

    if not model_path.exists():
        print(f"\n  [{disease_label}] Model file not found: '{model_path.name}'")
        print(f"  Run 'python src/train_model.py' first.\n")
        return

    bundle      = joblib.load(model_path)
    all_metrics = bundle.get("all_metrics", {})
    best_name   = bundle.get("best_model_name", "")

    if not all_metrics:
        print(f"\n  [{disease_label}] No metrics stored in bundle.")
        print(f"  Retrain with 'python src/train_model.py' to generate metrics.\n")
        return

    # Print table header
    sep   = "=" * 65
    inner = "-" * 61
    print(f"\n{sep}")
    print(f"  Disease : {disease_label.upper()}")
    print(f"  Best    : {best_name}")
    print(f"{sep}")
    print(f"  {'Algorithm':<22} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1 Score':>9}  {'':>10}")
    print(f"  {inner}")

    for model_name, m in all_metrics.items():
        marker = "  <-- selected" if model_name == best_name else ""
        print(
            f"  {model_name:<22}"
            f"  {m['accuracy']:.4f}   "
            f"  {m['precision']:.4f}   "
            f"  {m['recall']:.4f}   "
            f"  {m['f1']:.4f}"
            f"{marker}"
        )

    print(f"{sep}")

    # Plain-English summary
    best_m = all_metrics[best_name]
    print(
        f"\n  Summary: The {best_name} was selected for {disease_label}.\n"
        f"  It achieved an F1 Score of {best_m['f1']:.4f}, "
        f"Accuracy of {best_m['accuracy']:.4f},\n"
        f"  Precision of {best_m['precision']:.4f}, "
        f"and Recall of {best_m['recall']:.4f}.\n"
    )


def main():
    print("\n" + "=" * 65)
    print("  MODEL EVALUATION REPORT")
    print("  Hybrid Intelligent Multi-Condition Health Risk Prediction")
    print("=" * 65)
    print(METRIC_GUIDE)

    for disease_key, disease_label in DISEASES.items():
        print_disease_results(disease_key, disease_label)

    print("=" * 65)
    print("  End of evaluation report.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
