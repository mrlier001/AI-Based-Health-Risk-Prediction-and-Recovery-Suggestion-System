"""
validators.py
-------------
Shared input validation for the healthcare AI project.

This module is the single source of truth for all field rules.
Both app.py (UI validation) and predict.py (pipeline validation)
use the same rules so behaviour is consistent everywhere.

Public functions
----------------
validate_health_inputs(inputs)
    Validates a dict of health measurements.
    Returns a list of error strings — empty list means all valid.

FIELD_RULES
    Dict of field -> (min, max, label, unit, hint) used by both
    the validator and app.py to build helpful error messages.
"""

# =============================================================
# FIELD RULES
# Each entry: field_key -> (min, max, display_label, unit, clinical_hint)
# =============================================================

FIELD_RULES = {
    "age": (
        1, 120,
        "Age", "years",
        "Enter your age in whole years (1-120).",
    ),
    "bmi": (
        10.0, 80.0,
        "BMI", "kg/m²",
        "Body Mass Index between 10.0 and 80.0. "
        "A healthy adult BMI is typically 18.5-24.9.",
    ),
    "blood_pressure": (
        40, 300,
        "Blood Pressure", "mmHg",
        "Systolic blood pressure between 40 and 300 mmHg. "
        "A normal reading is around 90-120 mmHg.",
    ),
    "glucose": (
        20, 600,
        "Glucose", "mg/dL",
        "Fasting blood glucose between 20 and 600 mg/dL. "
        "A normal fasting level is 70-100 mg/dL.",
    ),
    "cholesterol": (
        50, 700,
        "Cholesterol", "mg/dL",
        "Total cholesterol between 50 and 700 mg/dL. "
        "A desirable level is below 200 mg/dL.",
    ),
    "heart_rate": (
        20, 300,
        "Heart Rate", "bpm",
        "Resting heart rate between 20 and 300 bpm. "
        "A normal resting rate is 60-100 bpm.",
    ),
}

# Clinical warning thresholds — values that are technically valid
# but unusual enough to show a soft warning in the UI.
# Format: field -> list of (condition_fn, warning_message)
_CLINICAL_WARNINGS = {
    "glucose": [
        (lambda v: v < 54,  "Glucose below 54 mg/dL is clinically very low. Please verify this reading."),
        (lambda v: v > 500, "Glucose above 500 mg/dL is extremely high. Please verify this reading."),
    ],
    "blood_pressure": [
        (lambda v: v < 60,  "Blood pressure below 60 mmHg is very low. Please verify this reading."),
        (lambda v: v > 250, "Blood pressure above 250 mmHg is extremely high. Please verify this reading."),
    ],
    "heart_rate": [
        (lambda v: v < 30,  "Heart rate below 30 bpm is very low. Please verify this reading."),
        (lambda v: v > 220, "Heart rate above 220 bpm is extremely high. Please verify this reading."),
    ],
    "bmi": [
        (lambda v: v < 12.0, "BMI below 12.0 is extremely low. Please verify this value."),
        (lambda v: v > 70.0, "BMI above 70.0 is extremely high. Please verify this value."),
    ],
}


def validate_health_inputs(inputs: dict) -> tuple:
    """
    Validate a dict of health measurement values.

    Checks each field in FIELD_RULES for:
        - presence in the dict
        - correct numeric type (int or float)
        - value within the allowed min/max range

    Also checks for clinically unusual values and returns soft
    warnings for those — they do not block submission.

    Parameters
    ----------
    inputs : dict
        Must contain keys matching FIELD_RULES.
        Extra keys are ignored.

    Returns
    -------
    tuple (errors, warnings)
        errors   : list of str — hard errors that must be fixed
        warnings : list of str — soft warnings shown to the user
                   but do not block submission

    Examples
    --------
    errors, warnings = validate_health_inputs({"age": -5, ...})
    # errors = ["Age must be between 1 and 120 years."]

    errors, warnings = validate_health_inputs({"age": 35, "glucose": 30, ...})
    # errors   = []
    # warnings = ["Glucose below 54 mg/dL is clinically very low..."]
    """
    if not isinstance(inputs, dict):
        return [f"inputs must be a dict. Got: {type(inputs).__name__}"], []

    errors   = []
    warnings = []

    for field, (lo, hi, label, unit, hint) in FIELD_RULES.items():
        value = inputs.get(field)

        # Missing field
        if value is None:
            errors.append(f"{label} is required.")
            continue

        # Wrong type
        if not isinstance(value, (int, float)):
            errors.append(
                f"{label} must be a number. Got: {type(value).__name__}."
            )
            continue

        # Out of range
        if not (lo <= value <= hi):
            errors.append(
                f"{label} must be between {lo} and {hi} {unit}. "
                f"You entered {value}. {hint}"
            )
            continue

        # Clinical warnings (soft — do not block)
        for condition, message in _CLINICAL_WARNINGS.get(field, []):
            if condition(value):
                warnings.append(message)

    return errors, warnings
