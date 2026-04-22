"""
preprocess.py
-------------
Phase 3 — Data Loading and Preprocessing

This module handles everything needed to prepare a CSV dataset
for machine learning training.

Functions
---------
load_dataset(file_path)
    Validates the file path and loads the CSV into a DataFrame.

preprocess_data(df, target_column)
    Cleans, encodes, scales, and splits the DataFrame.
    Always returns a dictionary — never a tuple.

scale_user_input(user_dict, feature_names, scaler)
    Scales one row of user input at prediction time.
    Used by predict.py, not during training.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow config.py to be imported from the project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TEST_SIZE, RANDOM_STATE, DATA_DIR


# =============================================================
# FUNCTION 1 — load_dataset
# =============================================================

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Validate a file path and load the CSV into a pandas DataFrame.

    This function only loads the raw data.
    Cleaning and transformation are handled by preprocess_data().

    Parameters
    ----------
    file_path : str
        Path to the CSV file, e.g. "data/diabetes.csv"

    Returns
    -------
    pd.DataFrame
        The raw loaded DataFrame with all original columns intact.

    Raises
    ------
    TypeError
        If file_path is not a string.
    ValueError
        If file_path is an empty string or does not end with .csv
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the CSV loads successfully but contains no data rows.
    """

    # --- Validation 1: must be a string ---
    if not isinstance(file_path, str):
        raise TypeError(
            f"file_path must be a string. "
            f"Got: {type(file_path).__name__}"
        )

    # --- Validation 2: must not be empty or only spaces ---
    if not file_path.strip():
        raise ValueError(
            "file_path must not be an empty string. "
            "Provide a valid path to a CSV file."
        )

    # --- Validation 3: must end with .csv ---
    if not file_path.strip().lower().endswith(".csv"):
        raise ValueError(
            f"file_path must point to a .csv file. "
            f"Got: '{file_path}'"
        )

    # --- Validation 4: path must not escape the data/ directory ---
    # Resolving the path converts any "../" sequences to their real location.
    # We then confirm the resolved path starts inside DATA_DIR.
    # This prevents path traversal attacks (CWE-22).
    path = Path(file_path).resolve()
    if not str(path).startswith(str(DATA_DIR.resolve())):
        raise ValueError(
            f"file_path must point to a file inside the data/ folder. "
            f"Got: '{file_path}'"
        )

    # --- Validation 5: file must exist on disk ---
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{file_path}'\n"
            f"Make sure the CSV file is placed inside the data/ folder."
        )

    # --- Load the CSV ---
    df = pd.read_csv(path)
    print(f"[load_dataset] '{path.name}': {df.shape[0]} rows, {df.shape[1]} cols")

    # --- Validation 5: must not be empty after loading ---
    if df.empty:
        raise ValueError(
            f"The CSV at '{file_path}' loaded successfully "
            f"but contains no data rows."
        )

    return df


# =============================================================
# FUNCTION 2 — preprocess_data
# =============================================================

def preprocess_data(df: pd.DataFrame, target_column: str) -> dict:
    """
    Clean, encode, scale, and split a DataFrame for ML training.

    Steps performed:
        1. Validate inputs
        2. Drop rows with missing values
        3. One-hot encode any text (categorical) columns
        4. Separate features (X) from the label (y)
        5. Scale all numeric features using StandardScaler
        6. Split into training and testing sets

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame returned by load_dataset().
    target_column : str
        The name of the column to predict, e.g. "Outcome" or "target".

    Returns
    -------
    dict
        Always a dictionary — never a tuple. Keys:
            "X_train"       : numpy array — scaled training features
            "X_test"        : numpy array — scaled testing features
            "y_train"       : pandas Series — training labels
            "y_test"        : pandas Series — testing labels
            "scaler"        : fitted StandardScaler instance
            "feature_names" : list of str — column names in training order

    Raises
    ------
    TypeError
        If df is not a DataFrame or target_column is not a string.
    ValueError
        If df is empty, target_column is empty, or not found in df.
    ValueError
        If the DataFrame becomes empty after dropping missing rows.
    """

    # --- Validation 1: df must be a pandas DataFrame ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame. "
            f"Got: {type(df).__name__}"
        )

    # --- Validation 2: df must not be empty ---
    if df.empty:
        raise ValueError(
            "df is empty. "
            "Make sure load_dataset() returned valid data."
        )

    # --- Validation 3: target_column must be a string ---
    if not isinstance(target_column, str):
        raise TypeError(
            f"target_column must be a string. "
            f"Got: {type(target_column).__name__}"
        )

    # --- Validation 4: target_column must not be empty ---
    if not target_column.strip():
        raise ValueError(
            "target_column must not be an empty string. "
            "Provide the name of the label column."
        )

    # --- Validation 5: target_column must exist in the DataFrame ---
    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' not found in the DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # --- Step 1: Drop rows with missing values ---
    # ML models cannot handle NaN values.
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"[preprocess_data] Dropped {rows_dropped} rows with missing values.")

    # After dropping, check the DataFrame is still usable
    if df.empty:
        raise ValueError(
            "DataFrame is empty after dropping rows with missing values. "
            "Check your CSV file for data quality issues."
        )

    # --- Step 1b: Guard — target column must contain only 0 and 1 ---
    # All six disease datasets (including kidney and lung) must use a
    # binary 0/1 target. String targets like "ckd"/"notckd" will cause
    # train_model.py metrics (precision_score pos_label=1) to silently
    # return wrong results. Catching it here gives a clear fix message.
    unique_targets = set(df[target_column].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must contain only 0 and 1. "
            f"Found: {sorted(unique_targets)}. "
            f"Encode your target to binary integers before training "
            f"(e.g. 'ckd'->1, 'notckd'->0)."
        )

    # --- Step 2: One-hot encode categorical (text) columns ---
    # Finds all columns with text values, excluding the target column.
    # Example: a "sex" column with "Male"/"Female" becomes two 0/1 columns.
    # drop_first=True removes one column to avoid the dummy variable trap.
    cat_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col != target_column
    ]
    if cat_cols:
        print(f"[preprocess_data] Encoding categorical columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # --- Step 3: Separate features (X) from the label (y) ---
    X = df.drop(columns=[target_column])   # all columns except the label
    y = df[target_column]                  # only the label column

    # Save the feature column names in training order.
    # This is critical — predict.py must use the same order at prediction time.
    feature_names = X.columns.tolist()

    # --- Step 4: Scale all numeric features ---
    # StandardScaler transforms each column to mean=0 and std=1.
    # This is important for Logistic Regression and other distance-based models.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Step 5: Split into training and testing sets ---
    # stratify=y ensures both splits have the same class ratio (important for
    # imbalanced datasets like ours where one class is more common).
    # Fallback: if a class has too few samples for stratified splitting
    # (sklearn requires >= 2 per class in each split), retry without stratify
    # so training still proceeds — a warning is printed instead of crashing.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError:
        print(
            f"[preprocess_data] Warning: stratified split failed "
            f"(too few samples in one class). Falling back to random split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    print(
        f"[preprocess_data] Train: {len(X_train)} rows | "
        f"Test: {len(X_test)} rows | "
        f"Features: {len(feature_names)}"
    )

    # Return a dictionary — never a tuple.
    # Callers access values by name: result["X_train"], result["scaler"], etc.
    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "feature_names": feature_names,
    }


# =============================================================
# FUNCTION 3 — scale_user_input
# =============================================================

def scale_user_input(
    user_dict: dict,
    feature_names: list,
    scaler: StandardScaler,
):
    """
    Scale one row of user input so it matches what model.predict() expects.

    Called by predict.py at prediction time — NOT during training.

    Why this is needed:
        The model was trained on scaled data in a specific column order.
        When a user submits the Streamlit form, their input must be
        aligned to that same column order and scaled the same way.

    Parameters
    ----------
    user_dict     : dict           {feature_name: value} from Streamlit form
    feature_names : list           column order the model was trained on
    scaler        : StandardScaler fitted scaler from preprocess_data()

    Returns
    -------
    numpy array of shape (1, n_features) — ready for model.predict()

    Raises
    ------
    TypeError  if any argument has the wrong type
    ValueError if user_dict or feature_names is empty
    """

    # Validate user_dict
    if not isinstance(user_dict, dict):
        raise TypeError(
            f"user_dict must be a dict. Got: {type(user_dict).__name__}"
        )
    if not user_dict:
        raise ValueError("user_dict must not be empty.")

    # Validate feature_names
    if not isinstance(feature_names, list):
        raise TypeError(
            f"feature_names must be a list. Got: {type(feature_names).__name__}"
        )
    if not feature_names:
        raise ValueError("feature_names must not be empty.")

    # Validate scaler
    if not isinstance(scaler, StandardScaler):
        raise TypeError(
            f"scaler must be a fitted StandardScaler. "
            f"Got: {type(scaler).__name__}"
        )

    # Build a single-row DataFrame aligned to the training column order.
    # Any column not present in user_dict is filled with 0 automatically.
    input_df = pd.DataFrame([user_dict], columns=feature_names)
    input_df = input_df.fillna(0)

    # Apply the same scaling that was used during training
    return scaler.transform(input_df)
