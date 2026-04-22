"""
app.py
------
Hybrid Intelligent Multi-Condition Health Risk Prediction
and Recovery Suggestion System

Run with:
    streamlit run app.py

Security:
    - unsafe_allow_html used ONLY for the CSS block below (no user input)
    - No user input rendered inside HTML strings
    - All display uses safe Streamlit components only
"""

from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

import joblib

from src.predict         import predict_all, get_feature_importance
from src.recommendations import get_all_recommendations, CATEGORY_LABELS
from src.database        import init_db, insert_prediction, fetch_history
from src.validators      import validate_health_inputs
from config              import MODEL_FILES, MODELS_DIR

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="🏥",
    layout="wide",
)

# ── Global CSS styling ───────────────────────────────────────────
# unsafe_allow_html is used here ONLY to inject a <style> block.
# No user input or dynamic values are included in this string.
st.markdown("""
<style>
/* ── Main background ── */
.stApp {
    background: #f4f6fb;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b4c 0%, #1a237e 60%, #1565c0 100%);
    border-right: 1px solid #0d1640;
}
[data-testid="stSidebar"] * {
    color: #e8eaf6 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    padding: 4px 0 !important;
}

/* ── Cards / bordered containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e0e4ef !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 4px 18px rgba(26, 35, 126, 0.08) !important;
}

/* ── Metric labels ── */
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #5c6bc0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1a237e !important;
}

/* ── Primary / submit buttons ── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 3px 10px rgba(26, 35, 126, 0.25) !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.88 !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: #ffffff !important;
    color: #1a237e !important;
    border: 2px solid #1a237e !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: background 0.2s ease, color 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1a237e !important;
    color: #ffffff !important;
}

/* ── Expander header ── */
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #1a237e !important;
}

/* ── Dividers ── */
hr {
    border-color: #e0e4ef !important;
    margin: 0.6rem 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── Startup database check ────────────────────────────────────────
try:
    init_db()
except Exception:
    st.error("⚠️  Database unavailable. Please refresh the page.")
    st.stop()

# ── Disease display names and icons ──────────────────────────────
DISEASE_LABELS = {
    "diabetes":     "Diabetes",
    "heart":        "Heart Disease",
    "hypertension": "Hypertension",
    "stroke":       "Stroke",
    "kidney":       "Kidney Disease",
    "lung":         "Lung Disease",
}

DISEASE_ICONS = {
    "diabetes":     "🩸",
    "heart":        "❤️",
    "hypertension": "💢",
    "stroke":       "🧠",
    "kidney":       "🫘",
    "lung":         "🫁",
}

# Category icons for recommendations display
CATEGORY_ICONS = {
    "diet":      "🥗",
    "exercise":  "🏃",
    "hydration": "💧",
    "sleep":     "😴",
    "stress":    "🧘",
    "doctor":    "🩺",
}

# Map predict.py disease keys -> recommendations.py disease keys
_REC_KEY = {
    "diabetes":     "diabetes",
    "heart":        "heart_disease",
    "hypertension": "hypertension",
    "kidney":       "kidney_disease",
    "lung":         "lung_disease",
}

# Activity level labels for history display
_ACTIVITY_LABELS = {0: "Sedentary", 1: "Light", 2: "Moderate", 3: "Active"}

# Database column name constants — used in insert and history display
_COL_DIAB_RISK    = "diabetes_risk"
_COL_DIAB_CONF    = "diabetes_confidence"
_COL_HEART_RISK   = "heart_risk"
_COL_HEART_CONF   = "heart_confidence"
_COL_HYP_RISK     = "hypertension_risk"
_COL_HYP_CONF     = "hypertension_confidence"
_COL_KIDNEY_RISK  = "kidney_risk"
_COL_KIDNEY_CONF  = "kidney_confidence"
_COL_LUNG_RISK    = "lung_risk"
_COL_LUNG_CONF    = "lung_confidence"

# Human-readable display names for raw dataset column names.
# Used only in the Feature Importance chart — does not affect any
# training, prediction, or database logic.
_FEATURE_DISPLAY_NAMES = {
    # Diabetes (Pima Indians format)
    "Pregnancies":              "Pregnancies",
    "Glucose":                  "Blood Glucose",
    "BloodPressure":            "Blood Pressure",
    "SkinThickness":            "Skin Thickness",
    "Insulin":                  "Insulin Level",
    "BMI":                      "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree",
    "Age":                      "Age",
    # Heart disease (UCI format)
    "age":                      "Age",
    "sex":                      "Sex",
    "cp":                       "Chest Pain Type",
    "trestbps":                 "Resting Blood Pressure",
    "chol":                     "Cholesterol",
    "fbs":                      "Fasting Blood Sugar",
    "restecg":                  "Resting ECG",
    "thalach":                  "Max Heart Rate",
    "exang":                    "Exercise Angina",
    "oldpeak":                  "ST Depression",
    "slope":                    "ST Slope",
    "ca":                       "Major Vessels",
    "thal":                     "Thalassemia",
    # Hypertension (Framingham format)
    "male":                     "Sex (Male)",
    "education":                "Education Level",
    "currentSmoker":            "Current Smoker",
    "cigsPerDay":               "Cigarettes Per Day",
    "BPMeds":                   "BP Medication",
    "prevalentStroke":          "Prior Stroke",
    "prevalentHyp":             "Prior Hypertension",
    "diabetes":                 "Diabetes",
    "totChol":                  "Total Cholesterol",
    "sysBP":                    "Systolic BP",
    "diaBP":                    "Diastolic BP",
    "heartRate":                "Heart Rate",
    "glucose":                  "Blood Glucose",
    # Stroke
    "hypertension":             "Hypertension",
    "heart_disease":            "Heart Disease",
    "ever_married":             "Ever Married",
    "avg_glucose_level":        "Avg Glucose Level",
    "bmi":                      "BMI",
    "smoking_status":           "Smoking Status",
    "systolic_bp":              "Systolic BP",
    # Kidney disease (UCI CKD format)
    "bp":                       "Blood Pressure",
    "sg":                       "Specific Gravity",
    "al":                       "Albumin",
    "su":                       "Sugar",
    "bgr":                      "Blood Glucose Random",
    "bu":                       "Blood Urea",
    "sc":                       "Serum Creatinine",
    "sod":                      "Sodium",
    "pot":                      "Potassium",
    "hemo":                     "Haemoglobin",
    "pcv":                      "Packed Cell Volume",
    "wc":                       "White Blood Cell Count",
    "rc":                       "Red Blood Cell Count",
    # Lung disease
    "GENDER":                   "Gender",
    "AGE":                      "Age",
    "SMOKING":                  "Smoking",
    "YELLOW_FINGERS":           "Yellow Fingers",
    "ANXIETY":                  "Anxiety",
    "PEER_PRESSURE":            "Peer Pressure",
    "CHRONIC_DISEASE":          "Chronic Disease",
    "FATIGUE":                  "Fatigue",
    "ALLERGY":                  "Allergy",
    "WHEEZING":                 "Wheezing",
    "ALCOHOL_CONSUMING":        "Alcohol Consuming",
    "COUGHING":                 "Coughing",
    "SHORTNESS_OF_BREATH":      "Shortness of Breath",
    "SWALLOWING_DIFFICULTY":    "Swallowing Difficulty",
    "CHEST_PAIN":               "Chest Pain",
}


# ── Helper: check all model files exist ──────────────────────────
def _models_ready() -> bool:
    return all(Path(p).exists() for p in MODEL_FILES.values())


# ── Helper: display one risk result ──────────────────────────────
def _show_risk_card(disease: str, label: str, res: dict):
    """Display one disease result card. No HTML, no unsafe_allow_html."""
    icon = DISEASE_ICONS.get(disease, "")
    risk = res.get("label") or res.get("risk")
    conf = res.get("confidence")

    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.write(f"**{icon}  {label}**")
            st.caption(f"Confidence: {conf}%" if conf is not None else "Confidence: N/A")
        with right:
            if res.get("error") or not risk:
                st.warning("Unavailable")
            elif risk == "High Risk":
                st.error("⚠️ High Risk")
            elif risk == "Medium Risk":
                st.warning("🟡 Medium Risk")
            elif risk == "Low Risk":
                st.success("✅ Low Risk")


# ── Sidebar navigation ───────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Health Risk")
    st.caption("Prediction System")
    st.divider()

    # Navigation menu — selection stored in session state
    page = st.radio(
        "Navigate",
        options=["Health Assessment", "Recommendations", "Prediction History", "Model Evaluation"],
        label_visibility="collapsed",
        key="sidebar_page",
    )
    st.session_state["page"] = page

    st.divider()

    # Quick stats from history — shown in sidebar at all times
    _sidebar_records = fetch_history()
    st.caption("Dashboard Stats")
    st.metric("Total Assessments", len(_sidebar_records) if _sidebar_records else 0)

    if _sidebar_records:
        _latest = _sidebar_records[0]
        st.caption(f"Last run: {_latest.get('timestamp', '—')}")

    st.divider()
    st.caption("For academic use only.")
    st.caption("Not a medical diagnosis tool.")

# ── Dashboard header ─────────────────────────────────────────────
st.title("🏥 Health Risk Prediction System")
st.caption(
    "Hybrid Intelligent Multi-Condition Screening  •  "
    "Diabetes  ·  Heart Disease  ·  Hypertension  ·  Stroke  ·  Kidney  ·  Lung  •  "
    "For academic use only"
)
st.divider()

# ── Model readiness check ─────────────────────────────────────────
if not _models_ready():
    st.error("⚠️  Models not found. Please run the training script first.")
    st.code("python generate_data.py\npython src/train_model.py", language="bash")
    st.stop()

_page = st.session_state.get("page", "Health Assessment")

# ================================================================
# PAGE: HEALTH ASSESSMENT
# ================================================================
if _page == "Health Assessment":

    # ── ROW 1: Compact metric cards ──────────────────────────────
    _preds        = st.session_state.get("predictions", {})
    _diab_risk    = _preds.get("diabetes",     {}).get("risk") or "Not yet run"
    _heart_risk   = _preds.get("heart",        {}).get("risk") or "Not yet run"
    _hyp_risk     = _preds.get("hypertension", {}).get("risk") or "Not yet run"
    _kidney_risk  = _preds.get("kidney",       {}).get("risk") or "Not yet run"
    _lung_risk    = _preds.get("lung",         {}).get("risk") or "Not yet run"
    _risk_count   = sum(
        1 for d in ["diabetes", "heart", "hypertension", "kidney", "lung"]
        if _preds.get(d, {}).get("risk") in ("High Risk", "Medium Risk")
    )
    _overall = f"{_risk_count} / 5 High" if _preds else "—"
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("🩸 Diabetes",      _diab_risk)
    mc2.metric("❤️ Heart Disease", _heart_risk)
    mc3.metric("💢 Hypertension",  _hyp_risk)
    mc4.metric("🫘 Kidney",        _kidney_risk)
    mc5.metric("🫁 Lung",          _lung_risk)
    mc6.metric("📊 Overall Score",  _overall)
    st.write("")

    # ── ROW 2: Dashboard charts (always visible) ───────────────────
    _history_records = fetch_history()
    ch_left, ch_right = st.columns(2)

    with ch_left:
        with st.container(border=True):
            st.caption("📈  Risk Level Distribution (last 20 assessments)")
            if _history_records:
                _dist = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
                for _r in _history_records[:20]:
                    for _col in [
                        _COL_DIAB_RISK, _COL_HEART_RISK, _COL_HYP_RISK,
                        _COL_KIDNEY_RISK, _COL_LUNG_RISK,
                    ]:
                        _v = _r.get(_col)
                        if _v in _dist:
                            _dist[_v] += 1
                st.bar_chart(pd.DataFrame.from_dict(
                    {"Count": _dist}, orient="columns"
                ))
            else:
                st.info("Run your first assessment to see the chart.")

    with ch_right:
        with st.container(border=True):
            st.caption("📊  Assessment Count Over Time")
            if _history_records:
                _ts = (
                    pd.DataFrame(_history_records)[["timestamp"]]
                    .assign(timestamp=lambda df: pd.to_datetime(
                        df["timestamp"], errors="coerce"
                    ))
                    .dropna()
                    .assign(date=lambda df: df["timestamp"].dt.date)
                    .groupby("date")
                    .size()
                    .reset_index(name="Assessments")
                    .set_index("date")
                )
                st.line_chart(_ts)
            else:
                st.info("Run your first assessment to see the chart.")

    st.write("")

    # ── ROW 3: Input form (inside expander to keep dashboard first) ─
    with st.expander("🔍  Enter Health Details & Run Prediction", expanded=True):
        st.caption(
            "Fill in your health details below and click **Run Prediction** "
            "to receive your personalised risk assessment."
        )
        with st.form("health_form"):

            # ── Demographics ─────────────────────────────────────────
            st.subheader("Demographics")
            c1, c2 = st.columns(2)
            age = c1.number_input(
            "Age (years)", min_value=1, max_value=120, value=35,
            help="Your current age in years"
            )
            sex = c2.selectbox(
            "Biological Sex", ["Male", "Female"],
            help="Required for accurate risk calculation"
            )

            # ── Body measurements ─────────────────────────────────────
            st.subheader("Body Measurements")
            c3, c4 = st.columns(2)
            bmi = c3.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=80.0,
            value=25.0, step=0.1,
            help="Body Mass Index — weight(kg) / height(m)²"
            )
            weight = c4.number_input(
            "Weight (kg)", min_value=20.0, max_value=300.0,
            value=70.0, step=0.5
            )

            # ── Blood and cardiovascular ──────────────────────────────
            st.subheader("Blood & Cardiovascular Measurements")
            c5, c6, c7 = st.columns(3)
            glucose = c5.number_input(
            "Glucose (mg/dL)", min_value=20, max_value=600, value=100,
            help="Fasting blood glucose level"
            )
            blood_pressure = c6.number_input(
            "Blood Pressure (mmHg)", min_value=40, max_value=300, value=120,
            help="Systolic blood pressure"
            )
            cholesterol = c7.number_input(
            "Cholesterol (mg/dL)", min_value=50, max_value=700, value=200,
            help="Total cholesterol level"
            )

            c8, c9 = st.columns(2)
            heart_rate = c8.number_input(
            "Heart Rate (bpm)", min_value=20, max_value=300, value=75,
            help="Resting heart rate"
            )
            pregnancies = c9.number_input(
            "Number of Pregnancies", min_value=0, max_value=20, value=0,
            help="Total number of pregnancies. Enter 0 if not applicable."
            )

            # ── Lifestyle ─────────────────────────────────────────────
            st.subheader("Lifestyle Factors")
            c10, c11 = st.columns(2)
            smoking_status = c10.selectbox(
            "Smoking Status", ["Non-Smoker", "Smoker"]
            )
            activity_level = c11.selectbox(
            "Physical Activity Level",
            ["Sedentary", "Light", "Moderate", "Active"],
            help="Sedentary: little or no exercise.  Active: intense daily exercise."
            )

            submitted = st.form_submit_button(
                "🔍  Run Prediction", use_container_width=True
            )

    # ── Input validation & prediction (outside expander) ─────────
    if submitted:
        raw_inputs = {
            "age":            age,
            "bmi":            bmi,
            "blood_pressure": blood_pressure,
            "glucose":        glucose,
            "cholesterol":    cholesterol,
            "heart_rate":     heart_rate,
        }
        errors, warnings = validate_health_inputs(raw_inputs)

        if errors:
            st.error("⚠️  Please correct the following before continuing:")
            for msg in errors:
                st.write(f"\u2022 {msg}")
            st.stop()

        if warnings:
            st.warning("🟡  Some values are outside typical clinical ranges. You may continue, but please verify your readings.")
            for msg in warnings:
                st.write(f"\u2022 {msg}")

        # Encode categorical fields
        sex_val      = 1 if sex == "Male" else 0
        smoking_val  = 1 if smoking_status == "Smoker" else 0
        activity_val = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3}[activity_level]

        # Build patient input dict with all dataset aliases
        patient_input = {
            "age": age, "bmi": bmi, "blood_pressure": blood_pressure,
            "glucose": glucose, "cholesterol": cholesterol,
            "heart_rate": heart_rate, "smoking_status": smoking_val,
            "activity_level": activity_val, "sex": sex_val, "weight": weight,
            # Diabetes aliases
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": 20,
            "Insulin": 80, "BMI": bmi,
            "DiabetesPedigreeFunction": 0.5, "Age": age,
            # Heart disease aliases
            "sex": sex_val, "cp": 0, "trestbps": blood_pressure,
            "chol": cholesterol, "fbs": 1 if glucose > 120 else 0,
            "restecg": 0, "thalach": heart_rate, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
            # Hypertension aliases
            "male": sex_val, "education": 2,
            "currentSmoker": smoking_val,
            "cigsPerDay": 10 if smoking_val else 0,
            "BPMeds": 0, "prevalentStroke": 0,
            "prevalentHyp": 1 if blood_pressure > 140 else 0,
            "diabetes": 1 if glucose > 126 else 0,
            "totChol": cholesterol, "sysBP": blood_pressure,
            "diaBP": int(blood_pressure * 0.65), "heartRate": heart_rate,
            # Stroke aliases
            "hypertension": 1 if blood_pressure > 140 else 0,
            "heart_disease": 0,
            "ever_married": 1 if age > 30 else 0,
            "avg_glucose_level": glucose, "systolic_bp": blood_pressure,
        }

        # Run predictions
        try:
            with st.spinner("Analysing your health data..."):
                predictions = predict_all(patient_input)
        except Exception:
            st.error("⚠️  Prediction failed. Please check your inputs and try again.")
            st.info("If this keeps happening, re-run: python src/train_model.py")
            st.stop()

        # Build flat risks for recommendations
        flat_risks = {}
        for pred_key, rec_key in _REC_KEY.items():
            res  = predictions.get(pred_key, {})
            risk = res.get("risk")
            flat_risks[rec_key] = risk if risk else "Unknown"

        try:
            recommendations = get_all_recommendations(flat_risks)
        except Exception:
            recommendations = {}
            st.warning("🟡  Recommendations could not be generated. Your prediction results are still shown below.")

        # Build summary for database
        summary_lines = []
        for rec_key, rec in recommendations.items():
            summary_lines.append(f"{rec_key.upper()} - {rec.get('risk') or 'N/A'}")
            for cat, tips in rec.get("suggestions", {}).items():
                for tip in tips:
                    summary_lines.append(f"  {cat}: {tip}")

        # Save to database — failure is logged silently, does not block results
        def _p(disease, key):
            return predictions.get(disease, {}).get(key)

        _saved = False
        try:
            row_id = insert_prediction({
                "age": age, "bmi": bmi, "blood_pressure": blood_pressure,
                "glucose": glucose, "cholesterol": cholesterol,
                "heart_rate": heart_rate, "smoking_status": smoking_val,
                "activity_level": activity_val,
                _COL_DIAB_RISK:  _p("diabetes",     "risk"),
                _COL_DIAB_CONF:  _p("diabetes",     "confidence"),
                _COL_HEART_RISK: _p("heart",        "risk"),
                _COL_HEART_CONF: _p("heart",        "confidence"),
                _COL_HYP_RISK:    _p("hypertension", "risk"),
                _COL_HYP_CONF:    _p("hypertension", "confidence"),
                _COL_KIDNEY_RISK: _p("kidney",       "risk"),
                _COL_KIDNEY_CONF: _p("kidney",       "confidence"),
                _COL_LUNG_RISK:   _p("lung",         "risk"),
                _COL_LUNG_CONF:   _p("lung",         "confidence"),
                "recommendations_summary": "\n".join(summary_lines),
            })
            _saved = isinstance(row_id, int) and row_id > 0
        except Exception:
            pass  # Do not surface database errors — results are still shown

        # Store in session state for Tab 2
        st.session_state["predictions"]     = predictions
        st.session_state["recommendations"] = recommendations

        # ── Results Summary ───────────────────────────────────────
        assessment_time = datetime.now().strftime("%d %b %Y  %H:%M")

        st.divider()

        # Header row: title on left, timestamp + save status on right
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            st.subheader("Assessment Results")
            st.caption(
                "AI-based screening tool — not a substitute for medical advice."
            )
        with hc2:
            st.write("")
            st.caption(f"📅  {assessment_time}")
            st.caption("💾  Saved to history" if _saved else "💾  Not saved")

        st.write("")

        # ── Section 1: Prediction results — row A (existing 3) ────
        st.write("**Prediction Outcomes**")
        r1, r2, r3 = st.columns(3)
        for col, (disease, label) in zip([r1, r2, r3], [
            ("diabetes",     "Diabetes"),
            ("heart",        "Heart Disease"),
            ("hypertension", "Hypertension"),
        ]):
            with col:
                _show_risk_card(disease, label, predictions.get(disease, {}))

        # ── Section 1b: Prediction results — row B (new 2) ─────────
        r4, r5, _ = st.columns(3)
        for col, (disease, label) in zip([r4, r5], [
            ("kidney", "Kidney Disease"),
            ("lung",   "Lung Disease"),
        ]):
            with col:
                _show_risk_card(disease, label, predictions.get(disease, {}))

        st.write("")

        # ── Section 2: Key recommendations (one tip per condition) ─
        if recommendations:
            st.write("**Key Recommendations**")
            rec_display = [
                ("diabetes",       "diabetes",     DISEASE_ICONS["diabetes"]),
                ("heart_disease",  "heart",        DISEASE_ICONS["heart"]),
                ("hypertension",   "hypertension", DISEASE_ICONS["hypertension"]),
                ("kidney_disease", "kidney",       DISEASE_ICONS["kidney"]),
                ("lung_disease",   "lung",         DISEASE_ICONS["lung"]),
            ]
            # Render in rows of 3 to keep layout stable
            for row_start in range(0, len(rec_display), 3):
                rec_cols = st.columns(3)
                for col, (rec_key, pred_key, icon) in zip(
                    rec_cols, rec_display[row_start:row_start + 3]
                ):
                    rec   = recommendations.get(rec_key, {})
                    label = DISEASE_LABELS.get(pred_key, rec_key.replace("_", " ").title())
                    tips  = rec.get("suggestions", {}).get("diet", [])
                    tip   = tips[0] if tips else "No recommendation available."
                    with col:
                        with st.container(border=True):
                            st.caption(f"{icon}  {label}")
                            st.write(tip)

        st.write("")

        # ── Section 3: Input summary (collapsible) ────────────────
        with st.expander("📋  View Your Input Summary"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Age", f"{age} years")
                st.metric("BMI", f"{bmi} kg/m²")
                st.metric("Blood Pressure", f"{blood_pressure} mmHg")
                st.metric("Glucose", f"{glucose} mg/dL")
            with sc2:
                st.metric("Cholesterol", f"{cholesterol} mg/dL")
                st.metric("Heart Rate", f"{heart_rate} bpm")
                st.metric("Smoking", "Yes" if smoking_val else "No")
                st.metric("Activity Level", activity_level)

        # ── Section 4: Download report ────────────────────────────
        st.write("")
        st.subheader("📄  Download Your Report")
        st.caption("Save a copy of your results and recommendations as a text file.")

        report_lines = [
            "=" * 60,
            "  HEALTH RISK PREDICTION REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "PATIENT INPUT",
            "-" * 40,
            f"  Age:            {age} years",
            f"  BMI:            {bmi} kg/m2",
            f"  Blood Pressure: {blood_pressure} mmHg",
            f"  Glucose:        {glucose} mg/dL",
            f"  Cholesterol:    {cholesterol} mg/dL",
            f"  Heart Rate:     {heart_rate} bpm",
            f"  Smoking:        {'Yes' if smoking_val else 'No'}",
            f"  Activity Level: {activity_level}",
            "",
            "PREDICTION RESULTS",
            "-" * 40,
        ]
        for disease, label in DISEASE_LABELS.items():
            res  = predictions.get(disease, {})
            risk = res.get("risk") or "Unavailable"
            conf = f"{res['confidence']}%" if res.get("confidence") else "N/A"
            report_lines.append(f"  {label:<20}: {risk:<12}  Confidence: {conf}")

        report_lines += ["", "PERSONALISED RECOMMENDATIONS", "-" * 40]
        for rec_key, rec in recommendations.items():
            label = rec_key.replace("_", " ").title()
            risk  = rec.get("risk") or "N/A"
            report_lines.append(f"\n  {label} ({risk})")
            for cat, tips in rec.get("suggestions", {}).items():
                cat_label = CATEGORY_LABELS.get(cat, cat)
                report_lines.append(f"    {cat_label}:")
                for tip in tips:
                    report_lines.append(f"      - {tip}")

        report_lines += [
            "",
            "=" * 60,
            "  DISCLAIMER",
            "  This report is generated by an AI-based screening tool.",
            "  It is not a substitute for professional medical advice.",
            "  Please consult a qualified healthcare professional.",
            "=" * 60,
        ]

        st.download_button(
            label="⬇️  Download Full Report (.txt)",
            data="\n".join(report_lines).encode("utf-8"),
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


        # ── Section 5: Model Performance (static, isolated) ─────
        st.write("")
        with st.expander("📈  Model Performance", expanded=False):
            st.caption("Best model per condition — evaluated on a held-out test set (20% of data).")
            _perf_df = pd.DataFrame([
                {"Model": "🩸 Diabetes",      "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model": "❤️ Heart Disease", "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model": "💢 Hypertension",  "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model": "🫘 Kidney",        "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model": "🫁 Lung",          "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
            ])
            # Try to load real metrics from trained bundles; fall back to "—" if not available
            _perf_diseases = [
                ("diabetes",     "🩸 Diabetes",      0),
                ("heart",        "❤️ Heart Disease", 1),
                ("hypertension", "💢 Hypertension",  2),
                ("kidney",       "🫘 Kidney",        3),
                ("lung",         "🫁 Lung",          4),
            ]
            for _dk, _dl, _idx in _perf_diseases:
                try:
                    _mp = Path(MODEL_FILES[_dk]).resolve()
                    _bundle = joblib.load(_mp)
                    _best = _bundle.get("best_model_name", "—")
                    _m = _bundle.get("all_metrics", {}).get(_best, {})
                    if _m:
                        _perf_df.at[_idx, "Algorithm"] = _best
                        _perf_df.at[_idx, "Accuracy"]  = f"{_m.get('accuracy',  0):.3f}"
                        _perf_df.at[_idx, "Precision"] = f"{_m.get('precision', 0):.3f}"
                        _perf_df.at[_idx, "Recall"]    = f"{_m.get('recall',    0):.3f}"
                        _perf_df.at[_idx, "F1 Score"]  = f"{_m.get('f1',        0):.3f}"
                except Exception:
                    pass  # Keep "—" placeholders — does not affect any other logic
            st.dataframe(_perf_df, use_container_width=True, hide_index=True)

        st.write("")
        st.info(
            "💡  Use the sidebar to navigate to **Recommendations** "
            "for the full personalised lifestyle advice for each condition."
        )


# ================================================================
# PAGE: RECOMMENDATIONS
# ================================================================
elif _page == "Recommendations":
    st.subheader("💡  Personalised Lifestyle Recommendations")

    if "recommendations" not in st.session_state:
        st.info(
            "No assessment completed yet. "
            "Use the sidebar to go to **Health Assessment** and click **Run Prediction** first."
        )
    else:
        recommendations = st.session_state["recommendations"]
        predictions     = st.session_state["predictions"]

        st.caption(
            "Personalised suggestions across six lifestyle categories "
            "based on your risk assessment. High Risk conditions are expanded automatically."
        )
        st.write("")

        # Display one expander per disease
        for rec_key, pred_key in [
            ("diabetes",       "diabetes"),
            ("heart_disease",  "heart"),
            ("hypertension",   "hypertension"),
            ("kidney_disease", "kidney"),
            ("lung_disease",   "lung"),
        ]:
            rec   = recommendations.get(rec_key, {})
            risk  = rec.get("risk") or "N/A"
            label = DISEASE_LABELS.get(pred_key, rec_key.replace("_", " ").title())
            icon  = DISEASE_ICONS.get(pred_key, "")
            badge = "⚠️ High Risk" if risk == "High Risk" else ("🟡 Medium Risk" if risk == "Medium Risk" else "✅ Low Risk")

            with st.expander(
                f"{icon}  {label}  —  {badge}",
                expanded=(risk == "High Risk"),
            ):
                if rec.get("error"):
                    st.warning(
                        "Recommendations are not available for this condition. "
                        "Please re-run the assessment."
                    )
                    continue

                suggestions = rec.get("suggestions", {})
                if not suggestions:
                    st.write("No suggestions available.")
                    continue

                # Two-column grid layout
                cats = list(suggestions.items())
                for i in range(0, len(cats), 2):
                    row_cols = st.columns(2)
                    for j, (cat, tips) in enumerate(cats[i:i + 2]):
                        with row_cols[j]:
                            cat_icon  = CATEGORY_ICONS.get(cat, "")
                            cat_label = CATEGORY_LABELS.get(cat, cat.title())
                            st.write(f"**{cat_icon}  {cat_label}**")
                            for tip in tips:
                                st.write(f"- {tip}")
                    st.write("")


# ================================================================
# PAGE: PREDICTION HISTORY
# ================================================================
elif _page == "Prediction History":
    st.subheader("🕓  Prediction History")

    records = fetch_history()

    if records is None or not isinstance(records, list):
        st.warning(
            "Your assessment history could not be loaded at this time. "
            "Please refresh the page."
        )
    elif not records:
        st.info(
            "No assessments saved yet. "
            "Complete a Health Assessment to see your history here."
        )
    else:
        st.caption(f"{len(records)} assessment(s) on record — most recent first.")
        st.write("")

        # ── Summary table with human-readable column names ────────
        rows = []
        for r in records:
            rows.append({
                "Date & Time":   r.get("timestamp", ""),
                "Diabetes":      r.get(_COL_DIAB_RISK)    or "—",
                "Diab. Conf.":   f"{r[_COL_DIAB_CONF]}%"    if r.get(_COL_DIAB_CONF)    is not None else "—",
                "Heart Disease": r.get(_COL_HEART_RISK)   or "—",
                "Heart Conf.":   f"{r[_COL_HEART_CONF]}%"   if r.get(_COL_HEART_CONF)   is not None else "—",
                "Hypertension":  r.get(_COL_HYP_RISK)     or "—",
                "Hypert. Conf.": f"{r[_COL_HYP_CONF]}%"     if r.get(_COL_HYP_CONF)     is not None else "—",
                "Kidney":        r.get(_COL_KIDNEY_RISK)  or "—",
                "Kidney Conf.":  f"{r[_COL_KIDNEY_CONF]}%"  if r.get(_COL_KIDNEY_CONF)  is not None else "—",
                "Lung":          r.get(_COL_LUNG_RISK)    or "—",
                "Lung Conf.":    f"{r[_COL_LUNG_CONF]}%"    if r.get(_COL_LUNG_CONF)    is not None else "—",
            })

        history_df = pd.DataFrame(rows)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # ── Download button ───────────────────────────────────────
        st.download_button(
            label="⬇️  Download History as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        # ── Detailed record viewer ────────────────────────────────
        st.divider()
        st.subheader("Full Record Details")

        selected_idx = st.selectbox(
            "Select a record to view:",
            options=range(len(records)),
            format_func=lambda i: (
                f"Record {records[i].get('id')}  —  "
                f"{records[i].get('timestamp')}"
            ),
        )

        sel = records[selected_idx]
        st.write("")

        # Patient input metrics
        st.write("**Patient Inputs**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Age",            f"{sel.get('age')} yrs")
        mc2.metric("BMI",            f"{sel.get('bmi')} kg/m²")
        mc3.metric("Blood Pressure", f"{sel.get('blood_pressure')} mmHg")
        mc4.metric("Glucose",        f"{sel.get('glucose')} mg/dL")

        mc5, mc6, mc7, mc8 = st.columns(4)
        mc5.metric("Cholesterol",    f"{sel.get('cholesterol')} mg/dL")
        mc6.metric("Heart Rate",     f"{sel.get('heart_rate')} bpm")
        mc7.metric("Smoking",        "Yes" if sel.get("smoking_status") else "No")
        mc8.metric("Activity",       _ACTIVITY_LABELS.get(sel.get("activity_level"), "—"))

        # Prediction results
        st.write("")
        st.write("**Prediction Outcomes**")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(
            "🩸 Diabetes",
            sel.get(_COL_DIAB_RISK) or "—",
            delta=f"Confidence: {sel.get(_COL_DIAB_CONF)}%" if sel.get(_COL_DIAB_CONF) else None,
            delta_color="off",
        )
        pc2.metric(
            "❤️ Heart Disease",
            sel.get(_COL_HEART_RISK) or "—",
            delta=f"Confidence: {sel.get(_COL_HEART_CONF)}%" if sel.get(_COL_HEART_CONF) else None,
            delta_color="off",
        )
        pc3.metric(
            "💢 Hypertension",
            sel.get(_COL_HYP_RISK) or "—",
            delta=f"Confidence: {sel.get(_COL_HYP_CONF)}%" if sel.get(_COL_HYP_CONF) else None,
            delta_color="off",
        )
        pc4, pc5, _ = st.columns(3)
        pc4.metric(
            "🫘 Kidney Disease",
            sel.get(_COL_KIDNEY_RISK) or "—",
            delta=f"Confidence: {sel.get(_COL_KIDNEY_CONF)}%" if sel.get(_COL_KIDNEY_CONF) else None,
            delta_color="off",
        )
        pc5.metric(
            "🫁 Lung Disease",
            sel.get(_COL_LUNG_RISK) or "—",
            delta=f"Confidence: {sel.get(_COL_LUNG_CONF)}%" if sel.get(_COL_LUNG_CONF) else None,
            delta_color="off",
        )

        with st.expander("📋  Recommendations Summary"):
            summary = sel.get("recommendations_summary") or "No summary saved for this record."
            st.text(summary)


# ================================================================
# PAGE: MODEL EVALUATION
# ================================================================
elif _page == "Model Evaluation":
    st.subheader("📊  Model Evaluation")
    st.caption(
        "Each disease model was trained using three algorithms and evaluated "
        "on a held-out test set (20% of data). "
        "The best model by F1 Score was selected for predictions."
    )

    # ── Metric explanations ───────────────────────────────────────
    with st.expander("ℹ️  What do these metrics mean?", expanded=False):
        mc1, mc2 = st.columns(2)
        mc1.write("**Accuracy** — percentage of all predictions that were correct.")
        mc1.write("**Precision** — of all High Risk predictions, how many were correct. High precision means fewer false alarms.")
        mc2.write("**Recall** — of all actual High Risk cases, how many were correctly identified. High recall means fewer missed cases.")
        mc2.write("**F1 Score** — balance between Precision and Recall. Used to select the best model. 1.0 is perfect.")

    st.write("")

    # ── Load and display metrics for each disease ─────────────────
    eval_diseases = [
        ("diabetes",     "🩸  Diabetes"),
        ("heart",        "❤️  Heart Disease"),
        ("hypertension", "💢  Hypertension"),
        ("kidney",       "🫘  Kidney Disease"),
        ("lung",         "🫁  Lung Disease"),
    ]

    for disease_key, disease_label in eval_diseases:
        model_path = Path(MODEL_FILES[disease_key]).resolve()
        if not str(model_path).startswith(str(MODELS_DIR.resolve())):
            st.warning(f"{disease_label}: This model could not be loaded. Please re-run training.")
            continue
        if not model_path.exists():
            st.warning(f"{disease_label}: Model file not found. Please run the training script and refresh this page.")
            continue

        try:
            bundle = joblib.load(model_path)
        except Exception:
            st.warning(f"{disease_label}: Model file could not be read. Please re-run the training script.")
            continue

        all_metrics = bundle.get("all_metrics", {})
        best_name   = bundle.get("best_model_name", "")

        if not all_metrics:
            st.info(f"{disease_label}: no evaluation data found. Please re-run training to see results.")
            continue

        st.write(f"#### {disease_label}")
        st.caption(f"Best performing model: **{best_name}**")

        # Build metrics table
        rows = []
        for model_name, m in all_metrics.items():
            rows.append({
                "Algorithm": model_name,
                "Accuracy":  f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall":    f"{m['recall']:.4f}",
                "F1 Score":  f"{m['f1']:.4f}",
                "Selected":  "✔" if model_name == best_name else "",
            })

        df_metrics = pd.DataFrame(rows)

        # Highlight the selected model row
        def _highlight(row):
            if row["Selected"] == "✔":
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_metrics.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        best_m = all_metrics[best_name]
        st.success(
            f"Selected: **{best_name}** — "
            f"F1 {best_m['f1']:.4f}  |  "
            f"Accuracy {best_m['accuracy']:.4f}  |  "
            f"Precision {best_m['precision']:.4f}  |  "
            f"Recall {best_m['recall']:.4f}"
        )

        # ── Feature importance ────────────────────────────────────
        importance = get_feature_importance(disease_key)
        if importance:
            with st.expander("🔍  Feature Importance — what drives this prediction?"):
                st.caption(
                    "Shows how much each input feature influenced the model's "
                    "decisions during training. Higher score = stronger influence "
                    "on the predicted outcome."
                )

                top = importance[:8]

                # Apply human-readable display names
                top_display = [
                    (_FEATURE_DISPLAY_NAMES.get(name, name), score)
                    for name, score in top
                ]

                # Two-column layout: ranked list on left, bar chart on right
                col_list, col_chart = st.columns([1, 2])

                with col_list:
                    st.write("**Ranked List**")
                    for rank, (fname, score) in enumerate(top_display, start=1):
                        pct = round(score * 100, 1)
                        st.write(f"{rank}. {fname} — {pct}%")

                with col_chart:
                    st.write("**Importance Chart**")
                    fi_df = pd.DataFrame(
                        top_display, columns=["Feature", "Importance"]
                    ).set_index("Feature")
                    st.bar_chart(fi_df, height=260)

        else:
            st.caption(
                "Feature importance is not available for this model type. "
                "Only Decision Tree and Random Forest support it."
            )

        st.write("")
