"""
recommendations.py
------------------
Phase 6 — Recommendation Engine

Generates structured, condition-specific lifestyle suggestions based
on the risk level predicted by predict.py.

How it connects to predict.py
------------------------------
predict.py returns a flat dictionary like this:

    {
        "diabetes":        "High Risk",
        "heart_disease":   "Low Risk",
        "hypertension":    "Medium Risk",
        "kidney_disease":  "High Risk",
        "lung_disease":    "Low Risk",
    }

This module accepts that dictionary and returns structured suggestions
for each condition and risk level.

Public functions
----------------
get_recommendations(disease, risk)
    Returns suggestions for one disease and one risk level.
    Output: dict with keys — disease, risk, suggestions, error

get_all_recommendations(predictions)
    Accepts the flat predictions dict from predict.py.
    Returns suggestions for all five diseases at once.
    Output: dict keyed by disease name

Suggestion categories (4 per condition)
-----------------------------------------
    diet        — what to eat and what to avoid
    exercise    — physical activity guidance
    monitoring  — what to track and how often
    doctor      — when and why to consult a doctor

Supported diseases
------------------
    diabetes       (key used in predictions dict)
    heart_disease  (key used in predictions dict)
    hypertension   (key used in predictions dict)
    kidney_disease (key used in predictions dict)
    lung_disease   (key used in predictions dict)

Supported risk levels
---------------------
    "High Risk"    — model predicted positive class with high confidence
    "Medium Risk"  — model predicted positive class with low confidence
    "Low Risk"     — model predicted negative class

Note on Medium Risk
-------------------
Medium Risk uses the same suggestions as High Risk but with softer
monitoring and doctor advice — the patient should be aware and
monitored, but the model is not highly confident in the prediction.
"""


# =============================================================
# SUGGESTION BANK
# Structure: disease_key -> risk_level -> category -> list of tips
# =============================================================

_SUGGESTIONS = {

    # ----------------------------------------------------------
    # DIABETES
    # ----------------------------------------------------------
    "diabetes": {

        "High Risk": {
            "diet": [
                "Avoid sugary drinks, white bread, and processed snacks.",
                "Choose fibre-rich foods: vegetables, legumes, and whole grains.",
                "Control portion sizes and eat meals at regular times.",
                "Replace fruit juices with whole fruits to reduce sugar spikes.",
                "Limit refined carbohydrates such as white rice and pastries.",
            ],
            "exercise": [
                "Walk briskly for at least 30 minutes on most days of the week.",
                "Try low-impact activities such as cycling or swimming.",
                "Break up long periods of sitting with a short walk every hour.",
                "Aim for at least 150 minutes of moderate activity per week.",
            ],
            "monitoring": [
                "Track your fasting blood glucose level regularly.",
                "Keep a food diary to identify meals that raise your blood sugar.",
                "Monitor your weight weekly — even small reductions help.",
                "Note any symptoms such as unusual thirst, fatigue, or blurred vision.",
            ],
            "doctor": [
                "Book an appointment for an HbA1c blood glucose test.",
                "Ask your doctor about a personalised diabetes management plan.",
                "Schedule follow-up check-ups every 3 to 6 months.",
                "Discuss whether medication or dietary support is appropriate.",
            ],
        },

        "Medium Risk": {
            "diet": [
                "Reduce added sugars and refined carbohydrates in your daily meals.",
                "Increase vegetables, legumes, and whole grains.",
                "Avoid skipping meals — irregular eating can affect blood sugar.",
                "Choose water or unsweetened drinks over sugary beverages.",
            ],
            "exercise": [
                "Aim for at least 150 minutes of moderate activity per week.",
                "Walking, cycling, or swimming are all suitable options.",
                "Avoid long periods of inactivity during the day.",
            ],
            "monitoring": [
                "Consider checking your fasting blood glucose once a month.",
                "Pay attention to energy levels and unusual thirst.",
                "Track your weight and waist measurement monthly.",
            ],
            "doctor": [
                "Discuss your result with a doctor at your next routine visit.",
                "Ask about a fasting blood glucose or HbA1c screening test.",
                "No urgent action is needed, but monitoring is recommended.",
            ],
        },

        "Low Risk": {
            "diet": [
                "Maintain a balanced diet with plenty of vegetables and whole grains.",
                "Limit added sugars and highly processed foods.",
            ],
            "exercise": [
                "Keep up at least 150 minutes of physical activity per week.",
                "Any form of regular movement helps maintain healthy blood sugar.",
            ],
            "monitoring": [
                "Get a routine fasting blood glucose check once a year.",
                "Maintain a healthy weight to reduce future risk.",
            ],
            "doctor": [
                "Mention your family history of diabetes at your next check-up.",
                "Annual blood sugar screening is sufficient at this risk level.",
            ],
        },
    },

    # ----------------------------------------------------------
    # HEART DISEASE
    # ----------------------------------------------------------
    "heart_disease": {

        "High Risk": {
            "diet": [
                "Reduce saturated fats — limit red meat, butter, and full-fat dairy.",
                "Eat omega-3 rich foods: oily fish, walnuts, and flaxseeds.",
                "Increase fruits, vegetables, and whole grains daily.",
                "Avoid trans fats found in fried and packaged foods.",
                "Reduce salt intake to help manage blood pressure.",
            ],
            "exercise": [
                "Start with light cardio such as walking or gentle swimming.",
                "Exercise for 20 to 30 minutes daily — consult your doctor first.",
                "Avoid sudden strenuous activity, especially if you feel chest discomfort.",
                "Gradually increase intensity only with medical guidance.",
            ],
            "monitoring": [
                "Check your blood pressure and resting heart rate regularly.",
                "Monitor your cholesterol levels through periodic blood tests.",
                "Keep a record of any chest discomfort, breathlessness, or palpitations.",
                "Track your weight — excess weight increases cardiac workload.",
            ],
            "doctor": [
                "See a doctor or cardiologist for a full cardiac evaluation.",
                "Do not ignore chest pain, shortness of breath, or irregular heartbeat.",
                "Ask about an ECG or stress test if recommended.",
                "Discuss cholesterol-lowering or blood pressure medication if needed.",
            ],
        },

        "Medium Risk": {
            "diet": [
                "Reduce saturated fats and increase heart-healthy foods.",
                "Include oats, berries, and leafy greens in your meals.",
                "Limit salt and avoid processed or packaged foods.",
                "Choose grilled or baked foods over fried options.",
            ],
            "exercise": [
                "Aim for 150 minutes of moderate aerobic activity per week.",
                "Walking, cycling, or light swimming are good starting points.",
                "Avoid very intense exercise until you have spoken to a doctor.",
            ],
            "monitoring": [
                "Check your blood pressure at least once a month.",
                "Have your cholesterol tested at your next routine appointment.",
                "Note any unusual fatigue, breathlessness, or chest tightness.",
            ],
            "doctor": [
                "Mention this result at your next routine medical appointment.",
                "Ask for a blood pressure and cholesterol check.",
                "No urgent action is needed, but regular monitoring is advised.",
            ],
        },

        "Low Risk": {
            "diet": [
                "Continue a balanced diet low in saturated fats and salt.",
                "Include oats, berries, and leafy greens regularly.",
            ],
            "exercise": [
                "Maintain at least 150 minutes of aerobic activity per week.",
                "Regular exercise is one of the best protections against heart disease.",
            ],
            "monitoring": [
                "Have blood pressure and cholesterol checked annually.",
                "Maintain a healthy weight and avoid smoking.",
            ],
            "doctor": [
                "Annual cardiovascular screening is sufficient at this risk level.",
                "Inform your doctor of any family history of heart disease.",
            ],
        },
    },

    # ----------------------------------------------------------
    # HYPERTENSION
    # ----------------------------------------------------------
    "hypertension": {

        "High Risk": {
            "diet": [
                "Follow a low-sodium diet — aim for less than 1,500 mg of salt per day.",
                "Eat potassium-rich foods: bananas, spinach, and sweet potatoes.",
                "Avoid processed meats, canned soups, and fast food.",
                "Reduce alcohol consumption — it raises blood pressure.",
                "Increase fruits, vegetables, and low-fat dairy in your meals.",
            ],
            "exercise": [
                "Engage in regular aerobic exercise such as brisk walking.",
                "Aim for 30 minutes of moderate activity on most days.",
                "Avoid heavy weightlifting without medical clearance.",
                "Consistency matters more than intensity — daily movement helps.",
            ],
            "monitoring": [
                "Measure your blood pressure at home daily if possible.",
                "Record readings in a log to share with your doctor.",
                "Note any headaches, dizziness, or visual disturbances.",
                "Track your salt intake and weight weekly.",
            ],
            "doctor": [
                "See a doctor to have your blood pressure measured accurately.",
                "Do not self-medicate — prescribed medication may be needed.",
                "Ask about a 24-hour blood pressure monitoring test.",
                "Discuss lifestyle changes and whether medication is appropriate.",
            ],
        },

        "Medium Risk": {
            "diet": [
                "Reduce salt in cooking and avoid adding salt at the table.",
                "Eat more fruits, vegetables, and whole grains.",
                "Limit alcohol and caffeine intake.",
                "Avoid processed and packaged foods where possible.",
            ],
            "exercise": [
                "Aim for at least 30 minutes of moderate exercise most days.",
                "Walking, cycling, or yoga are all suitable options.",
                "Avoid very intense exercise until blood pressure is assessed.",
            ],
            "monitoring": [
                "Check your blood pressure at a pharmacy or clinic monthly.",
                "Keep a note of any headaches or dizziness.",
                "Monitor your weight and salt intake.",
            ],
            "doctor": [
                "Mention this result at your next routine appointment.",
                "Ask for a blood pressure check and discuss your lifestyle.",
                "No urgent action is needed, but monitoring is recommended.",
            ],
        },

        "Low Risk": {
            "diet": [
                "Maintain a low-sodium diet as a preventive measure.",
                "Eat plenty of fruits, vegetables, and low-fat dairy.",
            ],
            "exercise": [
                "Stay active with at least 30 minutes of exercise most days.",
                "Regular physical activity helps keep blood pressure in a healthy range.",
            ],
            "monitoring": [
                "Check blood pressure at least once a year.",
                "Maintain a healthy weight and limit salt intake.",
            ],
            "doctor": [
                "Annual blood pressure screening is sufficient at this risk level.",
                "Inform your doctor if you have a family history of hypertension.",
            ],
        },
    },

    # ----------------------------------------------------------
    # KIDNEY DISEASE
    # ----------------------------------------------------------
    "kidney_disease": {

        "High Risk": {
            "diet": [
                "Strictly reduce salt intake — aim for less than 1,500 mg of sodium per day.",
                "Limit high-potassium foods such as bananas, oranges, and potatoes.",
                "Avoid processed and packaged foods — they are high in sodium and phosphorus.",
                "Reduce protein intake from red meat — choose fish or plant-based proteins.",
                "Limit phosphorus-rich foods such as dairy, nuts, and cola drinks.",
            ],
            "exercise": [
                "Engage in light aerobic activity such as walking for 20 to 30 minutes daily.",
                "Avoid strenuous exercise that puts strain on the kidneys.",
                "Gentle yoga or stretching can help manage blood pressure and stress.",
                "Always consult your doctor before starting a new exercise routine.",
            ],
            "monitoring": [
                "Monitor your blood pressure daily — hypertension accelerates kidney damage.",
                "Track your fluid intake and urine output if advised by your doctor.",
                "Watch for swelling in the legs, ankles, or face — a sign of fluid retention.",
                "Note any changes in urine colour, frequency, or foaming.",
            ],
            "doctor": [
                "See a nephrologist (kidney specialist) as soon as possible.",
                "Ask for a full kidney function panel: creatinine, eGFR, and urine albumin.",
                "Discuss whether dialysis or other interventions are needed.",
                "Consult your doctor regularly — at least every 3 months.",
            ],
        },

        "Medium Risk": {
            "diet": [
                "Reduce salt intake — avoid adding salt at the table or in cooking.",
                "Stay well hydrated — drink 6 to 8 glasses of water daily unless advised otherwise.",
                "Avoid processed food, fast food, and canned goods high in sodium.",
                "Moderate your protein intake — avoid very high-protein diets.",
            ],
            "exercise": [
                "Aim for 30 minutes of moderate activity such as walking most days.",
                "Avoid dehydration during exercise — drink water before and after.",
                "Low-impact activities like swimming or cycling are suitable.",
            ],
            "monitoring": [
                "Have your kidney function (creatinine and eGFR) tested every 6 months.",
                "Monitor blood pressure monthly and keep a record.",
                "Watch for unusual fatigue, swelling, or changes in urination.",
            ],
            "doctor": [
                "Discuss this result with your doctor at your next routine visit.",
                "Ask for a urine albumin test to check for early kidney damage.",
                "No urgent action is needed, but regular monitoring is recommended.",
            ],
        },

        "Low Risk": {
            "diet": [
                "Maintain a balanced diet low in salt and processed foods.",
                "Stay hydrated — drink adequate water throughout the day.",
            ],
            "exercise": [
                "Keep up regular physical activity to maintain healthy blood pressure.",
                "At least 150 minutes of moderate exercise per week is recommended.",
            ],
            "monitoring": [
                "Have kidney function checked as part of your annual health screening.",
                "Maintain a healthy weight and avoid long-term use of painkillers (NSAIDs).",
            ],
            "doctor": [
                "Annual kidney function screening is sufficient at this risk level.",
                "Inform your doctor if you have diabetes or hypertension — both affect kidneys.",
            ],
        },
    },

    # ----------------------------------------------------------
    # LUNG DISEASE
    # ----------------------------------------------------------
    "lung_disease": {

        "High Risk": {
            "diet": [
                "Eat antioxidant-rich foods: berries, leafy greens, and citrus fruits.",
                "Include omega-3 fatty acids from oily fish, walnuts, and flaxseeds.",
                "Avoid foods that trigger mucus production such as dairy and processed foods.",
                "Stay well hydrated — water helps thin mucus and keep airways clear.",
                "Maintain a healthy weight — excess weight puts pressure on the lungs.",
            ],
            "exercise": [
                "Practice diaphragmatic (belly) breathing exercises daily.",
                "Try pursed-lip breathing to slow your breathing and improve oxygen flow.",
                "Light walking or gentle yoga can improve lung capacity over time.",
                "Avoid outdoor exercise on days with high pollution or pollen counts.",
            ],
            "monitoring": [
                "Track any worsening of coughing, wheezing, or shortness of breath.",
                "Note how far you can walk before becoming breathless — record changes.",
                "Monitor your oxygen saturation with a pulse oximeter if available.",
                "Keep a symptom diary to share with your doctor at appointments.",
            ],
            "doctor": [
                "Seek medical attention promptly — do not ignore persistent symptoms.",
                "Ask for a spirometry test to measure your lung function.",
                "Discuss whether an inhaler, medication, or pulmonary rehabilitation is needed.",
                "If you smoke, ask your doctor about a cessation programme immediately.",
            ],
        },

        "Medium Risk": {
            "diet": [
                "Increase fruits and vegetables rich in vitamins C and E.",
                "Stay hydrated to keep airways moist and reduce irritation.",
                "Avoid heavily processed or fried foods that may worsen inflammation.",
                "Limit dairy if you notice it increases mucus production.",
            ],
            "exercise": [
                "Practice breathing exercises such as diaphragmatic breathing daily.",
                "Gentle walks in clean-air environments help maintain lung capacity.",
                "Avoid polluted or smoky environments during outdoor activity.",
            ],
            "monitoring": [
                "Note any new or worsening cough, wheeze, or breathlessness.",
                "Monitor how your breathing feels during normal daily activities.",
                "Track any chest tightness or pain and report it to your doctor.",
            ],
            "doctor": [
                "Mention this result at your next routine medical appointment.",
                "Ask about a lung function test if you have not had one recently.",
                "No urgent action is needed, but monitoring is recommended.",
            ],
        },

        "Low Risk": {
            "diet": [
                "Maintain a diet rich in fruits, vegetables, and whole grains.",
                "Stay well hydrated to support healthy lung function.",
            ],
            "exercise": [
                "Keep up regular aerobic activity to maintain lung capacity.",
                "Avoid smoking and second-hand smoke in all environments.",
            ],
            "monitoring": [
                "Be aware of any new respiratory symptoms such as persistent cough.",
                "Avoid prolonged exposure to dust, fumes, or air pollution.",
            ],
            "doctor": [
                "Annual respiratory check is sufficient at this risk level.",
                "Inform your doctor if you develop a persistent cough or breathlessness.",
            ],
        },
    },
}

# Valid risk levels — must match the output of predict.py exactly
_VALID_RISK_LEVELS = ("High Risk", "Medium Risk", "Low Risk")

# Supported disease keys — must match the keys used in app.py _REC_KEY
_SUPPORTED_DISEASES = (
    "diabetes",
    "heart_disease",
    "hypertension",
    "kidney_disease",
    "lung_disease",
)

# Human-readable category labels used by app.py for display
CATEGORY_LABELS = {
    "diet":       "Diet",
    "exercise":   "Exercise",
    "monitoring": "Monitoring",
    "doctor":     "Doctor Advice",
}


# =============================================================
# FUNCTION 1 — get_recommendations
# =============================================================

def get_recommendations(disease: str, risk: str) -> dict:
    """
    Return structured suggestions for one disease and risk level.

    Parameters
    ----------
    disease : str
        One of: 'diabetes', 'heart_disease', 'hypertension'

    risk : str
        One of: 'High Risk', 'Medium Risk', 'Low Risk'

    Returns
    -------
    dict with four keys — always returned even if inputs are invalid:
        "disease"     : str        the disease key passed in
        "risk"        : str        the risk level passed in
        "suggestions" : dict       category -> list of tip strings
                                   empty dict {} if inputs are invalid
        "error"       : str|None   None on success, message on failure

    Example (success):
        {
            "disease": "diabetes",
            "risk": "High Risk",
            "suggestions": {
                "diet":       ["Avoid sugary drinks...", ...],
                "exercise":   ["Walk briskly for 30 minutes...", ...],
                "monitoring": ["Track your fasting blood glucose...", ...],
                "doctor":     ["Book an HbA1c test...", ...],
            },
            "error": None,
        }

    Example (invalid disease):
        {
            "disease": "cancer",
            "risk": "High Risk",
            "suggestions": {},
            "error": "No suggestions for disease 'cancer'.",
        }
    """
    result = {
        "disease":     disease,
        "risk":        risk,
        "suggestions": {},
        "error":       None,
    }

    if not isinstance(disease, str) or not disease.strip():
        result["error"] = (
            f"disease must be a non-empty string. "
            f"Supported: {list(_SUPPORTED_DISEASES)}"
        )
        return result

    if disease not in _SUGGESTIONS:
        result["error"] = (
            f"No suggestions for disease '{disease}'. "
            f"Supported: {list(_SUPPORTED_DISEASES)}"
        )
        return result

    if not isinstance(risk, str) or not risk.strip():
        result["error"] = (
            f"risk must be a non-empty string. "
            f"Supported: {list(_VALID_RISK_LEVELS)}"
        )
        return result

    if risk not in _VALID_RISK_LEVELS:
        result["error"] = (
            f"Unknown risk level '{risk}'. "
            f"Supported: {list(_VALID_RISK_LEVELS)}"
        )
        return result

    result["suggestions"] = _SUGGESTIONS[disease][risk]
    return result


# =============================================================
# FUNCTION 2 — get_all_recommendations
# =============================================================

def get_all_recommendations(predictions: dict) -> dict:
    """
    Generate suggestions for all five diseases from the predictions dict.

    Parameters
    ----------
    predictions : dict
        Flat dictionary mapping disease key -> risk label. Expected shape:
        {
            "diabetes":       "High Risk" | "Medium Risk" | "Low Risk",
            "heart_disease":  "High Risk" | "Medium Risk" | "Low Risk",
            "hypertension":   "High Risk" | "Medium Risk" | "Low Risk",
            "kidney_disease": "High Risk" | "Medium Risk" | "Low Risk",
            "lung_disease":   "High Risk" | "Medium Risk" | "Low Risk",
        }

    Returns
    -------
    dict  {disease_key: get_recommendations() result dict}

        Example:
        {
            "diabetes":       { "disease": "diabetes",      "risk": "High Risk", "suggestions": {...}, "error": None },
            "heart_disease":  { "disease": "heart_disease", "risk": "Low Risk",  "suggestions": {...}, "error": None },
            "hypertension":   { ... },
            "kidney_disease": { ... },
            "lung_disease":   { ... },
        }

    Notes
    -----
    - Non-dict input returns error entries for all five diseases.
    - Missing disease key returns an error entry for that disease only.
    - Values starting with "Error:" (from predict.py failures) are
      passed through as error entries.
    - All other diseases are still processed normally.
    """
    if not isinstance(predictions, dict):
        return {
            d: {
                "disease":     d,
                "risk":        None,
                "suggestions": {},
                "error": (
                    f"predictions must be a dict. "
                    f"Got: {type(predictions).__name__}"
                ),
            }
            for d in _SUPPORTED_DISEASES
        }

    output = {}

    for disease in _SUPPORTED_DISEASES:
        risk = predictions.get(disease)

        if risk is None:
            output[disease] = {
                "disease":     disease,
                "risk":        None,
                "suggestions": {},
                "error":       f"No prediction found for '{disease}'.",
            }
            continue

        if isinstance(risk, str) and risk.startswith("Error:"):
            output[disease] = {
                "disease":     disease,
                "risk":        None,
                "suggestions": {},
                "error":       risk,
            }
            continue

        output[disease] = get_recommendations(disease, risk)

    return output
