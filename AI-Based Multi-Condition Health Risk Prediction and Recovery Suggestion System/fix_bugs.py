content = open("app.py", encoding="utf-8").read()

# 1. Remove duplicate "import joblib" inside Model Evaluation page
content = content.replace(
    "\n    # \u2500\u2500 Load and display metrics for each disease \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    import joblib\n\n    eval_diseases",
    "\n    # \u2500\u2500 Load and display metrics for each disease \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    eval_diseases"
)

# 2. Fix width="stretch" -> use_container_width=True
content = content.replace(
    'st.dataframe(history_df, width="stretch", hide_index=True)',
    'st.dataframe(history_df, use_container_width=True, hide_index=True)'
)

# 3. Fix "elif not importance:" -> "else:"
content = content.replace(
    "        elif not importance:\n            st.caption(\n                \"Feature importance is not available for this model type. \"\n                \"Only Decision Tree and Random Forest support it.\"\n            )",
    "        else:\n            st.caption(\n                \"Feature importance is not available for this model type. \"\n                \"Only Decision Tree and Random Forest support it.\"\n            )"
)

# 4. Fix results section: replace 4-disease loop with 3-disease explicit columns
old_results = '''        # \u2500\u2500 Section 1: Prediction results (4 cards) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        st.write("**Prediction Outcomes**")
        result_cols = st.columns(len(DISEASE_LABELS))
        for i, (disease, label) in enumerate(DISEASE_LABELS.items()):
            with result_cols[i]:
                _show_risk_card(disease, label, predictions.get(disease, {}))'''

new_results = '''        # \u2500\u2500 Section 1: Prediction results (3 trained diseases) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        st.write("**Prediction Outcomes**")
        _result_diseases = [
            ("diabetes",     "Diabetes"),
            ("heart",        "Heart Disease"),
            ("hypertension", "Hypertension"),
        ]
        r1, r2, r3 = st.columns(3)
        for col, (disease, label) in zip([r1, r2, r3], _result_diseases):
            with col:
                _show_risk_card(disease, label, predictions.get(disease, {}))'''

content = content.replace(old_results, new_results)

open("app.py", "w", encoding="utf-8").write(content)
print("done")
