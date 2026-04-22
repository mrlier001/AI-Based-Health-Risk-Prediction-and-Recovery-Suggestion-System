content = open("app.py", encoding="utf-8").read()

# ── Fix 1: width="stretch" in Model Evaluation ──────────────────
content = content.replace(
    'st.dataframe(\n            df_metrics.style.apply(_highlight, axis=1),\n            width="stretch",\n            hide_index=True,\n        )',
    'st.dataframe(\n            df_metrics.style.apply(_highlight, axis=1),\n            use_container_width=True,\n            hide_index=True,\n        )'
)

# ── Fix 2: 4-disease results loop → 3-disease explicit columns ───
old_results = '''        # ── Section 1: Prediction results (4 cards) ───────────────
        st.write("**Prediction Outcomes**")
        result_cols = st.columns(len(DISEASE_LABELS))
        for i, (disease, label) in enumerate(DISEASE_LABELS.items()):
            with result_cols[i]:
                _show_risk_card(disease, label, predictions.get(disease, {}))'''

new_results = '''        # ── Section 1: Prediction results (3 trained diseases) ─────
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

# ── Fix 3: Add Model Performance expander at bottom of Health Assessment ──
# Place it just before the closing of the Health Assessment page,
# after the download button / st.info block and before the blank lines
# that separate it from the Recommendations page.

model_perf_section = '''
        # ── Section 5: Model Performance (compact, collapsed) ────
        st.write("")
        with st.expander("📈  Model Performance Summary", expanded=False):
            st.caption(
                "Best-performing algorithm per disease, evaluated on a "
                "held-out test set (20% of data)."
            )
            _perf_rows = []
            _perf_diseases = [
                ("diabetes",     "🩸 Diabetes"),
                ("heart",        "❤️ Heart Disease"),
                ("hypertension", "💢 Hypertension"),
            ]
            for _dk, _dl in _perf_diseases:
                _mp = Path(MODEL_FILES[_dk]).resolve()
                if not _mp.exists():
                    continue
                try:
                    _bundle = joblib.load(_mp)
                except Exception:
                    continue
                _best = _bundle.get("best_model_name", "—")
                _mets = _bundle.get("all_metrics", {}).get(_best, {})
                if not _mets:
                    continue
                _perf_rows.append({
                    "Disease":   _dl,
                    "Algorithm": _best,
                    "Accuracy":  f"{_mets.get('accuracy', 0):.3f}",
                    "Precision": f"{_mets.get('precision', 0):.3f}",
                    "Recall":    f"{_mets.get('recall', 0):.3f}",
                    "F1 Score":  f"{_mets.get('f1', 0):.3f}",
                })
            if _perf_rows:
                st.dataframe(
                    pd.DataFrame(_perf_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Run training first to see model performance metrics.")

'''

# Insert just before the closing info message of the Health Assessment page
target = '''        st.write("")
        st.info(
            "💡  Use the sidebar to navigate to **Recommendations** "
            "for the full personalised lifestyle advice for each condition."
        )'''

replacement = model_perf_section + '''        st.write("")
        st.info(
            "💡  Use the sidebar to navigate to **Recommendations** "
            "for the full personalised lifestyle advice for each condition."
        )'''

content = content.replace(target, replacement)

open("app.py", "w", encoding="utf-8").write(content)
print("done")
