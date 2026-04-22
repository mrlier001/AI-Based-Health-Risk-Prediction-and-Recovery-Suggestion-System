content = open("app.py", encoding="utf-8").read()

old = '''        # ── Section 5: Model Performance (compact, collapsed) ────
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

new = '''        # ── Section 5: Model Performance (static, isolated) ─────
        st.write("")
        with st.expander("📈  Model Performance", expanded=False):
            st.caption("Best model per condition — evaluated on a held-out test set (20% of data).")
            _perf_df = pd.DataFrame([
                {"Model":     "🩸 Diabetes",      "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model":     "❤️ Heart Disease", "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
                {"Model":     "💢 Hypertension",  "Algorithm": "Random Forest", "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
            ])
            # Try to load real metrics from trained bundles; fall back to "—" if not available
            _perf_diseases = [
                ("diabetes",     "🩸 Diabetes",      0),
                ("heart",        "❤️ Heart Disease", 1),
                ("hypertension", "💢 Hypertension",  2),
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
'''

if old in content:
    content = content.replace(old, new)
    open("app.py", "w", encoding="utf-8").write(content)
    print("REPLACED OK")
else:
    print("NOT FOUND — no change made")
