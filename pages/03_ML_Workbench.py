import streamlit as st
import pandas as pd
import plotly.express as px
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.ml_models import TARGETS, fit_single_target, predict_single, available_features
from src.data_utils import load_exact_process_rows, load_demo_training


st.title("Machine-Learning Workbench")
st.caption(
    "Regression when enough real numerical rows exist; classification/benchmarking when exact data are still sparse."
)

exact = load_exact_process_rows()
demo = load_demo_training()

source_mode = st.radio(
    "Training data",
    [
        "Exact literature rows only",
        "Exact + demo rows for interface testing",
        "Upload CSV",
    ],
    horizontal=True,
)

if source_mode == "Exact literature rows only":
    df = exact.copy()
    default_strict = True

elif source_mode == "Exact + demo rows for interface testing":
    df = pd.concat([exact, demo], ignore_index=True, sort=False)
    default_strict = False
    st.warning(
        "Demo rows are included. Use this only to test the ML interface; do not report these metrics as literature validation."
    )

else:
    file = st.file_uploader("Upload training CSV", type=["csv"])
    if file is None:
        st.stop()
    df = pd.read_csv(file)
    default_strict = True

st.subheader("Dataset")
st.dataframe(df, use_container_width=True, height=240)

strict = st.checkbox("Exclude demo rows from regression training", value=default_strict)

train_df = df.copy()
if strict and "data_quality" in train_df.columns:
    train_df = train_df[
        ~train_df["data_quality"].astype(str).str.contains("demo", case=False, na=False)
    ]

st.subheader("Target coverage")

coverage = []
for t in TARGETS:
    if t in train_df.columns:
        coverage.append(
            {
                "target": t,
                "non_null_rows_after_filter": int(
                    pd.to_numeric(train_df[t], errors="coerce").notna().sum()
                ),
            }
        )

coverage_df = pd.DataFrame(coverage)
st.dataframe(coverage_df, use_container_width=True)

available_regression_targets = (
    coverage_df[coverage_df["non_null_rows_after_filter"] >= 8]["target"].tolist()
    if len(coverage_df)
    else []
)

tab_reg, tab_class, tab_notes = st.tabs(
    ["Regression", "Functional classification", "Why sparse data is okay"]
)

with tab_reg:
    if not available_regression_targets:
        st.warning(
            "No regression target has at least 8 non-null rows after the current filter."
        )

        st.markdown(
            """
This is not a bug. The exact extracted dataset currently contains strong **process/function labels**, but not enough row-level density, DSC, tensile or hysteresis values for a defensible regression model.

Use one of these options:

1. Switch to **Exact + demo rows** and uncheck demo exclusion to test the interface.
2. Upload your own CSV with row-level values for `Af_C`, `relative_density_pct`, `hysteresis_K`, `UTS_MPa`, etc.
3. Continue using the exact dataset for **classification and benchmark comparison**, which is valid now.
"""
        )

    else:
        c1, c2 = st.columns(2)
        target = c1.selectbox("Regression target", available_regression_targets)
        model_kind = c2.selectbox(
            "Model",
            ["Gaussian Process", "Random Forest", "Extra Trees", "Gradient Boosting"],
        )

        try:
            model, cols, metrics = fit_single_target(
                df, target, model_kind=model_kind, strict_real_only=strict
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows used", metrics["n_rows"])
        m2.metric("CV R²", f"{metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}")
        m3.metric("CV MAE", f"{metrics['mae_mean']:.3f}")
        m4.metric("CV RMSE", f"{metrics['rmse_mean']:.3f}")

        st.subheader("Prediction panel")

        values = {}
        cols_ui = st.columns(4)

        for i, col in enumerate(cols):
            default = float(pd.to_numeric(df[col], errors="coerce").median())
            values[col] = cols_ui[i % 4].number_input(col, value=default)

        mean, std = predict_single(model, cols, values, model_kind=model_kind)

        if std is None:
            st.metric(f"Predicted {target}", f"{mean:.3f}")
        else:
            st.metric(f"Predicted {target}", f"{mean:.3f} ± {std:.3f}")

        if "VED_J_mm3" in df.columns:
            fig = px.scatter(
                df,
                x="VED_J_mm3",
                y=target,
                color="data_quality" if "data_quality" in df.columns else None,
                hover_data=[c for c in ["source_id", "sample_id"] if c in df.columns],
            )
            fig.add_hline(y=mean, line_dash="dash", annotation_text="current prediction")
            st.plotly_chart(fig, use_container_width=True)

        model_card = {
            "model_kind": model_kind,
            "target": target,
            "features": cols,
            "metrics": metrics,
            "strict_real_only": strict,
            "warning": "Metrics are defensible only when trained on exact_table/exact_text/digitized_figure rows, not demo_synthetic rows.",
        }

        st.download_button(
            "Download model card JSON",
            json.dumps(model_card, indent=2),
            "model_card.json",
        )

with tab_class:
    st.markdown(
        """
The current exact literature rows are already useful for a **functional class classifier** because each row has a process condition and a label such as excellent, partial, or poor/not tested.
"""
    )

    features = [
        c
        for c in [
            "laser_power_W",
            "scan_speed_mm_s",
            "hatch_spacing_mm",
            "layer_thickness_mm",
            "VED_J_mm3",
            "powder_Ni_at_pct",
            "measured_Ni_at_pct",
        ]
        if c in exact.columns
    ]

    class_df = exact.dropna(subset=["functional_class"]).copy()

    if len(class_df) >= 8 and len(features) >= 2:
        X = class_df[features]
        y = class_df["functional_class"].astype(str)

        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        )

        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            st.metric(
                "Exact-row classification CV accuracy",
                f"{scores.mean():.2f} ± {scores.std():.2f}",
            )
        except Exception:
            st.info(
                "Class distribution is small; cross-validation is unstable, but the classifier can still be fitted for demonstration."
            )

        clf.fit(X, y)

        st.subheader("Classify a new process")

        values = {}
        cols = st.columns(4)

        for i, col in enumerate(features):
            default = float(pd.to_numeric(exact[col], errors="coerce").median())
            values[col] = cols[i % 4].number_input(
                f"classifier_{col}", value=default, key=f"class_{col}"
            )

        pred = clf.predict(pd.DataFrame([values]))[0]
        proba = clf.predict_proba(pd.DataFrame([values]))[0]

        st.metric("Predicted functional class", pred)

        st.dataframe(
            pd.DataFrame({"class": clf.classes_, "probability": proba}).sort_values(
                "probability", ascending=False
            ),
            use_container_width=True,
        )

        fig = px.scatter(
            exact,
            x="VED_J_mm3",
            y="laser_power_W",
            color="functional_class",
            hover_data=["sample_id", "functional_observation"],
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough exact functional-class data.")

with tab_notes:
    st.markdown(
        """
### Why the app refuses some regression runs

If a target has zero exact rows, a regression model would be fake.  
For example, `relative_density_pct` is in the schema, but the current exact A1-D4 table does not provide row-level density values. Therefore the app should not train density regression on exact data yet.

### What to add next

To make ML top-level, extract these row-level columns from papers:

- `relative_density_pct`
- `porosity_pct`
- `Ms_C`, `Mf_C`, `As_C`, `Af_C`
- `hysteresis_K`
- `recoverable_strain_pct`
- `residual_strain_pct`
- `UTS_MPa`
- `elongation_pct`
- `oxygen_ppm`
- `measured_Ni_at_pct`
- `heat_treatment_C`, `heat_treatment_min`

### Strong model strategy

Use exact rows only for reported metrics. Use demo rows only to test the app.  
When the benchmark grows beyond about 100 rows, compare Gaussian Process, Extra Trees, Random Forest, SVR/GRNN and XGBoost-like models under the same external validation split.
"""
    )
