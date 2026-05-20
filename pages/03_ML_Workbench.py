import streamlit as st
import pandas as pd
import plotly.express as px
import json
from src.ml_models import FEATURES, TARGETS, available_targets, fit_single_target, predict_single, available_features
from src.data_utils import load_exact_process_rows, load_demo_training

st.title("Machine-Learning Workbench")
st.caption("Train defensible models. The page warns when the dataset is too small or demo rows are used.")

exact = load_exact_process_rows()
demo = load_demo_training()

source_mode = st.radio("Training data", ["Exact literature rows only", "Exact + demo rows for interface testing", "Upload CSV"], horizontal=True)
if source_mode == "Exact literature rows only":
    df = exact.copy()
elif source_mode == "Exact + demo rows for interface testing":
    df = pd.concat([exact, demo], ignore_index=True, sort=False)
    st.warning("Demo rows are included. Do not report these metrics as literature validation.")
else:
    file = st.file_uploader("Upload training CSV", type=["csv"])
    if file is None:
        st.stop()
    df = pd.read_csv(file)

st.subheader("Dataset")
st.dataframe(df, use_container_width=True, height=240)

features = available_features(df)
targets = [t for t in TARGETS if t in df.columns and pd.to_numeric(df[t], errors="coerce").notna().sum() >= 5]
if len(targets) == 0:
    st.error("No target has at least five numeric rows. Add more exact rows or switch to demo mode to test the interface.")
    st.stop()

c1, c2, c3 = st.columns(3)
target = c1.selectbox("Target", targets)
model_kind = c2.selectbox("Model", ["Gaussian Process", "Random Forest", "Extra Trees", "Gradient Boosting"])
strict = c3.checkbox("Exclude demo rows from training", value=True)

try:
    model, cols, metrics = fit_single_target(df, target, model_kind=model_kind, strict_real_only=strict)
except Exception as e:
    st.error(str(e))
    st.info("This is intentional: the app should not pretend to validate ML on tiny or non-real datasets.")
    st.stop()

m1,m2,m3,m4 = st.columns(4)
m1.metric("Rows used", metrics["n_rows"])
m2.metric("CV R²", f"{metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}")
m3.metric("CV MAE", f"{metrics['mae_mean']:.3f}")
m4.metric("CV RMSE", f"{metrics['rmse_mean']:.3f}")

st.subheader("Prediction panel")
values = {}
cols_ui = st.columns(4)
for i, col in enumerate(cols):
    default = float(pd.to_numeric(df[col], errors="coerce").median())
    values[col] = cols_ui[i%4].number_input(col, value=default)

mean, std = predict_single(model, cols, values, model_kind=model_kind)
if std is None:
    st.metric(f"Predicted {target}", f"{mean:.3f}")
else:
    st.metric(f"Predicted {target}", f"{mean:.3f} ± {std:.3f}")

if "VED_J_mm3" in df.columns:
    fig = px.scatter(df, x="VED_J_mm3", y=target, color="data_quality" if "data_quality" in df.columns else None,
                     hover_data=[c for c in ["source_id","sample_id"] if c in df.columns])
    fig.add_hline(y=mean, line_dash="dash", annotation_text="current prediction")
    st.plotly_chart(fig, use_container_width=True)

model_card = {
    "model_kind": model_kind,
    "target": target,
    "features": cols,
    "metrics": metrics,
    "strict_real_only": strict,
    "warning": "Metrics are only defensible if trained on exact_table/exact_text/digitized_figure rows, not demo_synthetic rows."
}
st.download_button("Download model card JSON", json.dumps(model_card, indent=2), "model_card.json")
