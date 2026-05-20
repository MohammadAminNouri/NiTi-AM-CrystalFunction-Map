import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

FEATURES = [
    "laser_power_W", "scan_speed_mm_s", "hatch_spacing_mm", "layer_thickness_mm",
    "VED_J_mm3", "powder_Ni_at_pct", "measured_Ni_at_pct", "oxygen_ppm",
    "build_plate_C", "remelt_passes", "heat_treatment_C", "heat_treatment_min",
    "B2_a_A", "B19p_a_A", "B19p_b_A", "B19p_c_A", "B19p_beta_deg", "lambda2_proxy"
]

TARGETS = [
    "relative_density_pct", "porosity_pct", "Ms_C", "Mf_C", "As_C", "Af_C",
    "hysteresis_K", "recoverable_strain_pct", "residual_strain_pct",
    "UTS_MPa", "elongation_pct"
]

def available_features(df):
    return [c for c in FEATURES if c in df.columns]

def available_targets(df):
    return [c for c in TARGETS if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5]

def make_model(kind):
    if kind == "Gaussian Process":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                normalize_y=True,
                random_state=42
            )),
        ])
    if kind == "Extra Trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)),
        ])
    if kind == "Gradient Boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=42)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=2)),
    ])

def fit_single_target(df, target, model_kind="Random Forest", strict_real_only=True):
    work = df.copy()
    if strict_real_only and "data_quality" in work.columns:
        work = work[~work["data_quality"].astype(str).str.contains("demo", case=False, na=False)]
    cols = available_features(work)
    if len(cols) < 2:
        raise ValueError("Need at least two usable feature columns.")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    if len(work) < 8:
        raise ValueError(f"Need at least 8 non-null rows for defensible training of {target}; got {len(work)}.")
    X, y = work[cols], work[target].astype(float)
    model = make_model(model_kind)
    cv = KFold(n_splits=min(5, len(work)), shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"), return_train_score=False)
    model.fit(X, y)
    return model, cols, {
        "n_rows": int(len(work)),
        "target": target,
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"])),
        "mae_mean": float(-np.mean(scores["test_neg_mean_absolute_error"])),
        "rmse_mean": float(-np.mean(scores["test_neg_root_mean_squared_error"])),
    }

def predict_single(model, cols, values, model_kind="Random Forest"):
    X = pd.DataFrame([{c: values.get(c, np.nan) for c in cols}])
    if model_kind == "Gaussian Process":
        Xt = model[:-1].transform(X)
        mean, std = model[-1].predict(Xt, return_std=True)
        return float(mean[0]), float(std[0])
    return float(model.predict(X)[0]), None

def feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=8, random_state=42)
    return pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
