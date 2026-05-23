import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import load_demo_training, load_exact_process_rows

try:
    from src.ml_models import TARGETS
except Exception:
    TARGETS = [
        "relative_density_pct",
        "porosity_pct",
        "Ms_C",
        "Mf_C",
        "As_C",
        "Af_C",
        "hysteresis_K",
        "recoverable_strain_pct",
        "residual_strain_pct",
        "UTS_MPa",
        "elongation_pct",
    ]


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Physics-Informed ML Workbench", layout="wide")

st.title("Physics-Informed Machine-Learning Workbench")
st.caption(
    "ML for NiTi-AM using process descriptors, vaporization risk, thermal history, "
    "B2/B19′ phase logic, crystallography metrics, and cyclic-function indicators."
)

st.markdown(
    """
This upgraded ML page does **not** treat LPBF NiTi as a simple `P–v–h–t → property` problem.

It adds physics-informed descriptors from the project logic:

> LPBF parameters → thermal history → Ni loss / oxygen risk → B2/B19′ phase state → crystallography → cyclic superelastic stability

Use **exact literature rows** for defensible metrics.  
Use demo rows only to test the interface.
"""
)


# =============================================================================
# Helpers
# =============================================================================

def clamp(x, lo=0.0, hi=100.0):
    return np.maximum(lo, np.minimum(hi, x))


def risk_between(x, low_good, high_bad):
    x = pd.to_numeric(x, errors="coerce")
    if high_bad == low_good:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return pd.Series(clamp(100.0 * (x - low_good) / (high_bad - low_good)), index=x.index)


def inverse_risk(r):
    return 100.0 - pd.to_numeric(r, errors="coerce").fillna(50.0)


def safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def first_existing(df: pd.DataFrame, candidates: List[str], default: float) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def fill_reasonable(s: pd.Series, default: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().any():
        return s.fillna(float(s.median()))
    return pd.Series(default, index=s.index, dtype=float)


def quality_weight_from_data_quality(value: str) -> int:
    text = str(value).lower()

    if "exact" in text or "table" in text:
        return 100
    if "digitized" in text:
        return 80
    if "text" in text:
        return 70
    if "inferred" in text:
        return 45
    if "demo" in text or "synthetic" in text:
        return 10
    return 50


def score_label(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 65:
        return "promising"
    if score >= 45:
        return "mixed / uncertain"
    return "high-risk"


def risk_label(risk: float) -> str:
    if risk < 25:
        return "low"
    if risk < 50:
        return "moderate"
    if risk < 75:
        return "high"
    return "critical"


# =============================================================================
# Physics-informed feature engineering
# =============================================================================

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds NiTi-AM descriptors without requiring every column to exist.
    Missing inputs are filled with transparent defaults/medians.
    """
    out = df.copy()

    P = fill_reasonable(first_existing(out, ["laser_power_W", "power_W", "P_W"], 140.0), 140.0)
    v = fill_reasonable(first_existing(out, ["scan_speed_mm_s", "speed_mm_s", "v_mm_s"], 900.0), 900.0)
    h = fill_reasonable(first_existing(out, ["hatch_spacing_mm", "hatch_mm"], 0.08), 0.08)
    t = fill_reasonable(first_existing(out, ["layer_thickness_mm", "layer_mm"], 0.03), 0.03)

    remelt = fill_reasonable(first_existing(out, ["remelt_passes", "rescan_passes", "interlayer_remelting_passes"], 0.0), 0.0)
    overlap = fill_reasonable(first_existing(out, ["hatch_overlap_pct", "track_overlap_pct"], 35.0), 35.0)
    reheat = fill_reasonable(first_existing(out, ["layer_reheat_severity_pct", "thermal_reheat_pct"], 35.0), 35.0)
    buildT = fill_reasonable(first_existing(out, ["build_plate_C", "preheat_C", "substrate_temperature_C"], 100.0), 100.0)

    oxygen = fill_reasonable(first_existing(out, ["oxygen_ppm", "powder_oxygen_ppm", "chamber_oxygen_ppm"], 120.0), 120.0)
    reuse = fill_reasonable(first_existing(out, ["powder_reuse_cycles", "reuse_cycles"], 0.0), 0.0)
    secondary = fill_reasonable(first_existing(out, ["secondary_phase_signal_pct", "oxide_phase_signal_pct"], 10.0), 10.0)

    powder_Ni = fill_reasonable(first_existing(out, ["powder_Ni_at_pct", "nominal_Ni_at_pct"], 51.0), 51.0)
    measured_Ni = first_existing(out, ["measured_Ni_at_pct", "printed_Ni_at_pct", "effective_Ni_at_pct"], np.nan)

    Ms = fill_reasonable(first_existing(out, ["Ms_C"], -20.0), -20.0)
    Af = fill_reasonable(first_existing(out, ["Af_C"], 5.0), 5.0)
    service_T = fill_reasonable(first_existing(out, ["service_temperature_C", "test_temperature_C"], 25.0), 25.0)
    dsc_width = fill_reasonable(first_existing(out, ["DSC_peak_width_C", "transformation_peak_width_C"], 25.0), 25.0)
    hysteresis = fill_reasonable(first_existing(out, ["hysteresis_K", "hysteresis_C", "transformation_hysteresis_C"], 35.0), 35.0)

    B2 = fill_reasonable(first_existing(out, ["B2_fraction_pct", "b2_fraction_pct", "austenite_fraction_pct"], 80.0), 80.0)
    B19 = fill_reasonable(first_existing(out, ["B19_fraction_pct", "B19p_fraction_pct", "martensite_fraction_pct"], 20.0), 20.0)
    retained_B19 = fill_reasonable(first_existing(out, ["retained_B19_after_cycling_pct", "retained_martensite_pct"], 10.0), 10.0)

    kam = fill_reasonable(first_existing(out, ["mean_KAM_deg", "KAM_deg", "local_misorientation_deg"], 1.5), 1.5)
    high_kam = fill_reasonable(first_existing(out, ["high_KAM_fraction_pct"], 15.0), 15.0)
    residual_strain = fill_reasonable(first_existing(out, ["residual_strain_pct"], 1.0), 1.0)
    defect = fill_reasonable(first_existing(out, ["defect_fraction_pct", "porosity_pct"], 0.8), 0.8)

    lambda2_error = fill_reasonable(first_existing(out, ["lambda2_error", "abs_lambda2_minus_1"], 0.012), 0.012)
    vol_change = fill_reasonable(first_existing(out, ["normalized_volume_change_pct", "volume_change_pct"], 0.8), 0.8)
    texture_align = fill_reasonable(first_existing(out, ["texture_variant_alignment_pct", "variant_alignment_pct"], 60.0), 60.0)

    sigma0 = fill_reasonable(first_existing(out, ["sigma_Ms_at_Ms_MPa"], 80.0), 80.0)
    cc_slope = fill_reasonable(first_existing(out, ["clausius_clapeyron_slope_MPa_C", "d_sigma_dT_MPa_C"], 6.0), 6.0)

    # Core LPBF descriptors
    out["VED_J_mm3"] = out.get("VED_J_mm3", P / (v * h * t))
    out["linear_energy_density_J_mm"] = out.get("linear_energy_density_J_mm", P / v)
    out["areal_energy_density_J_mm2"] = P / (v * h)
    out["power_speed_ratio"] = P / v
    out["layer_hatch_product_mm2"] = h * t

    VED = pd.to_numeric(out["VED_J_mm3"], errors="coerce")
    LED = pd.to_numeric(out["linear_energy_density_J_mm"], errors="coerce")

    # Thermal history / remelting logic
    ved_thermal = risk_between(VED, 55.0, 170.0)
    led_thermal = risk_between(LED, 0.10, 0.45)
    remelt_risk = risk_between(remelt, 0.0, 4.0)
    overlap_risk = risk_between(overlap, 10.0, 70.0)
    buildplate_risk = risk_between(buildT, 50.0, 450.0)
    layer_thin_risk = risk_between(0.080 - t, 0.000, 0.060)

    out["thermal_cycle_index"] = clamp(
        0.26 * ved_thermal
        + 0.18 * led_thermal
        + 0.20 * remelt_risk
        + 0.13 * overlap_risk
        + 0.11 * reheat
        + 0.06 * buildplate_risk
        + 0.06 * layer_thin_risk
    )

    out["lack_of_fusion_risk"] = clamp(
        0.60 * risk_between(55.0 - VED, 0.0, 35.0)
        + 0.40 * risk_between(0.12 - LED, 0.0, 0.10)
    )

    # Ni-loss / composition logic
    ni_loss_risk_proxy = clamp(
        0.35 * ved_thermal
        + 0.25 * led_thermal
        + 0.15 * remelt_risk
        + 0.10 * overlap_risk
        + 0.10 * risk_between(buildT, 100.0, 500.0)
        + 0.05 * risk_between(oxygen, 50.0, 800.0)
    )

    out["ni_loss_risk_proxy"] = ni_loss_risk_proxy

    estimated_effective_Ni = powder_Ni - 0.0035 * ni_loss_risk_proxy
    out["effective_Ni_at_pct_model"] = measured_Ni.where(measured_Ni.notna(), estimated_effective_Ni)
    out["delta_Ni_at_pct_model"] = out["effective_Ni_at_pct_model"] - powder_Ni
    out["Ni_deviation_from_50p8_at_pct"] = (out["effective_Ni_at_pct_model"] - 50.8).abs()

    # Oxygen / precipitate / oxide logic
    out["oxide_precipitate_risk"] = clamp(
        0.42 * risk_between(oxygen, 50.0, 800.0)
        + 0.18 * risk_between(reuse, 0.0, 8.0)
        + 0.18 * secondary
        + 0.14 * out["thermal_cycle_index"]
        + 0.08 * risk_between(defect, 0.2, 3.0)
    )

    # Transformation / phase logic
    martensite_at_service = pd.Series(0.0, index=out.index)
    martensite_at_service = martensite_at_service.mask(service_T < Ms, 100.0)
    martensite_at_service = martensite_at_service.mask((service_T >= Ms) & (service_T < Af), 65.0)
    martensite_at_service = martensite_at_service.mask(service_T >= Af, risk_between(Af - service_T, -80.0, 0.0))

    out["martensite_at_service_risk"] = martensite_at_service

    out["phase_suitability_score"] = clamp(
        0.55 * B2
        + 0.25 * inverse_risk(B19)
        + 0.20 * inverse_risk(out["martensite_at_service_risk"])
    )

    # Residual strain / KAM / cyclic martensite stabilization
    residual_distortion = clamp(
        0.34 * risk_between(residual_strain, 0.2, 4.0)
        + 0.22 * risk_between(kam, 0.8, 5.0)
        + 0.18 * high_kam
        + 0.14 * retained_B19
        + 0.12 * risk_between(defect, 0.2, 4.0)
    )

    out["residual_distortion_risk"] = residual_distortion

    out["b19_stabilization_risk"] = clamp(
        0.22 * B19
        + 0.22 * retained_B19
        + 0.17 * residual_distortion
        + 0.15 * risk_between(dsc_width, 8.0, 55.0)
        + 0.12 * out["thermal_cycle_index"]
        + 0.12 * out["martensite_at_service_risk"]
    )

    out["transformation_broadening_risk"] = clamp(
        0.32 * risk_between(dsc_width, 8.0, 55.0)
        + 0.24 * risk_between(hysteresis, 15.0, 75.0)
        + 0.16 * out["thermal_cycle_index"]
        + 0.12 * residual_distortion
        + 0.10 * risk_between(out["Ni_deviation_from_50p8_at_pct"], 0.05, 0.80)
        + 0.06 * out["oxide_precipitate_risk"]
    )

    # sigma_Ms proxy
    delta_T = service_T - Ms
    out["delta_T_service_minus_Ms_C"] = delta_T
    out["sigma_Ms_proxy_MPa"] = sigma0 + cc_slope * np.maximum(0.0, delta_T)

    stress_window = pd.Series(100.0, index=out.index)
    stress_window = stress_window.mask(delta_T < 25.0, clamp(100.0 * (delta_T + 20.0) / 45.0))
    stress_window = stress_window.mask(delta_T > 110.0, clamp(100.0 * (220.0 - delta_T) / 110.0))
    stress_window = stress_window.mask(service_T < Af, stress_window * 0.55)
    out["stress_window_score"] = clamp(stress_window)

    out["plasticity_during_loading_risk"] = clamp(
        0.55 * risk_between(out["sigma_Ms_proxy_MPa"], 350.0, 850.0)
        + 0.25 * residual_distortion
        + 0.20 * risk_between(defect, 0.2, 4.0)
    )

    # Crystallography
    out["lambda2_risk"] = risk_between(lambda2_error, 0.005, 0.035)
    out["volume_mismatch_risk"] = risk_between(vol_change.abs(), 0.5, 3.0)

    out["crystallography_score"] = clamp(
        0.42 * inverse_risk(out["lambda2_risk"])
        + 0.25 * inverse_risk(out["volume_mismatch_risk"])
        + 0.20 * texture_align
        + 0.13 * inverse_risk(risk_between(kam, 0.8, 5.0))
    )

    # Final cyclic degradation and function score
    out["cyclic_degradation_risk"] = clamp(
        0.26 * out["b19_stabilization_risk"]
        + 0.23 * out["residual_distortion_risk"]
        + 0.16 * out["transformation_broadening_risk"]
        + 0.13 * out["plasticity_during_loading_risk"]
        + 0.10 * out["oxide_precipitate_risk"]
        + 0.08 * risk_between(defect, 0.2, 4.0)
        + 0.04 * out["lack_of_fusion_risk"]
    )

    out["cyclic_function_score_proxy"] = clamp(
        0.19 * out["phase_suitability_score"]
        + 0.17 * out["stress_window_score"]
        + 0.16 * out["crystallography_score"]
        + 0.15 * inverse_risk(out["cyclic_degradation_risk"])
        + 0.12 * inverse_risk(out["b19_stabilization_risk"])
        + 0.10 * inverse_risk(out["transformation_broadening_risk"])
        + 0.07 * inverse_risk(out["oxide_precipitate_risk"])
        + 0.04 * inverse_risk(out["lack_of_fusion_risk"])
    )

    if "data_quality" in out.columns:
        out["evidence_quality_weight"] = out["data_quality"].apply(quality_weight_from_data_quality)
    else:
        out["evidence_quality_weight"] = 50

    return out


# =============================================================================
# Feature selection / anti-leakage logic
# =============================================================================

PROCESS_FEATURES = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_mm",
    "layer_thickness_mm",
    "VED_J_mm3",
    "linear_energy_density_J_mm",
    "areal_energy_density_J_mm2",
    "power_speed_ratio",
    "layer_hatch_product_mm2",
    "powder_Ni_at_pct",
    "measured_Ni_at_pct",
    "effective_Ni_at_pct_model",
    "delta_Ni_at_pct_model",
    "Ni_deviation_from_50p8_at_pct",
]

PHYSICS_FEATURES = PROCESS_FEATURES + [
    "thermal_cycle_index",
    "lack_of_fusion_risk",
    "ni_loss_risk_proxy",
    "oxide_precipitate_risk",
    "martensite_at_service_risk",
    "phase_suitability_score",
    "delta_T_service_minus_Ms_C",
    "sigma_Ms_proxy_MPa",
    "stress_window_score",
]

FULL_EVIDENCE_FEATURES = PHYSICS_FEATURES + [
    "residual_distortion_risk",
    "b19_stabilization_risk",
    "transformation_broadening_risk",
    "plasticity_during_loading_risk",
    "lambda2_risk",
    "volume_mismatch_risk",
    "crystallography_score",
    "cyclic_degradation_risk",
    "cyclic_function_score_proxy",
    "oxygen_ppm",
    "remelt_passes",
    "hatch_overlap_pct",
    "mean_KAM_deg",
    "residual_strain_pct",
    "B2_fraction_pct",
    "B19_fraction_pct",
    "retained_B19_after_cycling_pct",
    "lambda2_error",
    "normalized_volume_change_pct",
]

LEAKY_BY_TARGET = {
    "residual_strain_pct": [
        "residual_distortion_risk",
        "cyclic_degradation_risk",
        "cyclic_function_score_proxy",
    ],
    "recoverable_strain_pct": [
        "cyclic_function_score_proxy",
        "cyclic_degradation_risk",
    ],
    "hysteresis_K": [
        "transformation_broadening_risk",
        "cyclic_degradation_risk",
        "cyclic_function_score_proxy",
    ],
    "Ms_C": [
        "delta_T_service_minus_Ms_C",
        "sigma_Ms_proxy_MPa",
        "stress_window_score",
        "martensite_at_service_risk",
    ],
    "Af_C": [
        "martensite_at_service_risk",
        "phase_suitability_score",
        "stress_window_score",
    ],
}


def usable_features(df: pd.DataFrame, feature_mode: str, target: str = None) -> List[str]:
    if feature_mode == "Process-only baseline":
        candidates = PROCESS_FEATURES
    elif feature_mode == "Physics-informed prediction descriptors":
        candidates = PHYSICS_FEATURES
    else:
        candidates = FULL_EVIDENCE_FEATURES

    features = [c for c in candidates if c in df.columns]

    # Remove target and constant/empty features
    remove = {target} if target else set()

    if target in LEAKY_BY_TARGET:
        remove.update(LEAKY_BY_TARGET[target])

    clean = []

    for c in features:
        if c in remove:
            continue

        s = pd.to_numeric(df[c], errors="coerce")

        if s.notna().sum() >= 2 and s.nunique(dropna=True) > 1:
            clean.append(c)

    return clean


# =============================================================================
# Model builders and CV
# =============================================================================

def make_regressor(kind: str):
    if kind == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if kind == "Extra Trees":
        model = ExtraTreesRegressor(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if kind == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if kind == "Ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42,
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def make_classifier(kind: str):
    if kind == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=1,
            class_weight="balanced",
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if kind == "Extra Trees":
        model = ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=1,
            class_weight="balanced",
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="auto",
                ),
            ),
        ]
    )


def regression_cv(df: pd.DataFrame, target: str, features: List[str], kind: str) -> Tuple[Dict, pd.DataFrame]:
    data = df[features + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])

    X = data[features]
    y = data[target]

    n = len(data)

    if n < 4:
        raise ValueError("Need at least 4 non-null rows for cross-validation.")

    n_splits = min(5, n)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []

    for fold, (tr, te) in enumerate(cv.split(X), start=1):
        model = make_regressor(kind)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": mean_absolute_error(y.iloc[te], pred),
                "RMSE": math.sqrt(mean_squared_error(y.iloc[te], pred)),
                "R2": r2_score(y.iloc[te], pred) if len(te) >= 2 else np.nan,
            }
        )

    cv_df = pd.DataFrame(rows)

    metrics = {
        "n_rows": n,
        "n_features": len(features),
        "MAE_mean": float(cv_df["MAE"].mean()),
        "MAE_std": float(cv_df["MAE"].std(ddof=0)),
        "RMSE_mean": float(cv_df["RMSE"].mean()),
        "RMSE_std": float(cv_df["RMSE"].std(ddof=0)),
        "R2_mean": float(cv_df["R2"].mean(skipna=True)),
        "R2_std": float(cv_df["R2"].std(skipna=True, ddof=0)),
    }

    return metrics, cv_df


def classification_cv(df: pd.DataFrame, target: str, features: List[str], kind: str) -> Tuple[Dict, pd.DataFrame]:
    data = df[features + [target]].copy()
    data = data.dropna(subset=[target])
    data[target] = data[target].astype(str)

    X = data[features]
    y = data[target]

    counts = y.value_counts()
    min_class = int(counts.min())
    n_classes = int(y.nunique())

    if len(data) < 6 or n_classes < 2:
        raise ValueError("Need at least 6 rows and at least 2 classes.")

    if min_class < 2:
        raise ValueError("At least one class has fewer than 2 rows, so stratified CV is not possible.")

    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    all_true = []
    all_pred = []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        model = make_classifier(kind)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])

        all_true.extend(y.iloc[te].tolist())
        all_pred.extend(pred.tolist())

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "accuracy": accuracy_score(y.iloc[te], pred),
                "balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
                "macro_F1": f1_score(y.iloc[te], pred, average="macro"),
            }
        )

    cv_df = pd.DataFrame(rows)

    metrics = {
        "n_rows": len(data),
        "n_features": len(features),
        "n_classes": n_classes,
        "accuracy_mean": float(cv_df["accuracy"].mean()),
        "balanced_accuracy_mean": float(cv_df["balanced_accuracy"].mean()),
        "macro_F1_mean": float(cv_df["macro_F1"].mean()),
    }

    return metrics, cv_df


def feature_importance_table(model: Pipeline, features: List[str]) -> pd.DataFrame:
    fitted = model.named_steps.get("model")

    if hasattr(fitted, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": features,
                "importance": fitted.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(fitted, "coef_"):
        coef = fitted.coef_

        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)

        return pd.DataFrame(
            {
                "feature": features,
                "importance": coef,
            }
        ).sort_values("importance", ascending=False)

    return pd.DataFrame()


# =============================================================================
# Load data
# =============================================================================

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
    key="source_mode",
)

if source_mode == "Exact literature rows only":
    raw_df = exact.copy()
    default_strict = True

elif source_mode == "Exact + demo rows for interface testing":
    raw_df = pd.concat([exact, demo], ignore_index=True, sort=False)
    default_strict = False
    st.warning(
        "Demo rows are included. Use this only to test the ML interface; do not report these metrics as real validation."
    )

else:
    uploaded = st.file_uploader("Upload training CSV", type=["csv"], key="uploaded_training_csv")

    if uploaded is None:
        st.stop()

    raw_df = pd.read_csv(uploaded)
    default_strict = True

strict = st.checkbox(
    "Exclude demo/synthetic rows from training",
    value=default_strict,
    key="strict_real_only",
)

df = raw_df.copy()

if strict and "data_quality" in df.columns:
    df = df[
        ~df["data_quality"]
        .astype(str)
        .str.contains("demo|synthetic", case=False, na=False)
    ].copy()

df = add_physics_features(df)

st.subheader("Dataset after filtering and physics-feature engineering")
st.dataframe(df, use_container_width=True, height=280)

with st.expander("Engineered feature preview"):
    engineered_cols = [
        c for c in df.columns
        if c in FULL_EVIDENCE_FEATURES or c in ["evidence_quality_weight"]
    ]
    st.dataframe(df[engineered_cols].head(50), use_container_width=True)


# =============================================================================
# Coverage audit
# =============================================================================

tab_audit, tab_reg, tab_class, tab_index, tab_plan = st.tabs(
    [
        "1. Data audit",
        "2. Regression",
        "3. Functional classification",
        "4. Rule-based index benchmark",
        "5. What data to extract next",
    ]
)


with tab_audit:
    st.subheader("Target coverage")

    numeric_targets = []

    for t in TARGETS:
        if t in df.columns:
            numeric_targets.append(t)

    extra_numeric_candidates = [
        "cyclic_function_score_proxy",
        "phase_suitability_score",
        "crystallography_score",
        "cyclic_degradation_risk",
        "b19_stabilization_risk",
        "thermal_cycle_index",
    ]

    for t in extra_numeric_candidates:
        if t in df.columns and t not in numeric_targets:
            numeric_targets.append(t)

    coverage = []

    for t in numeric_targets:
        coverage.append(
            {
                "target": t,
                "non_null_rows": int(pd.to_numeric(df[t], errors="coerce").notna().sum()),
                "target_type": "measured/literature" if t in TARGETS else "engineered proxy",
                "warning": (
                    "reportable if exact measured rows"
                    if t in TARGETS
                    else "diagnostic proxy, not independent experimental target"
                ),
            }
        )

    coverage_df = pd.DataFrame(coverage)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    st.subheader("Feature groups")

    feature_mode_preview = st.selectbox(
        "Preview feature group",
        [
            "Process-only baseline",
            "Physics-informed prediction descriptors",
            "Full evidence descriptors / diagnostic mode",
        ],
        key="feature_preview_mode",
    )

    preview_features = usable_features(df, feature_mode_preview)
    st.write(f"{len(preview_features)} usable features")
    st.dataframe(pd.DataFrame({"feature": preview_features}), use_container_width=True, hide_index=True)

    st.info(
        "Use Process-only for early process prediction. Use Physics-informed descriptors for stronger prediction. "
        "Use Full evidence descriptors only for diagnosis after characterization, because it may include post-process evidence."
    )


# =============================================================================
# Regression tab
# =============================================================================

with tab_reg:
    st.subheader("Regression with anti-leakage feature modes")

    available_targets = [
        row["target"]
        for _, row in coverage_df.iterrows()
        if row["non_null_rows"] >= 6
    ] if len(coverage_df) else []

    if not available_targets:
        st.warning("No numeric target has at least 6 non-null rows after filtering.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    target = c1.selectbox("Regression target", available_targets, key="reg_target")

    feature_mode = c2.selectbox(
        "Feature mode",
        [
            "Process-only baseline",
            "Physics-informed prediction descriptors",
            "Full evidence descriptors / diagnostic mode",
        ],
        key="reg_feature_mode",
    )

    model_kind = c3.selectbox(
        "Model",
        [
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "Gaussian Process",
            "Ridge",
        ],
        key="reg_model_kind",
    )

    features = usable_features(df, feature_mode, target=target)

    if len(features) < 2:
        st.error("Not enough usable features for this target/mode.")
        st.stop()

    st.caption(
        f"Using {len(features)} features. Features that directly leak the selected target are automatically removed when known."
    )

    try:
        metrics, cv_df = regression_cv(df, target, features, model_kind)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows used", metrics["n_rows"])
    m2.metric("Features", metrics["n_features"])
    m3.metric("CV MAE", f"{metrics['MAE_mean']:.3f}")
    m4.metric("CV RMSE", f"{metrics['RMSE_mean']:.3f}")
    m5.metric("CV R²", f"{metrics['R2_mean']:.3f}")

    if metrics["n_rows"] < 20:
        st.warning(
            "Small dataset: treat metrics as screening only. Use external validation once more literature rows are extracted."
        )

    if target == "cyclic_function_score_proxy":
        st.warning(
            "You selected an engineered proxy as target. This is useful for benchmarking/design-space ranking, "
            "but it is not an independent experimental validation target."
        )

    st.subheader("Cross-validation folds")
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    train_data = df[features + [target]].copy()
    train_data[target] = pd.to_numeric(train_data[target], errors="coerce")
    train_data = train_data.dropna(subset=[target])

    final_model = make_regressor(model_kind)
    final_model.fit(train_data[features], train_data[target])

    st.subheader("Feature importance / coefficient magnitude")
    importance = feature_importance_table(final_model, features)

    if len(importance):
        st.dataframe(importance, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                importance.head(20).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top feature importance",
            ),
            use_container_width=True,
        )
    else:
        st.info("This model type does not expose simple feature importances.")

    st.subheader("Predict a new condition")

    with st.expander("Edit model input features", expanded=True):
        pred_values = {}

        cols_ui = st.columns(4)

        for i, f in enumerate(features):
            default = float(pd.to_numeric(train_data[f], errors="coerce").median())
            pred_values[f] = cols_ui[i % 4].number_input(
                f,
                value=default,
                key=f"pred_reg_{f}",
            )

    pred_df = pd.DataFrame([pred_values])
    pred = float(final_model.predict(pred_df)[0])

    st.metric(f"Predicted {target}", f"{pred:.3f}")

    if "VED_J_mm3" in df.columns and target in df.columns:
        plot_df = df.copy()
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")

        fig = px.scatter(
            plot_df,
            x="VED_J_mm3",
            y=target,
            color="data_quality" if "data_quality" in plot_df.columns else None,
            hover_data=[c for c in ["sample_id", "source_id", "functional_class"] if c in plot_df.columns],
            title=f"{target} vs VED",
        )
        fig.add_hline(y=pred, line_dash="dash", annotation_text="current prediction")
        st.plotly_chart(fig, use_container_width=True)

    model_card = {
        "page": "Physics-Informed ML Workbench",
        "task": "regression",
        "target": target,
        "model_kind": model_kind,
        "feature_mode": feature_mode,
        "features": features,
        "metrics": metrics,
        "strict_real_only": strict,
        "warnings": [
            "Use exact/digitized literature rows for reportable metrics.",
            "Demo/synthetic rows are only for interface testing.",
            "Full evidence descriptors may include post-process characterization and should be used for diagnosis, not early process-only prediction.",
            "Engineered proxy targets are not independent experimental validation targets.",
        ],
    }

    st.download_button(
        "Download regression model card JSON",
        json.dumps(model_card, indent=2),
        "regression_model_card.json",
        mime="application/json",
    )


# =============================================================================
# Classification tab
# =============================================================================

with tab_class:
    st.subheader("Functional classification")

    possible_class_targets = []

    for c in df.columns:
        if c in ["functional_class", "outcome_class", "superelastic_class", "process_class"]:
            possible_class_targets.append(c)

    for c in df.columns:
        if df[c].dtype == "object":
            nunique = df[c].dropna().astype(str).nunique()
            if 2 <= nunique <= 10 and c not in possible_class_targets:
                possible_class_targets.append(c)

    if not possible_class_targets:
        st.warning("No usable class-label column found. Add `functional_class` to your dataset.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    class_target = c1.selectbox("Class target", possible_class_targets, key="class_target")

    class_feature_mode = c2.selectbox(
        "Feature mode",
        [
            "Process-only baseline",
            "Physics-informed prediction descriptors",
            "Full evidence descriptors / diagnostic mode",
        ],
        key="class_feature_mode",
    )

    class_model_kind = c3.selectbox(
        "Classifier",
        ["Random Forest", "Extra Trees", "Logistic Regression"],
        key="class_model_kind",
    )

    class_features = usable_features(df, class_feature_mode, target=class_target)

    if len(class_features) < 2:
        st.error("Not enough usable features for classification.")
        st.stop()

    class_data = df[class_features + [class_target]].dropna(subset=[class_target]).copy()
    class_data[class_target] = class_data[class_target].astype(str)

    st.subheader("Class distribution")
    st.dataframe(
        class_data[class_target].value_counts().rename_axis("class").reset_index(name="rows"),
        use_container_width=True,
        hide_index=True,
    )

    try:
        class_metrics, class_cv_df = classification_cv(
            class_data,
            class_target,
            class_features,
            class_model_kind,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows used", class_metrics["n_rows"])
        m2.metric("Classes", class_metrics["n_classes"])
        m3.metric("Balanced accuracy", f"{class_metrics['balanced_accuracy_mean']:.3f}")
        m4.metric("Macro F1", f"{class_metrics['macro_F1_mean']:.3f}")

        st.dataframe(class_cv_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.warning(
            f"Cross-validation is unstable or impossible: {exc}. "
            "The model will still fit for interface testing."
        )

    clf = make_classifier(class_model_kind)
    clf.fit(class_data[class_features], class_data[class_target])

    importance = feature_importance_table(clf, class_features)

    st.subheader("Classification feature importance")

    if len(importance):
        st.dataframe(importance, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(
                importance.head(20).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top classification features",
            ),
            use_container_width=True,
        )
    else:
        st.info("This classifier does not expose simple feature importance.")

    st.subheader("Classify a new process / evidence row")

    with st.expander("Edit classifier input features", expanded=True):
        class_values = {}
        cols_ui = st.columns(4)

        for i, f in enumerate(class_features):
            default = float(pd.to_numeric(class_data[f], errors="coerce").median())
            class_values[f] = cols_ui[i % 4].number_input(
                f,
                value=default,
                key=f"pred_class_{f}",
            )

    new_class_df = pd.DataFrame([class_values])
    pred_class = clf.predict(new_class_df)[0]

    st.metric("Predicted class", pred_class)

    if hasattr(clf.named_steps["model"], "predict_proba"):
        proba = clf.predict_proba(new_class_df)[0]

        proba_df = pd.DataFrame(
            {
                "class": clf.named_steps["model"].classes_,
                "probability": proba,
            }
        ).sort_values("probability", ascending=False)

        st.dataframe(proba_df, use_container_width=True, hide_index=True)

    if "VED_J_mm3" in df.columns and "thermal_cycle_index" in df.columns:
        fig = px.scatter(
            df,
            x="VED_J_mm3",
            y="thermal_cycle_index",
            color=class_target,
            hover_data=[c for c in ["sample_id", "source_id", "functional_observation"] if c in df.columns],
            title="Functional class map: VED vs thermal-cycle index",
        )
        st.plotly_chart(fig, use_container_width=True)

    class_card = {
        "page": "Physics-Informed ML Workbench",
        "task": "classification",
        "target": class_target,
        "model_kind": class_model_kind,
        "feature_mode": class_feature_mode,
        "features": class_features,
        "strict_real_only": strict,
        "warning": "Classification is more defensible than regression when exact row-level numerical targets are sparse.",
    }

    st.download_button(
        "Download classification model card JSON",
        json.dumps(class_card, indent=2),
        "classification_model_card.json",
        mime="application/json",
    )


# =============================================================================
# Rule-based index benchmark
# =============================================================================

with tab_index:
    st.subheader("Rule-based physics index benchmark")

    st.markdown(
        """
This tab ranks rows using the engineered **cyclic_function_score_proxy**.

This is not trained ML.  
It is a transparent physics-informed benchmark that helps you see whether the ML agrees with the metallurgical logic.
"""
    )

    rank_cols = [
        c for c in [
            "sample_id",
            "source_id",
            "data_quality",
            "functional_class",
            "VED_J_mm3",
            "linear_energy_density_J_mm",
            "thermal_cycle_index",
            "ni_loss_risk_proxy",
            "oxide_precipitate_risk",
            "phase_suitability_score",
            "crystallography_score",
            "cyclic_degradation_risk",
            "cyclic_function_score_proxy",
        ]
        if c in df.columns
    ]

    ranked = df[rank_cols].copy()

    if "cyclic_function_score_proxy" in ranked.columns:
        ranked = ranked.sort_values("cyclic_function_score_proxy", ascending=False)

    st.dataframe(ranked, use_container_width=True, hide_index=True)

    if "cyclic_function_score_proxy" in df.columns:
        fig = px.scatter(
            df,
            x="thermal_cycle_index",
            y="cyclic_function_score_proxy",
            color="functional_class" if "functional_class" in df.columns else None,
            hover_data=[c for c in ["sample_id", "source_id", "data_quality"] if c in df.columns],
            title="Rule-based benchmark: thermal history vs cyclic function proxy",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How to interpret the proxy")

    explain = pd.DataFrame(
        [
            {
                "proxy feature": "thermal_cycle_index",
                "meaning": "Repeated heat exposure from VED, LED, remelting, overlap, reheating and layer thickness.",
                "bad sign": "Too high can stabilize B19′, increase residual strain and broaden transformation.",
            },
            {
                "proxy feature": "ni_loss_risk_proxy",
                "meaning": "Estimated selective Ni evaporation tendency.",
                "bad sign": "High values may shift Af/Ms upward and create martensite-at-service risk.",
            },
            {
                "proxy feature": "oxide_precipitate_risk",
                "meaning": "Oxygen/reuse/secondary-phase risk.",
                "bad sign": "Can pin transformation and disturb matrix Ni/Ti balance.",
            },
            {
                "proxy feature": "crystallography_score",
                "meaning": "λ₂, volume mismatch, texture alignment and KAM support.",
                "bad sign": "Low values suggest poor B2/B19′ compatibility or high local distortion.",
            },
            {
                "proxy feature": "cyclic_degradation_risk",
                "meaning": "Risk of retained B19′, residual strain, broad transformation and plasticity.",
                "bad sign": "High values predict superelastic degradation after cycling.",
            },
        ]
    )

    st.dataframe(explain, use_container_width=True, hide_index=True)


# =============================================================================
# Data extraction plan
# =============================================================================

with tab_plan:
    st.subheader("What to extract next to make the ML genuinely strong")

    st.markdown(
        """
The next jump is not a more complicated model.  
The next jump is **more exact row-level data** from LPBF / DED / EBF³ NiTi papers.

The model should learn from measured rows, while the physics-informed descriptors help it generalize.
"""
    )

    plan = pd.DataFrame(
        [
            {
                "data group": "Process",
                "columns to extract": "laser_power_W, scan_speed_mm_s, hatch_spacing_mm, layer_thickness_mm, remelt_passes, scan_strategy, build_plate_C",
                "why": "Needed for VED, LED, thermal-cycle index and process comparison.",
            },
            {
                "data group": "Composition",
                "columns to extract": "powder_Ni_at_pct, measured_Ni_at_pct, oxygen_ppm, powder_reuse_cycles",
                "why": "Ni loss and oxygen strongly affect transformation behavior.",
            },
            {
                "data group": "Phase / transformation",
                "columns to extract": "B2_fraction_pct, B19_fraction_pct, Ms_C, Mf_C, As_C, Af_C, DSC_peak_width_C, hysteresis_K",
                "why": "Defines whether service behavior is B2, B19′ or mixed.",
            },
            {
                "data group": "Crystallography",
                "columns to extract": "B2_a_A, B19p_a_A, B19p_b_A, B19p_c_A, beta_deg, lambda2_error, normalized_volume_change_pct",
                "why": "Needed for compatibility and correspondence-theory descriptors.",
            },
            {
                "data group": "EBSD/TKD",
                "columns to extract": "texture_variant_alignment_pct, mean_KAM_deg, high_KAM_fraction_pct, grain_size_um, phase_map_fraction",
                "why": "Needed to quantify texture, residual strain and variant accommodation.",
            },
            {
                "data group": "Mechanical function",
                "columns to extract": "recoverable_strain_pct, residual_strain_pct, sigma_Ms_MPa, UTS_MPa, elongation_pct, cycles_tested",
                "why": "Final validation of superelastic/shape-memory performance.",
            },
            {
                "data group": "Cyclic stability",
                "columns to extract": "retained_B19_after_cycling_pct, residual_strain_after_cycles_pct, loop_hysteresis_MJ_m3",
                "why": "Directly captures degradation mechanism after repeated loading.",
            },
            {
                "data group": "Data quality",
                "columns to extract": "data_quality, source_id, sample_id, extraction_method, measurement_method",
                "why": "Keeps exact data separate from demo/inferred data.",
            },
        ]
    )

    st.dataframe(plan, use_container_width=True, hide_index=True)

    st.subheader("Best ML strategy for the paper/repo")

    st.markdown(
        """
1. **Now:** use exact rows for classification and physics-index benchmarking.
2. **Short term:** add 30–50 exact rows from LPBF/DED/EBF³ NiTi papers.
3. **Medium term:** train regression only for targets with enough exact measured rows.
4. **Long term:** use external validation by holding out entire papers, not random rows.
5. **Always:** compare ML against the rule-based physics index. If ML contradicts metallurgy, inspect the data.
"""
    )
