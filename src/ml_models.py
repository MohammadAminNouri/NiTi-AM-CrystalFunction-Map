import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_mm",
    "layer_thickness_mm",
    "VED_J_mm3",
    "powder_Ni_at_pct",
    "measured_Ni_at_pct",
    "oxygen_ppm",
    "build_plate_C",
    "remelt_passes",
    "heat_treatment_C",
    "heat_treatment_min",
    "B2_a_A",
    "B19p_a_A",
    "B19p_b_A",
    "B19p_c_A",
    "B19p_beta_deg",
    "lambda2_proxy",
]

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


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in df.columns]


def available_targets(df: pd.DataFrame) -> list[str]:
    usable = []

    for target in TARGETS:
        if target not in df.columns:
            continue

        non_null = pd.to_numeric(df[target], errors="coerce").notna().sum()

        if non_null >= 5:
            usable.append(target)

    return usable


def make_model(kind: str):
    if kind == "Gaussian Process":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=(
                            ConstantKernel(1.0)
                            * RBF(length_scale=1.0)
                            + WhiteKernel(noise_level=1.0)
                        ),
                        normalize_y=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    if kind == "Extra Trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        random_state=42,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )

    if kind == "Gradient Boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def fit_single_target(
    df: pd.DataFrame,
    target: str,
    model_kind: str = "Random Forest",
    strict_real_only: bool = True,
):
    work = df.copy()

    if strict_real_only and "data_quality" in work.columns:
        work = work[
            ~work["data_quality"]
            .astype(str)
            .str.contains("demo", case=False, na=False)
        ]

    cols = available_features(work)

    if len(cols) < 2:
        raise ValueError("Need at least two usable feature columns.")

    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])

    if len(work) < 8:
        raise ValueError(
            f"Need at least 8 non-null rows for defensible training of {target}; "
            f"got {len(work)}."
        )

    X = work[cols]
    y = work[target].astype(float)

    model = make_model(model_kind)

    cv = KFold(
        n_splits=min(5, len(work)),
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=(
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ),
        return_train_score=False,
    )

    model.fit(X, y)

    metrics = {
        "n_rows": int(len(work)),
        "target": target,
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"])),
        "mae_mean": float(-np.mean(scores["test_neg_mean_absolute_error"])),
        "rmse_mean": float(-np.mean(scores["test_neg_root_mean_squared_error"])),
    }

    return model, cols, metrics


def predict_single(
    model,
    cols: list[str],
    values: dict,
    model_kind: str = "Random Forest",
):
    X = pd.DataFrame([{c: values.get(c, np.nan) for c in cols}])

    if model_kind == "Gaussian Process":
        transformed = model[:-1].transform(X)
        mean, std = model[-1].predict(transformed, return_std=True)

        return float(mean[0]), float(std[0])

    prediction = model.predict(X)

    return float(prediction[0]), None
