from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

from src.image_analysis import load_image, porosity_screen, image_texture_descriptors
from src.xrd_tools import read_xrd_csv
from src.ebsd_tools import read_ebsd_csv, ebsd_summary, grid_for_plot


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Characterization + Crystallography Lab", layout="wide")

st.title("Characterization + Crystallography Lab")
st.caption(
    "Microstructure, XRD, EBSD/TKD and B2→B19′ crystallography screening for LPBF NiTi."
)

st.markdown(
    """
This page connects characterization evidence to the NiTi function question:

> Is the printed NiTi mainly B2/austenite, B19′/martensite, mixed, defective, textured, or crystallographically compatible with recoverable transformation?

This is a **screening and reporting tool**, not a replacement for DSC, Rietveld refinement, SEM validation, ICP/EDS, or full EBSD/TKD variant reconstruction.
"""
)


# =============================================================================
# Constants and helper functions
# =============================================================================

CU_K_ALPHA_A = 1.5406


def make_demo_microstructure(size: int = 512) -> Image.Image:
    """
    Fallback synthetic microstructure image.
    Used if assets/demo_microstructure.png is missing.
    """
    rng = np.random.default_rng(42)

    base = np.full((size, size), 185, dtype=np.uint8)
    noise = rng.normal(0, 12, (size, size))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # dark pores
    for _ in range(35):
        x = int(rng.integers(20, size - 20))
        y = int(rng.integers(20, size - 20))
        r = int(rng.integers(3, 12))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(25, 25, 25))

    # crack-like feature
    pts = []
    x0 = int(size * 0.18)
    y0 = int(size * 0.58)
    for i in range(11):
        pts.append((x0 + i * 30, y0 + int(rng.normal(0, 10))))
    draw.line(pts, fill=(20, 20, 20), width=4)

    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def load_demo_or_uploaded_image(demo: bool, uploaded_file) -> Optional[Image.Image]:
    if demo:
        candidates = [
            Path("assets/demo_microstructure.png"),
            Path("./assets/demo_microstructure.png"),
            Path("data/demo_microstructure.png"),
            Path("./data/demo_microstructure.png"),
        ]

        for p in candidates:
            if p.exists():
                return Image.open(p).convert("RGB")

        return make_demo_microstructure()

    if uploaded_file is not None:
        return load_image(uploaded_file)

    return None


def normalize_0_1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)

    return (arr - lo) / (hi - lo)


def bragg_d_spacing(two_theta_deg: float, wavelength_A: float = CU_K_ALPHA_A) -> float:
    theta = math.radians(two_theta_deg / 2.0)

    if theta <= 0:
        return float("nan")

    return wavelength_A / (2.0 * math.sin(theta))


def two_theta_from_d(d_A: float, wavelength_A: float = CU_K_ALPHA_A) -> float:
    if d_A <= 0:
        return float("nan")

    arg = wavelength_A / (2.0 * d_A)

    if arg <= 0 or arg >= 1:
        return float("nan")

    return math.degrees(2.0 * math.asin(arg))


def cubic_d_spacing(a_A: float, h: int, k: int, l: int) -> float:
    denom = math.sqrt(h * h + k * k + l * l)

    if denom <= 0:
        return float("nan")

    return a_A / denom


def monoclinic_d_spacing(
    a_A: float,
    b_A: float,
    c_A: float,
    beta_deg: float,
    h: int,
    k: int,
    l: int,
) -> float:
    """
    Monoclinic d-spacing, unique axis b.
    """
    beta = math.radians(beta_deg)
    sin2 = math.sin(beta) ** 2

    if sin2 <= 0:
        return float("nan")

    inv_d2 = (
        (h * h / (a_A * a_A))
        + (k * k * sin2 / (b_A * b_A))
        + (l * l / (c_A * c_A))
        - (2.0 * h * l * math.cos(beta) / (a_A * c_A))
    ) / sin2

    if inv_d2 <= 0:
        return float("nan")

    return 1.0 / math.sqrt(inv_d2)


def expected_niti_refs(
    b2_a_A: float,
    b19_a_A: float,
    b19_b_A: float,
    b19_c_A: float,
    b19_beta_deg: float,
    wavelength_A: float = CU_K_ALPHA_A,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    b2_hkls = [
        (1, 1, 0),
        (2, 0, 0),
        (2, 1, 1),
        (2, 2, 0),
        (3, 1, 0),
        (2, 2, 2),
    ]

    for h, k, l in b2_hkls:
        d = cubic_d_spacing(b2_a_A, h, k, l)
        tt = two_theta_from_d(d, wavelength_A)

        if np.isfinite(tt):
            rows.append(
                {
                    "phase": "B2 austenite",
                    "hkl": f"({h}{k}{l})",
                    "reference_two_theta_deg": tt,
                    "d_A": d,
                    "note": "computed from cubic B2 lattice parameter",
                }
            )

    b19_hkls = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (1, 1, 1),
        (0, 2, 0),
        (0, 0, 2),
        (2, 0, 0),
        (0, 1, 2),
        (2, 1, 1),
        (0, 2, 2),
        (2, 0, 2),
        (2, 2, 0),
    ]

    for h, k, l in b19_hkls:
        d = monoclinic_d_spacing(b19_a_A, b19_b_A, b19_c_A, b19_beta_deg, h, k, l)
        tt = two_theta_from_d(d, wavelength_A)

        if np.isfinite(tt) and 20.0 <= tt <= 100.0:
            rows.append(
                {
                    "phase": "B19′ martensite",
                    "hkl": f"({h}{k}{l})",
                    "reference_two_theta_deg": tt,
                    "d_A": d,
                    "note": "computed from monoclinic B19′ lattice parameters",
                }
            )

    secondary_refs = [
        ("Ni4Ti3 / Ni-rich precipitate", "warning", 43.5, "possible Ni-rich aging/precipitation"),
        ("Ti2Ni / Ti-rich intermetallic", "warning", 39.2, "possible Ti-rich secondary phase"),
        ("Ti2Ni / Ti-rich intermetallic", "warning", 42.2, "possible Ti-rich secondary phase"),
        ("TiO2 / oxide", "warning", 27.4, "oxide contamination indicator"),
        ("TiO2 / oxide", "warning", 36.1, "oxide contamination indicator"),
        ("secondary phase check", "warning", 44.5, "possible secondary-phase overlap"),
    ]

    for phase, hkl, tt, note in secondary_refs:
        rows.append(
            {
                "phase": phase,
                "hkl": hkl,
                "reference_two_theta_deg": tt,
                "d_A": bragg_d_spacing(tt, wavelength_A),
                "note": note,
            }
        )

    return pd.DataFrame(rows).sort_values("reference_two_theta_deg").reset_index(drop=True)


def prepare_xrd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    cols = {str(c).lower().strip(): c for c in data.columns}

    tt_col = None
    intensity_col = None

    for key, raw in cols.items():
        if key in ["two_theta_deg", "twotheta", "two_theta", "2theta", "2theta_deg", "angle", "x"]:
            tt_col = raw
        if key in ["intensity", "counts", "y", "i", "intensity_counts"]:
            intensity_col = raw

    if tt_col is None or intensity_col is None:
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]

        if len(numeric_cols) >= 2:
            tt_col = numeric_cols[0]
            intensity_col = numeric_cols[1]
        else:
            raise ValueError("XRD CSV needs two numeric columns: two_theta_deg and intensity.")

    out = data[[tt_col, intensity_col]].copy()
    out.columns = ["two_theta_deg", "intensity"]
    out = out.dropna().sort_values("two_theta_deg").reset_index(drop=True)
    out["two_theta_deg"] = out["two_theta_deg"].astype(float)
    out["intensity"] = out["intensity"].astype(float)

    return out


def smooth_signal(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window))

    if window % 2 == 0:
        window += 1

    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def estimate_baseline(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(11, window))

    if window % 2 == 0:
        window += 1

    s = pd.Series(y)

    try:
        base = s.rolling(window=window, center=True, min_periods=1).quantile(0.10)
    except Exception:
        base = s.rolling(window=window, center=True, min_periods=1).min()

    return smooth_signal(base.to_numpy(), max(5, window // 5))


def pick_xrd_peaks(
    df: pd.DataFrame,
    smooth_window: int,
    baseline_window: int,
    min_prominence_pct: float,
    min_spacing_deg: float,
    max_peaks: int,
    wavelength_A: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df["two_theta_deg"].to_numpy(dtype=float)
    y_raw = df["intensity"].to_numpy(dtype=float)

    y_smooth = smooth_signal(y_raw, smooth_window)
    baseline = estimate_baseline(y_smooth, baseline_window)
    y_corr = np.clip(y_smooth - baseline, 0, None)
    y_norm = normalize_0_1(y_corr) * 100.0

    processed = df.copy()
    processed["smoothed_intensity"] = y_smooth
    processed["baseline"] = baseline
    processed["corrected_intensity"] = y_corr
    processed["normalized_intensity_pct"] = y_norm

    candidates = []

    for i in range(1, len(x) - 1):
        is_peak = y_norm[i] >= y_norm[i - 1] and y_norm[i] >= y_norm[i + 1]
        is_strong = y_norm[i] >= min_prominence_pct

        if is_peak and is_strong:
            candidates.append(i)

    candidates = sorted(candidates, key=lambda idx: y_norm[idx], reverse=True)

    selected = []

    for idx in candidates:
        if all(abs(x[idx] - x[j]) >= min_spacing_deg for j in selected):
            selected.append(idx)

        if len(selected) >= max_peaks:
            break

    selected = sorted(selected, key=lambda idx: x[idx])

    peak_rows = []

    for idx in selected:
        half = y_corr[idx] / 2.0
        left = idx
        right = idx

        while left > 0 and y_corr[left] > half:
            left -= 1

        while right < len(y_corr) - 1 and y_corr[right] > half:
            right += 1

        fwhm = x[right] - x[left] if right > left else np.nan
        d_A = bragg_d_spacing(x[idx], wavelength_A)

        peak_rows.append(
            {
                "two_theta_deg": x[idx],
                "d_A": d_A,
                "raw_intensity": y_raw[idx],
                "corrected_intensity": y_corr[idx],
                "relative_intensity_pct": y_norm[idx],
                "FWHM_deg_screening": fwhm,
            }
        )

    return processed, pd.DataFrame(peak_rows)


def match_xrd_peaks(
    peaks: pd.DataFrame,
    refs: pd.DataFrame,
    tolerance_deg: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if peaks.empty or refs.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for _, ref in refs.iterrows():
        diffs = np.abs(peaks["two_theta_deg"].to_numpy() - float(ref["reference_two_theta_deg"]))
        idx = int(np.argmin(diffs))
        best_diff = float(diffs[idx])

        if best_diff <= tolerance_deg:
            pk = peaks.iloc[idx]

            rows.append(
                {
                    "phase": ref["phase"],
                    "hkl": ref["hkl"],
                    "reference_two_theta_deg": ref["reference_two_theta_deg"],
                    "matched_two_theta_deg": pk["two_theta_deg"],
                    "delta_deg": pk["two_theta_deg"] - ref["reference_two_theta_deg"],
                    "matched_d_A": pk["d_A"],
                    "relative_intensity_pct": pk["relative_intensity_pct"],
                    "note": ref["note"],
                }
            )

    matches = pd.DataFrame(rows)

    if matches.empty:
        return matches, pd.DataFrame()

    score_rows = []

    for phase, g in matches.groupby("phase"):
        score_rows.append(
            {
                "phase": phase,
                "matched_peaks": len(g),
                "intensity_sum_pct": float(g["relative_intensity_pct"].sum()),
                "mean_abs_delta_deg": float(g["delta_deg"].abs().mean()),
                "screening_score": float(len(g) * 25.0 + g["relative_intensity_pct"].sum() * 0.5),
            }
        )

    scores = pd.DataFrame(score_rows).sort_values("screening_score", ascending=False)

    return matches.sort_values(["phase", "reference_two_theta_deg"]), scores


def b19_basis(a_A: float, b_A: float, c_A: float, beta_deg: float) -> np.ndarray:
    beta = math.radians(beta_deg)

    return np.array(
        [
            [a_A, 0.0, c_A * math.cos(beta)],
            [0.0, b_A, 0.0],
            [0.0, 0.0, c_A * math.sin(beta)],
        ],
        dtype=float,
    )


def transformation_stretch_proxy(
    b2_a_A: float,
    b19_a_A: float,
    b19_b_A: float,
    b19_c_A: float,
    beta_deg: float,
) -> Dict[str, float]:
    """
    Simplified B2→B19′ correspondence proxy.

    This is not full PTMC. It is a useful screening metric.
    """
    parent_ref = np.diag(
        [
            b2_a_A,
            math.sqrt(2.0) * b2_a_A,
            math.sqrt(2.0) * b2_a_A,
        ]
    )

    mart = b19_basis(b19_a_A, b19_b_A, b19_c_A, beta_deg)

    F = mart @ np.linalg.inv(parent_ref)
    C = F.T @ F

    stretches = np.sqrt(np.maximum(np.linalg.eigvalsh(C), 0.0))
    stretches = np.sort(stretches)

    volume_ratio = float(np.linalg.det(F))
    lambda2_error = float(abs(stretches[1] - 1.0))
    max_strain = float(np.max(np.abs(stretches - 1.0)) * 100.0)

    compatibility_score = max(
        0.0,
        100.0 - 5000.0 * lambda2_error - 250.0 * abs(volume_ratio - 1.0),
    )

    return {
        "lambda1": float(stretches[0]),
        "lambda2": float(stretches[1]),
        "lambda3": float(stretches[2]),
        "abs_lambda2_minus_1": lambda2_error,
        "volume_ratio_detF": volume_ratio,
        "volume_change_pct": (volume_ratio - 1.0) * 100.0,
        "max_principal_strain_pct": max_strain,
        "compatibility_score_0_100": compatibility_score,
    }


def bunge_matrix(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> np.ndarray:
    phi1 = math.radians(float(phi1_deg))
    Phi = math.radians(float(Phi_deg))
    phi2 = math.radians(float(phi2_deg))

    c1, s1 = math.cos(phi1), math.sin(phi1)
    c, s = math.cos(Phi), math.sin(Phi)
    c2, s2 = math.cos(phi2), math.sin(phi2)

    return np.array(
        [
            [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
            [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
            [s1 * s, -c1 * s, c],
        ],
        dtype=float,
    )


def misorientation_angle_deg(g1: np.ndarray, g2: np.ndarray) -> float:
    delta = g1 @ g2.T
    val = (np.trace(delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))

    return math.degrees(math.acos(val))


def add_ebsd_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    euler_cols = None

    for candidate in [
        ("phi1", "Phi", "phi2"),
        ("phi1", "phi", "phi2"),
        ("euler1", "euler2", "euler3"),
    ]:
        if all(c in data.columns for c in candidate):
            euler_cols = candidate
            break

    if euler_cols is None:
        return data

    matrices = []
    rgb = []
    sample_z = np.array([0.0, 0.0, 1.0])

    for _, row in data.iterrows():
        g = bunge_matrix(row[euler_cols[0]], row[euler_cols[1]], row[euler_cols[2]])
        matrices.append(g)

        direction = np.abs(g @ sample_z)
        norm = np.linalg.norm(direction)

        if norm > 0:
            direction = direction / norm

        rgb.append(direction)

    rgb = np.asarray(rgb)

    data["ipf_r"] = rgb[:, 0]
    data["ipf_g"] = rgb[:, 1]
    data["ipf_b"] = rgb[:, 2]

    if "phi1" in data.columns:
        data["orientation_family_proxy"] = (
            np.floor((pd.to_numeric(data["phi1"], errors="coerce") % 360.0) / 60.0)
            .fillna(-1)
            .astype(int)
            + 1
        ).astype(str)

    if {"x", "y"}.issubset(data.columns) and len(data) <= 30000:
        xvals = np.sort(data["x"].dropna().unique())
        yvals = np.sort(data["y"].dropna().unique())

        if len(xvals) > 1 and len(yvals) > 1:
            idx_by_xy = {(row["x"], row["y"]): i for i, row in data[["x", "y"]].iterrows()}
            x_to_pos = {x: i for i, x in enumerate(xvals)}
            y_to_pos = {y: i for i, y in enumerate(yvals)}

            kam = np.full(len(data), np.nan)

            for i, row in data[["x", "y"]].iterrows():
                xi = x_to_pos.get(row["x"])
                yi = y_to_pos.get(row["y"])

                if xi is None or yi is None:
                    continue

                vals = []

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xx = xi + dx
                    yy = yi + dy

                    if 0 <= xx < len(xvals) and 0 <= yy < len(yvals):
                        j = idx_by_xy.get((xvals[xx], yvals[yy]))

                        if j is not None:
                            vals.append(misorientation_angle_deg(matrices[i], matrices[j]))

                if vals:
                    kam[i] = float(np.nanmean(vals))

            data["kam_deg_no_symmetry"] = kam

    return data


def make_ipf_image(df: pd.DataFrame) -> Optional[np.ndarray]:
    required = {"x", "y", "ipf_r", "ipf_g", "ipf_b"}

    if not required.issubset(set(df.columns)):
        return None

    xvals = np.sort(df["x"].dropna().unique())
    yvals = np.sort(df["y"].dropna().unique())

    if len(xvals) < 2 or len(yvals) < 2:
        return None

    xmap = {v: i for i, v in enumerate(xvals)}
    ymap = {v: i for i, v in enumerate(yvals)}

    img = np.ones((len(yvals), len(xvals), 3), dtype=float)

    for _, row in df.iterrows():
        xi = xmap.get(row["x"])
        yi = ymap.get(row["y"])

        if xi is not None and yi is not None:
            img[yi, xi, 0] = row["ipf_r"]
            img[yi, xi, 1] = row["ipf_g"]
            img[yi, xi, 2] = row["ipf_b"]

    return np.flipud(img)


def plot_grid_map(df: pd.DataFrame, value: str, title: str) -> None:
    pivot = grid_for_plot(df, value)

    if pivot is None:
        st.info("Could not reconstruct a regular x-y grid for this map.")
        return

    if not pd.api.types.is_numeric_dtype(pivot.to_numpy().ravel()):
        flat = pivot.to_numpy().ravel()
        codes, uniques = pd.factorize(flat)
        code_img = codes.reshape(pivot.shape)

        fig = px.imshow(code_img, aspect="equal", title=title)
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)

        legend_df = pd.DataFrame(
            {
                "code": list(range(len(uniques))),
                "label": [str(u) for u in uniques],
            }
        )
        st.dataframe(legend_df, use_container_width=True, hide_index=True)
    else:
        fig = px.imshow(pivot, aspect="equal", title=title)
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Tabs
# =============================================================================

tab_img, tab_xrd, tab_crystal, tab_ebsd, tab_report = st.tabs(
    [
        "SEM / optical defects",
        "XRD phase screening",
        "B2 ↔ B19′ crystallography",
        "EBSD / TKD orientation",
        "Evidence report",
    ]
)


# =============================================================================
# SEM / optical defects
# =============================================================================

with tab_img:
    st.subheader("Defect and pore/crack screening")

    st.markdown(
        """
Use this for polished SEM/optical images where dark or bright features may represent pores, cracks, pull-outs, etching contrast or polishing artifacts.
"""
    )

    col_settings, col_note = st.columns([0.9, 1.1])

    with col_settings:
        demo_img = st.checkbox("Use demo microstructure image", value=True, key="img_use_demo")
        uploaded_img = None if demo_img else st.file_uploader(
            "Upload SEM/optical image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="img_file_upload",
        )

        invert = st.checkbox("Dark features are pores/cracks", value=True, key="img_invert")
        threshold = st.slider("Manual threshold; 0 = Otsu", 0, 255, 0, key="img_threshold")
        pixel_size = st.number_input(
            "Pixel size (µm/pixel; 0 = unknown)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="img_pixel_size",
        )

    with col_note:
        st.info(
            "If the demo image was not showing before, it was probably because "
            "`assets/demo_microstructure.png` was missing. This version automatically "
            "creates a synthetic fallback image."
        )

    img = load_demo_or_uploaded_image(demo_img, uploaded_img)

    if img is not None:
        result = porosity_screen(
            img,
            invert=invert,
            manual_threshold=None if threshold == 0 else threshold,
            pixel_size_um=pixel_size if pixel_size > 0 else None,
        )

        texture = image_texture_descriptors(img)

        c1, c2 = st.columns(2)
        c1.image(img, caption="Input image", use_container_width=True)

        if "mask" in result:
            c2.image(result["mask"], caption="Segmentation mask", use_container_width=True)
        else:
            c2.warning("Segmentation mask was not returned by porosity_screen().")

        area_pct = float(result.get("feature_area_fraction_pct", 0.0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Feature area fraction", f"{area_pct:.3f}%")
        m2.metric("Feature count", str(result.get("feature_count", "n/a")))
        m3.metric("Mean feature area", f"{result.get('mean_feature_area_um2', 0):.2f} µm²")
        m4.metric("Gray-level std", f"{texture.get('std_gray', 0):.1f}")

        with st.expander("Detailed image descriptors"):
            st.json({k: v for k, v in result.items() if k != "mask"})
            st.json(texture)

        if area_pct < 0.25:
            st.success("Interpretation: very low segmented defect fraction. Verify contrast before claiming high density.")
        elif area_pct < 1.5:
            st.info("Interpretation: moderate segmented feature fraction. Check whether these are real pores/cracks or artifacts.")
        else:
            st.warning("Interpretation: high segmented feature fraction. Check for lack-of-fusion, keyhole pores, cracks or preparation artifacts.")
    else:
        st.warning("No image loaded. Upload an image or use the demo option.")


# =============================================================================
# XRD phase screening
# =============================================================================

with tab_xrd:
    st.subheader("XRD phase screening with NiTi peak matching")

    st.markdown(
        """
Upload a CSV with two columns: `two_theta_deg` and `intensity`.
The tool performs baseline correction, peak detection, d-spacing conversion, B2/B19′ matching and secondary-phase warnings.
"""
    )

    sx1, sx2, sx3 = st.columns(3)

    with sx1:
        demo_xrd = st.checkbox("Use demo XRD CSV", value=True, key="xrd_use_demo")
        xrd_file = None if demo_xrd else st.file_uploader("Upload XRD CSV", type=["csv"], key="xrd_file_upload")
        wavelength_xrd = st.number_input(
            "X-ray wavelength λ (Å)",
            min_value=0.5,
            max_value=3.0,
            value=CU_K_ALPHA_A,
            step=0.0001,
            format="%.4f",
            key="xrd_wavelength",
        )

    with sx2:
        smooth_win = st.slider("Smoothing window", 3, 31, 7, 2, key="xrd_smooth_window")
        baseline_win = st.slider("Baseline window", 21, 501, 101, 20, key="xrd_baseline_window")
        min_prom = st.slider("Minimum peak prominence (%)", 1.0, 30.0, 5.0, 0.5, key="xrd_min_prominence")

    with sx3:
        min_spacing = st.slider("Minimum peak spacing (°2θ)", 0.05, 1.50, 0.25, 0.05, key="xrd_min_spacing")
        match_tol = st.slider("Phase-match tolerance (°2θ)", 0.05, 1.50, 0.35, 0.05, key="xrd_match_tolerance")
        max_peaks = st.slider("Maximum peaks", 5, 100, 40, 5, key="xrd_max_peaks")

    with st.expander("Reference lattice parameters for XRD matching", expanded=False):
        px1, px2, px3, px4, px5 = st.columns(5)

        xrd_b2_a = px1.number_input("B2 a (Å)", value=3.0150, step=0.001, format="%.4f", key="xrd_b2_a")
        xrd_b19_a = px2.number_input("B19′ a (Å)", value=2.8890, step=0.001, format="%.4f", key="xrd_b19_a")
        xrd_b19_b = px3.number_input("B19′ b (Å)", value=4.1200, step=0.001, format="%.4f", key="xrd_b19_b")
        xrd_b19_c = px4.number_input("B19′ c (Å)", value=4.6220, step=0.001, format="%.4f", key="xrd_b19_c")
        xrd_b19_beta = px5.number_input("B19′ β (deg)", value=96.80, step=0.01, format="%.2f", key="xrd_b19_beta")

    xrd_ready = False

    if demo_xrd or xrd_file is not None:
        try:
            if demo_xrd:
                demo_path = Path("data/demo_xrd.csv")
                if not demo_path.exists():
                    st.warning("Demo XRD file data/demo_xrd.csv was not found. Upload a CSV instead.")
                    raw_xrd = None
                else:
                    raw_xrd = read_xrd_csv(str(demo_path))
            else:
                raw_xrd = read_xrd_csv(xrd_file)

            if raw_xrd is not None:
                xrd = prepare_xrd_dataframe(raw_xrd)

                processed_xrd, peaks = pick_xrd_peaks(
                    xrd,
                    smooth_window=smooth_win,
                    baseline_window=baseline_win,
                    min_prominence_pct=min_prom,
                    min_spacing_deg=min_spacing,
                    max_peaks=max_peaks,
                    wavelength_A=wavelength_xrd,
                )

                refs = expected_niti_refs(
                    xrd_b2_a,
                    xrd_b19_a,
                    xrd_b19_b,
                    xrd_b19_c,
                    xrd_b19_beta,
                    wavelength_xrd,
                )

                matches, phase_scores = match_xrd_peaks(peaks, refs, match_tol)
                xrd_ready = True

        except Exception as exc:
            st.error(f"Could not read/process XRD file: {exc}")

    if xrd_ready:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=processed_xrd["two_theta_deg"],
                y=processed_xrd["intensity"],
                mode="lines",
                name="raw intensity",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=processed_xrd["two_theta_deg"],
                y=processed_xrd["baseline"],
                mode="lines",
                name="baseline",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=processed_xrd["two_theta_deg"],
                y=processed_xrd["corrected_intensity"],
                mode="lines",
                name="baseline-corrected",
            )
        )

        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["two_theta_deg"],
                    y=peaks["corrected_intensity"],
                    mode="markers",
                    name="detected peaks",
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            height=520,
            xaxis_title="2θ (deg)",
            yaxis_title="Intensity",
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Detected peaks")
            st.dataframe(peaks, use_container_width=True, hide_index=True)

        with c2:
            st.subheader("Phase screening score")
            st.dataframe(phase_scores, use_container_width=True, hide_index=True)

            if not phase_scores.empty:
                top_phase = str(phase_scores.iloc[0]["phase"])

                if "B2" in top_phase:
                    st.success("Top XRD match: B2/austenite-like. Good sign for superelastic design, but DSC is still required.")
                elif "B19" in top_phase:
                    st.warning("Top XRD match: B19′/martensite-like. This can explain weak room-temperature superelasticity.")
                else:
                    st.warning("Top XRD match is a secondary/warning phase. Verify with refinement and microscopy.")

        st.subheader("Matched reference peaks")
        st.dataframe(matches, use_container_width=True, hide_index=True)

        with st.expander("Full generated reference peak table"):
            st.dataframe(refs, use_container_width=True, hide_index=True)


# =============================================================================
# Crystallography tab
# =============================================================================

with tab_crystal:
    st.subheader("B2 ↔ B19′ crystallography and compatibility metrics")

    st.markdown(
        """
This tab is for the crystallography argument. It calculates B2/B19′ d-spacings, a simplified transformation stretch, λ₂ compatibility proxy, volume change and strain indicators.

λ₂ close to 1 is generally a better compatibility sign. This is still a simplified proxy, not full PTMC.
"""
    )

    cc1, cc2 = st.columns([1.0, 1.0])

    with cc1:
        st.markdown("**Parent B2 / austenite**")
        cry_b2_a = st.number_input(
            "B2 a₀ (Å)",
            value=3.0150,
            step=0.0005,
            format="%.4f",
            key="cry_b2_a",
        )

        st.markdown("**Martensite B19′**")
        cry_b19_a = st.number_input(
            "B19′ a_m (Å)",
            value=2.8890,
            step=0.0005,
            format="%.4f",
            key="cry_b19_a",
        )
        cry_b19_b = st.number_input(
            "B19′ b_m (Å)",
            value=4.1200,
            step=0.0005,
            format="%.4f",
            key="cry_b19_b",
        )
        cry_b19_c = st.number_input(
            "B19′ c_m (Å)",
            value=4.6220,
            step=0.0005,
            format="%.4f",
            key="cry_b19_c",
        )
        cry_b19_beta = st.number_input(
            "B19′ β_m (deg)",
            value=96.80,
            step=0.01,
            format="%.2f",
            key="cry_b19_beta",
        )

    with cc2:
        metrics = transformation_stretch_proxy(
            cry_b2_a,
            cry_b19_a,
            cry_b19_b,
            cry_b19_c,
            cry_b19_beta,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("λ₁", f"{metrics['lambda1']:.5f}")
        m2.metric("λ₂", f"{metrics['lambda2']:.5f}")
        m3.metric("λ₃", f"{metrics['lambda3']:.5f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("|λ₂ − 1|", f"{metrics['abs_lambda2_minus_1']:.5f}")
        m5.metric("Volume change", f"{metrics['volume_change_pct']:+.2f}%")
        m6.metric("Compatibility score", f"{metrics['compatibility_score_0_100']:.1f}/100")

        if metrics["abs_lambda2_minus_1"] < 0.01:
            st.success("λ₂ is close to 1: good compatibility proxy for recoverable transformation.")
        elif metrics["abs_lambda2_minus_1"] < 0.03:
            st.info("λ₂ is moderately close to 1: possible compatibility, but sensitive to composition and heat treatment.")
        else:
            st.warning("λ₂ is far from 1: higher crystallographic mismatch risk. Validate with DSC and cyclic testing.")

    st.subheader("Transformation-stretch proxy table")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)

    st.subheader("d-spacing and 2θ calculator")

    dc1, dc2, dc3, dc4, dc5 = st.columns(5)

    calc_phase = dc1.selectbox(
        "Phase",
        ["B2 cubic", "B19′ monoclinic"],
        key="cry_calc_phase",
    )
    calc_h = dc2.number_input("h", value=1, step=1, key="cry_calc_h")
    calc_k = dc3.number_input("k", value=1, step=1, key="cry_calc_k")
    calc_l = dc4.number_input("l", value=0, step=1, key="cry_calc_l")
    calc_wavelength = dc5.number_input(
        "λ (Å)",
        value=CU_K_ALPHA_A,
        step=0.0001,
        format="%.4f",
        key="cry_calc_wavelength",
    )

    if calc_phase == "B2 cubic":
        calc_d = cubic_d_spacing(cry_b2_a, int(calc_h), int(calc_k), int(calc_l))
    else:
        calc_d = monoclinic_d_spacing(
            cry_b19_a,
            cry_b19_b,
            cry_b19_c,
            cry_b19_beta,
            int(calc_h),
            int(calc_k),
            int(calc_l),
        )

    calc_tt = two_theta_from_d(calc_d, calc_wavelength)

    d1, d2 = st.columns(2)
    d1.metric("d-spacing", f"{calc_d:.4f} Å" if np.isfinite(calc_d) else "invalid")
    d2.metric("Calculated 2θ", f"{calc_tt:.3f}°" if np.isfinite(calc_tt) else "invalid")

    st.subheader("Generated NiTi reference peaks from current lattice parameters")

    cry_refs = expected_niti_refs(
        cry_b2_a,
        cry_b19_a,
        cry_b19_b,
        cry_b19_c,
        cry_b19_beta,
        calc_wavelength,
    )

    st.dataframe(cry_refs, use_container_width=True, hide_index=True)


# =============================================================================
# EBSD / TKD orientation tab
# =============================================================================

with tab_ebsd:
    st.subheader("EBSD/TKD orientation and local crystallography screening")

    st.markdown(
        """
Upload a CSV containing at least `x`, `y`, and Euler angles: `phi1`, `Phi`, `phi2`.

Optional columns: `phase`, `ci`, `iq`, `grain_id`.
"""
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        demo_ebsd = st.checkbox("Use demo EBSD CSV", value=True, key="ebsd_use_demo")
        ebsd_file = None if demo_ebsd else st.file_uploader("Upload EBSD/TKD CSV", type=["csv"], key="ebsd_file_upload")

    with e2:
        ci_min = st.number_input("Minimum CI filter", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="ebsd_ci_min")

    with e3:
        iq_min = st.number_input("Minimum IQ filter", min_value=0.0, value=0.0, step=1.0, key="ebsd_iq_min")

    ebsd_ready = False

    if demo_ebsd or ebsd_file is not None:
        try:
            if demo_ebsd:
                demo_path = Path("data/demo_ebsd.csv")
                if not demo_path.exists():
                    st.warning("Demo EBSD file data/demo_ebsd.csv was not found. Upload a CSV instead.")
                    ebsd_raw = None
                else:
                    ebsd_raw = read_ebsd_csv(str(demo_path))
            else:
                ebsd_raw = read_ebsd_csv(ebsd_file)

            if ebsd_raw is not None:
                ebsd = ebsd_raw.copy()

                if "ci" in ebsd.columns and ci_min > 0:
                    ebsd = ebsd[ebsd["ci"] >= ci_min]

                if "iq" in ebsd.columns and iq_min > 0:
                    ebsd = ebsd[ebsd["iq"] >= iq_min]

                ebsd = add_ebsd_descriptors(ebsd)
                summary = ebsd_summary(ebsd)
                ebsd_ready = True

        except Exception as exc:
            st.error(f"Could not read/process EBSD/TKD file: {exc}")

    if ebsd_ready:
        st.subheader("Data preview")
        st.dataframe(ebsd.head(50), use_container_width=True)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Indexed points", f"{len(ebsd):,}")

        if "phase" in ebsd.columns:
            q2.metric("Phases", f"{ebsd['phase'].nunique()}")

        if "grain_id" in ebsd.columns:
            q3.metric("Grains", f"{ebsd['grain_id'].nunique()}")

        if "kam_deg_no_symmetry" in ebsd.columns:
            q4.metric("Mean KAM proxy", f"{ebsd['kam_deg_no_symmetry'].mean():.2f}°")

        if "phase" in ebsd.columns:
            phase_frac = (
                ebsd["phase"]
                .value_counts(normalize=True)
                .rename_axis("phase")
                .reset_index(name="fraction")
            )
            phase_frac["fraction_pct"] = 100.0 * phase_frac["fraction"]

            st.subheader("Indexed phase fraction")
            st.dataframe(phase_frac[["phase", "fraction_pct"]], use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.bar(phase_frac, x="phase", y="fraction_pct", title="Indexed phase fraction"),
                use_container_width=True,
            )

        map_options = [
            c
            for c in [
                "phase",
                "orientation_family_proxy",
                "kam_deg_no_symmetry",
                "ci",
                "iq",
                "phi1",
                "Phi",
                "phi2",
                "grain_id",
            ]
            if c in ebsd.columns
        ]

        if map_options:
            map_value = st.selectbox("Map value", map_options, key="ebsd_map_value")
            plot_grid_map(ebsd, map_value, f"Map: {map_value}")

        if {"ipf_r", "ipf_g", "ipf_b"}.issubset(ebsd.columns):
            st.subheader("IPF-Z color proxy")

            ipf_img = make_ipf_image(ebsd)

            if ipf_img is not None:
                fig_ipf = go.Figure(go.Image(z=(np.clip(ipf_img, 0, 1) * 255).astype(np.uint8)))
                fig_ipf.update_layout(height=550, title="IPF-Z proxy image")
                st.plotly_chart(fig_ipf, use_container_width=True)
            else:
                st.info("IPF proxy values calculated, but x-y grid could not be reconstructed.")

        if "kam_deg_no_symmetry" in ebsd.columns:
            st.subheader("KAM / orientation-gradient proxy")

            st.plotly_chart(
                px.histogram(
                    ebsd,
                    x="kam_deg_no_symmetry",
                    nbins=60,
                    title="KAM proxy distribution",
                ),
                use_container_width=True,
            )

            high_kam = float((ebsd["kam_deg_no_symmetry"] > 5.0).mean() * 100.0)

            if high_kam > 20:
                st.warning(f"{high_kam:.1f}% of points have KAM proxy > 5°. Possible strain, substructure, poor indexing or local distortion.")
            else:
                st.success(f"{high_kam:.1f}% of points have KAM proxy > 5°. Orientation-gradient level is not severe by this proxy.")

        with st.expander("Raw EBSD summary"):
            st.json(summary)

        st.info(
            "The orientation-family map is not exact B19′ variant indexing. Exact variant indexing needs parent B2 reconstruction, symmetry operators and an orientation relationship."
        )


# =============================================================================
# Evidence report tab
# =============================================================================

with tab_report:
    st.subheader("How to use the evidence in the NiTi project")

    st.markdown(
        """
Use the outputs in this order:

1. **Microstructure**: check pores, cracks, lack-of-fusion, keyhole defects or artifacts.
2. **XRD**: check whether the phase state is B2, B19′, mixed, oxide-rich or secondary-phase-rich.
3. **Crystallography**: use λ₂, volume change and principal strain as compatibility evidence.
4. **EBSD/TKD**: check phase distribution, texture, orientation gradients and grain-level evidence.
5. **Process-function link**: compare all of this with vaporization-induced Ni loss, effective Ni at.%, Af/Ms shift and DSC.
"""
    )

    st.subheader("Copy-ready interpretation")

    st.code(
        """
The characterization workflow combines defect segmentation, XRD phase screening,
B2/B19′ crystallography metrics and EBSD/TKD orientation mapping. The image analysis
checks whether the process window produced excessive pores, cracks or lack-of-fusion
features. XRD peak matching tests whether the room-temperature phase state is consistent
with B2/austenite, B19′/martensite or secondary phases. The crystallography module estimates
B2→B19′ compatibility using λ2, volume change and principal strain. EBSD/TKD maps provide
phase fraction, texture, local orientation-gradient and grain-scale evidence. Together, these
outputs connect LPBF processing and vaporization-induced composition shift to the final
functional NiTi state.
""".strip(),
        language="text",
    )

    st.subheader("Needed validation")

    st.markdown(
        """
- DSC for Ms, Mf, As and Af.
- ICP/EDS for powder vs printed-part Ni/Ti composition.
- Rietveld refinement for phase fraction and lattice parameters.
- SEM cross-section validation for pore/crack morphology.
- EBSD/TKD parent reconstruction and exact B19′ variant indexing.
- Cyclic superelastic or shape-memory testing.
"""
    )
