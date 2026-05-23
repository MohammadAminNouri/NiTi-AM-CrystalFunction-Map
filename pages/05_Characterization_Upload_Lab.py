from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.image_analysis import load_image, porosity_screen, image_texture_descriptors
from src.xrd_tools import read_xrd_csv
from src.ebsd_tools import read_ebsd_csv, ebsd_summary, grid_for_plot


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Characterization + Crystallography Lab", layout="wide")

st.title("Characterization + Crystallography Lab")
st.caption(
    "Evidence-driven microstructure, XRD, EBSD/TKD and B2→B19′ crystallography screening for LPBF NiTi."
)

st.markdown(
    """
This page connects uploaded characterization evidence to the real NiTi function question:

> Is this printed NiTi mainly B2/austenite, B19′/martensite, mixed, defective, textured, or crystallographically compatible with recoverable transformation?

It is a **screening and reporting tool**, not a replacement for full metallography, Rietveld refinement, commercial EBSD software, DSC, TEM, or mechanical cycling.
"""
)


# =============================================================================
# Shared helper functions
# =============================================================================

CU_K_ALPHA_A = 1.5406


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


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


def monoclinic_d_spacing(a_A: float, b_A: float, c_A: float, beta_deg: float, h: int, k: int, l: int) -> float:
    """
    Monoclinic d-spacing for unique axis b.
    """
    beta = math.radians(beta_deg)
    s2 = math.sin(beta) ** 2
    if s2 <= 0:
        return float("nan")

    inv_d2 = (
        (h * h / (a_A * a_A))
        + (k * k * s2 / (b_A * b_A))
        + (l * l / (c_A * c_A))
        - (2.0 * h * l * math.cos(beta) / (a_A * c_A))
    ) / s2

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
    """
    Generate a practical NiTi screening reference list.
    B2 and B19′ are calculated from lattice parameters.
    Secondary phases are included as warning windows.
    """
    rows: List[Dict[str, object]] = []

    b2_hkls = [
        (1, 1, 0, "B2 austenite"),
        (2, 0, 0, "B2 austenite"),
        (2, 1, 1, "B2 austenite"),
        (2, 2, 0, "B2 austenite"),
        (3, 1, 0, "B2 austenite"),
        (2, 2, 2, "B2 austenite"),
    ]

    for h, k, l, phase in b2_hkls:
        d = cubic_d_spacing(b2_a_A, h, k, l)
        tt = two_theta_from_d(d, wavelength_A)
        if np.isfinite(tt):
            rows.append(
                {
                    "phase": phase,
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

    secondary = [
        ("Ni4Ti3 / Ni-rich precipitate", "warning", 43.5, "possible Ni-rich precipitation / aging response"),
        ("Ti2Ni / Ti-rich intermetallic", "warning", 39.2, "possible Ti-rich secondary phase"),
        ("Ti2Ni / Ti-rich intermetallic", "warning", 42.2, "possible Ti-rich secondary phase"),
        ("TiO2 / oxide", "warning", 27.4, "oxide contamination / surface oxidation indicator"),
        ("TiO2 / oxide", "warning", 36.1, "oxide contamination / surface oxidation indicator"),
        ("NiTi2 or other secondary", "warning", 44.5, "secondary phase check required"),
    ]

    for phase, hkl, tt, note in secondary:
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
    """
    Make XRD input tolerant of different column names.
    """
    data = df.copy()
    cols = {c.lower().strip(): c for c in data.columns}

    tt_col = None
    inten_col = None

    for key, raw in cols.items():
        if key in ["two_theta_deg", "twotheta", "two_theta", "2theta", "2theta_deg", "angle", "x"]:
            tt_col = raw
        if key in ["intensity", "counts", "y", "i", "intensity_counts"]:
            inten_col = raw

    if tt_col is None or inten_col is None:
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        if len(numeric_cols) >= 2:
            tt_col, inten_col = numeric_cols[0], numeric_cols[1]
        else:
            raise ValueError("XRD CSV needs two numeric columns: two_theta_deg and intensity.")

    out = data[[tt_col, inten_col]].copy()
    out.columns = ["two_theta_deg", "intensity"]
    out = out.dropna().sort_values("two_theta_deg")
    out["two_theta_deg"] = out["two_theta_deg"].astype(float)
    out["intensity"] = out["intensity"].astype(float)

    return out.reset_index(drop=True)


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
    smooth_window: int = 7,
    baseline_window: int = 101,
    min_prominence_pct: float = 5.0,
    min_spacing_deg: float = 0.25,
    max_peaks: int = 40,
    wavelength_A: float = CU_K_ALPHA_A,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df["two_theta_deg"].to_numpy(dtype=float)
    y_raw = df["intensity"].to_numpy(dtype=float)

    y_s = smooth_signal(y_raw, smooth_window)
    base = estimate_baseline(y_s, baseline_window)
    y_corr = np.clip(y_s - base, 0, None)
    y_norm = normalize_0_1(y_corr) * 100.0

    work = df.copy()
    work["smoothed_intensity"] = y_s
    work["baseline"] = base
    work["corrected_intensity"] = y_corr
    work["normalized_intensity_pct"] = y_norm

    candidates = []

    for i in range(1, len(x) - 1):
        if y_norm[i] >= min_prominence_pct and y_norm[i] >= y_norm[i - 1] and y_norm[i] >= y_norm[i + 1]:
            candidates.append(i)

    candidates = sorted(candidates, key=lambda idx: y_norm[idx], reverse=True)

    selected: List[int] = []

    for idx in candidates:
        if all(abs(x[idx] - x[j]) >= min_spacing_deg for j in selected):
            selected.append(idx)
        if len(selected) >= max_peaks:
            break

    selected = sorted(selected, key=lambda idx: x[idx])

    rows = []

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

        beta_rad = math.radians(fwhm) if np.isfinite(fwhm) else np.nan
        theta_rad = math.radians(x[idx] / 2.0)

        if np.isfinite(beta_rad) and beta_rad > 0 and math.cos(theta_rad) > 0:
            scherrer_nm = 0.9 * wavelength_A / (beta_rad * math.cos(theta_rad)) * 0.1
        else:
            scherrer_nm = np.nan

        rows.append(
            {
                "two_theta_deg": x[idx],
                "d_A": d_A,
                "intensity": y_raw[idx],
                "corrected_intensity": y_corr[idx],
                "relative_intensity_pct": y_norm[idx],
                "FWHM_deg_screening": fwhm,
                "scherrer_size_nm_uncorrected": scherrer_nm,
            }
        )

    return work, pd.DataFrame(rows)


def match_xrd_peaks(peaks: pd.DataFrame, refs: pd.DataFrame, tolerance_deg: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if peaks.empty or refs.empty:
        return pd.DataFrame(), pd.DataFrame()

    match_rows = []

    for _, ref in refs.iterrows():
        diffs = np.abs(peaks["two_theta_deg"].to_numpy() - float(ref["reference_two_theta_deg"]))
        best_idx = int(np.argmin(diffs))
        best_diff = float(diffs[best_idx])

        if best_diff <= tolerance_deg:
            pk = peaks.iloc[best_idx]
            closeness = max(0.0, 1.0 - best_diff / tolerance_deg)

            match_rows.append(
                {
                    "phase": ref["phase"],
                    "hkl": ref["hkl"],
                    "reference_two_theta_deg": ref["reference_two_theta_deg"],
                    "matched_two_theta_deg": pk["two_theta_deg"],
                    "delta_deg": pk["two_theta_deg"] - ref["reference_two_theta_deg"],
                    "matched_d_A": pk["d_A"],
                    "relative_intensity_pct": pk["relative_intensity_pct"],
                    "closeness_0_1": closeness,
                    "note": ref["note"],
                }
            )

    matches = pd.DataFrame(match_rows)

    if matches.empty:
        return matches, pd.DataFrame()

    score_rows = []

    for phase, g in matches.groupby("phase"):
        n = len(g)
        intensity_score = float(g["relative_intensity_pct"].sum())
        closeness_score = float(g["closeness_0_1"].mean())
        combined = n * 25.0 + intensity_score * 0.45 + closeness_score * 30.0

        score_rows.append(
            {
                "phase": phase,
                "matched_peaks": n,
                "intensity_sum_pct": intensity_score,
                "mean_closeness": closeness_score,
                "screening_score": combined,
            }
        )

    scores = pd.DataFrame(score_rows).sort_values("screening_score", ascending=False)

    return matches.sort_values(["phase", "reference_two_theta_deg"]), scores


def b19_lattice_basis(a_A: float, b_A: float, c_A: float, beta_deg: float) -> np.ndarray:
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
    Simplified B2 -> B19′ correspondence proxy.

    Approximate reference B2 correspondence basis:
        a_ref ≈ a_B2
        b_ref ≈ sqrt(2) a_B2
        c_ref ≈ sqrt(2) a_B2

    This is not full PTMC and not exact variant reconstruction.
    It is useful as a compatibility indicator: λ2 close to 1 is generally desirable.
    """
    b_ref = np.diag(
        [
            b2_a_A,
            math.sqrt(2.0) * b2_a_A,
            math.sqrt(2.0) * b2_a_A,
        ]
    )

    b_m = b19_lattice_basis(b19_a_A, b19_b_A, b19_c_A, beta_deg)

    F = b_m @ np.linalg.inv(b_ref)
    C = F.T @ F

    eig = np.linalg.eigvalsh(C)
    stretches = np.sqrt(np.maximum(eig, 0.0))
    stretches = np.sort(stretches)

    volume_ratio = float(np.linalg.det(F))
    max_principal_strain_pct = float(np.max(np.abs(stretches - 1.0)) * 100.0)
    lambda2_error = float(abs(stretches[1] - 1.0))

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
        "max_principal_strain_pct": max_principal_strain_pct,
        "compatibility_score_0_100": compatibility_score,
    }


def bunge_matrix(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> np.ndarray:
    p1 = math.radians(phi1_deg)
    P = math.radians(Phi_deg)
    p2 = math.radians(phi2_deg)

    c1, s1 = math.cos(p1), math.sin(p1)
    c, s = math.cos(P), math.sin(P)
    c2, s2 = math.cos(p2), math.sin(p2)

    return np.array(
        [
            [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
            [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
            [s1 * s, -c1 * s, c],
        ],
        dtype=float,
    )


def misorientation_angle_deg(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    Fast misorientation proxy.
    Crystal symmetry is ignored here.
    """
    delta = g1 @ g2.T
    val = (np.trace(delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))

    return math.degrees(math.acos(val))


def add_orientation_descriptors(df: pd.DataFrame) -> pd.DataFrame:
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

    rgbs = []
    matrices = []
    sample_z = np.array([0.0, 0.0, 1.0])

    for _, row in data.iterrows():
        g = bunge_matrix(
            _safe_float(row[euler_cols[0]]),
            _safe_float(row[euler_cols[1]]),
            _safe_float(row[euler_cols[2]]),
        )

        matrices.append(g)

        crystal_dir = np.abs(g @ sample_z)
        norm = np.linalg.norm(crystal_dir)

        if norm > 0:
            crystal_dir = crystal_dir / norm

        rgbs.append(crystal_dir)

    rgb_arr = np.asarray(rgbs)

    data["ipf_r"] = rgb_arr[:, 0]
    data["ipf_g"] = rgb_arr[:, 1]
    data["ipf_b"] = rgb_arr[:, 2]

    if "x" in data.columns and "y" in data.columns:
        xvals = np.sort(data["x"].dropna().unique())
        yvals = np.sort(data["y"].dropna().unique())

        if len(xvals) > 1 and len(yvals) > 1 and len(data) <= 30000:
            idx_by_xy = {(row["x"], row["y"]): i for i, row in data[["x", "y"]].iterrows()}
            x_to_pos = {x: i for i, x in enumerate(xvals)}
            y_to_pos = {y: i for i, y in enumerate(yvals)}

            kam = np.full(len(data), np.nan)

            for i, row in data[["x", "y"]].iterrows():
                xi = x_to_pos.get(row["x"])
                yi = y_to_pos.get(row["y"])

                if xi is None or yi is None:
                    continue

                neigh_angles = []

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xx_i = xi + dx
                    yy_i = yi + dy

                    if 0 <= xx_i < len(xvals) and 0 <= yy_i < len(yvals):
                        j = idx_by_xy.get((xvals[xx_i], yvals[yy_i]))

                        if j is not None:
                            neigh_angles.append(misorientation_angle_deg(matrices[i], matrices[j]))

                if neigh_angles:
                    kam[i] = float(np.nanmean(neigh_angles))

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


def grain_statistics(df: pd.DataFrame) -> pd.DataFrame:
    if "grain_id" not in df.columns:
        return pd.DataFrame()

    group = df.groupby("grain_id", dropna=True)
    rows = []

    for gid, g in group:
        row = {"grain_id": gid, "points": len(g)}

        if "phase" in g.columns:
            mode = g["phase"].mode()
            row["dominant_phase"] = mode.iloc[0] if len(mode) else "unknown"

        if "kam_deg_no_symmetry" in g.columns:
            row["mean_KAM_deg_proxy"] = float(g["kam_deg_no_symmetry"].mean())

        if "ci" in g.columns:
            row["mean_CI"] = float(g["ci"].mean())

        if "iq" in g.columns:
            row["mean_IQ"] = float(g["iq"].mean())

        rows.append(row)

    return pd.DataFrame(rows).sort_values("points", ascending=False)


def orientation_family_bins(df: pd.DataFrame, n_bins: int = 6) -> pd.DataFrame:
    """
    Fast deterministic orientation-family proxy using phi1 bins.
    This is not exact B19′ variant indexing.
    """
    data = df.copy()

    if "phi1" not in data.columns:
        return data

    phi1_num = pd.to_numeric(data["phi1"], errors="coerce")
    width = 360.0 / max(1, n_bins)

    data["orientation_family_proxy"] = (
        np.floor((phi1_num % 360.0) / width).fillna(-1).astype(int) + 1
    ).astype(str)

    return data


def plot_grid_map(df: pd.DataFrame, value: str, title: str):
    pivot = grid_for_plot(df, value)

    if pivot is None:
        st.info("Could not reconstruct a regular x-y grid for this map.")
        return

    if pivot.dtypes.astype(str).str.contains("object|category|string").any():
        flat = pivot.to_numpy().ravel()
        codes, uniques = pd.factorize(flat)
        code_img = codes.reshape(pivot.shape)

        fig = px.imshow(code_img, aspect="equal", title=title)
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True)

        legend_df = pd.DataFrame(
            {
                "code": range(len(uniques)),
                "label": [str(u) for u in uniques],
            }
        )
        st.caption("Categorical map legend")
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
# SEM / optical image tab
# =============================================================================

with tab_img:
    st.subheader("Defect and pore/crack screening")

    st.markdown(
        """
Use this for polished SEM/optical images where dark or bright features may represent pores, cracks, pull-outs or etching contrast.

For publication claims, always calibrate pixel size and manually verify segmentation.
"""
    )

    cset, cview = st.columns([0.8, 1.2])

    with cset:
        demo = st.checkbox("Use demo microstructure image", value=True, key="img_demo")
        uploaded_img = None if demo else st.file_uploader(
            "Upload SEM/optical image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="img_upload",
        )

        invert = st.checkbox("Dark features are pores/cracks", value=True)
        th = st.slider("Manual threshold; 0 = Otsu", 0, 255, 0)
        pixel = st.number_input("Pixel size (µm/pixel; 0 = unknown)", min_value=0.0, value=0.0, step=0.01)

    img = None

    if demo:
        demo_path = Path("assets/demo_microstructure.png")

        if demo_path.exists():
            img = Image.open(demo_path)
        else:
            st.warning("Demo image not found. Upload an image instead.")
    elif uploaded_img is not None:
        img = load_image(uploaded_img)

    if img is not None:
        result = porosity_screen(
            img,
            invert=invert,
            manual_threshold=None if th == 0 else th,
            pixel_size_um=pixel if pixel > 0 else None,
        )

        texture = image_texture_descriptors(img)

        c1, c2 = st.columns(2)
        c1.image(img, caption="Input image", use_container_width=True)
        c2.image(result["mask"], caption="Segmentation mask", use_container_width=True)

        area_pct = float(result.get("feature_area_fraction_pct", 0.0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Feature area fraction", f"{area_pct:.3f}%")

        if "feature_count" in result:
            m2.metric("Feature count", f"{result['feature_count']}")

        if "mean_feature_area_um2" in result:
            m3.metric("Mean feature area", f"{result['mean_feature_area_um2']:.2f} µm²")

        m4.metric("Texture contrast proxy", f"{texture.get('std_gray', 0):.1f}")

        with st.expander("Detailed image descriptors"):
            st.json({k: v for k, v in result.items() if k != "mask"})
            st.json(texture)

        if area_pct < 0.25:
            st.success("Interpretation: very low segmented defect fraction. Verify contrast before calling it dense.")
        elif area_pct < 1.5:
            st.info("Interpretation: moderate feature fraction. Check whether the features are true pores/cracks or polishing/etching artifacts.")
        else:
            st.warning(
                "Interpretation: high segmented feature fraction. Check for lack-of-fusion pores, keyhole pores, cracks or preparation artifacts."
            )

        if pixel <= 0:
            st.caption("Pixel size is missing, so the result is area-fraction based only. Add µm/pixel for size-based defect reporting.")


# =============================================================================
# XRD tab
# =============================================================================

with tab_xrd:
    st.subheader("XRD phase screening with NiTi-specific peak matching")

    st.markdown(
        """
Upload a CSV with two columns: `two_theta_deg` and `intensity`.

The tool performs baseline correction, peak detection, d-spacing conversion, B2/B19′ matching, and secondary-phase warnings.
"""
    )

    sx1, sx2, sx3 = st.columns(3)

    with sx1:
        demo_xrd = st.checkbox("Use demo XRD CSV", value=True, key="xrd_demo")
        xrd_file = None if demo_xrd else st.file_uploader("Upload XRD CSV", type=["csv"], key="xrd_upload")
        wavelength = st.number_input("X-ray wavelength λ (Å)", min_value=0.5, max_value=3.0, value=CU_K_ALPHA_A, step=0.0001)

    with sx2:
        smooth_win = st.slider("Smoothing window", 3, 31, 7, 2)
        baseline_win = st.slider("Baseline window", 21, 501, 101, 20)
        min_prom = st.slider("Minimum peak prominence (%)", 1.0, 30.0, 5.0, 0.5)

    with sx3:
        min_spacing = st.slider("Minimum peak spacing (°2θ)", 0.05, 1.50, 0.25, 0.05)
        match_tol = st.slider("Phase-match tolerance (°2θ)", 0.05, 1.50, 0.35, 0.05)
        max_peaks = st.slider("Maximum peaks", 5, 100, 40, 5)

    with st.expander("Reference lattice parameters used for XRD matching", expanded=False):
        p1, p2, p3, p4, p5 = st.columns(5)

        b2_a_ref = p1.number_input("B2 a (Å)", value=3.015, step=0.001, format="%.4f")
        b19_a_ref = p2.number_input("B19′ a (Å)", value=2.889, step=0.001, format="%.4f")
        b19_b_ref = p3.number_input("B19′ b (Å)", value=4.120, step=0.001, format="%.4f")
        b19_c_ref = p4.number_input("B19′ c (Å)", value=4.622, step=0.001, format="%.4f")
        b19_beta_ref = p5.number_input("B19′ β (deg)", value=96.80, step=0.01, format="%.2f")

    xrd_loaded = False

    if demo_xrd or xrd_file is not None:
        try:
            raw_xrd = read_xrd_csv("data/demo_xrd.csv" if demo_xrd else xrd_file)
            xrd = prepare_xrd_dataframe(raw_xrd)

            processed_xrd, peaks = pick_xrd_peaks(
                xrd,
                smooth_window=smooth_win,
                baseline_window=baseline_win,
                min_prominence_pct=min_prom,
                min_spacing_deg=min_spacing,
                max_peaks=max_peaks,
                wavelength_A=wavelength,
            )

            refs = expected_niti_refs(
                b2_a_ref,
                b19_a_ref,
                b19_b_ref,
                b19_c_ref,
                b19_beta_ref,
                wavelength,
            )

            matches, phase_scores = match_xrd_peaks(peaks, refs, match_tol)
            xrd_loaded = True

        except Exception as exc:
            st.error(f"Could not read/process XRD file: {exc}")

    if xrd_loaded:
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
            yaxis_title="Intensity / corrected intensity",
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns([1.0, 1.0])

        with c1:
            st.subheader("Detected peaks")
            st.dataframe(peaks, use_container_width=True, hide_index=True)

        with c2:
            st.subheader("Phase screening score")
            st.dataframe(phase_scores, use_container_width=True, hide_index=True)

            if not phase_scores.empty:
                top_phase = str(phase_scores.iloc[0]["phase"])

                if "B2" in top_phase:
                    st.success("Top XRD match: B2/austenite-like. This supports superelastic design, but DSC is still required.")
                elif "B19" in top_phase:
                    st.warning("Top XRD match: B19′/martensite-like. Room-temperature martensite may weaken superelastic response.")
                else:
                    st.warning("Top XRD match is a secondary/warning phase. Verify with refinement and microscopy.")

        st.subheader("Matched reference peaks")
        st.dataframe(matches, use_container_width=True, hide_index=True)

        with st.expander("Reference peak table"):
            st.dataframe(refs, use_container_width=True, hide_index=True)

        st.markdown(
            """
**How to use this result**

- B2-dominant pattern near service temperature supports a superelastic design route.
- B19′ indicators near room temperature suggest transformation-temperature/composition problems.
- Ni-rich/Ti-rich/oxide warning peaks should trigger EDS/ICP, SEM, and heat-treatment checks.
- Peak overlap is severe in NiTi, so do not claim exact phase fractions from this screening table alone.
"""
        )


# =============================================================================
# Crystallography tab
# =============================================================================

with tab_crystal:
    st.subheader("B2 ↔ B19′ crystallography and compatibility metrics")

    st.markdown(
        """
This tab makes the crystallography useful instead of vague. It computes lattice metrics, d-spacings, a simplified B2→B19′ correspondence stretch, λ₂ compatibility proxy, volume change and transformation-strain indicators.

The λ₂ result is a **compatibility proxy**. Exact habit-plane prediction and variant reconstruction require a full PTMC/EBSD workflow.
"""
    )

    cl1, cl2 = st.columns([1.0, 1.0])

    with cl1:
        st.markdown("**Parent B2 / austenite**")
        b2_a = st.number_input("B2 lattice parameter a₀ (Å)", value=3.0150, step=0.0005, format="%.4f")

        st.markdown("**Martensite B19′**")
        b19_a = st.number_input("B19′ a (Å)", value=2.8890, step=0.0005, format="%.4f")
        b19_b = st.number_input("B19′ b (Å)", value=4.1200, step=0.0005, format="%.4f")
        b19_c = st.number_input("B19′ c (Å)", value=4.6220, step=0.0005, format="%.4f")
        b19_beta = st.number_input("B19′ β (deg)", value=96.80, step=0.01, format="%.2f")

    with cl2:
        metrics = transformation_stretch_proxy(b2_a, b19_a, b19_b, b19_c, b19_beta)

        m1, m2, m3 = st.columns(3)
        m1.metric("λ₁", f"{metrics['lambda1']:.5f}")
        m2.metric("λ₂", f"{metrics['lambda2']:.5f}")
        m3.metric("λ₃", f"{metrics['lambda3']:.5f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("|λ₂ − 1|", f"{metrics['abs_lambda2_minus_1']:.5f}")
        m5.metric("Volume change", f"{metrics['volume_change_pct']:+.2f}%")
        m6.metric("Compatibility score", f"{metrics['compatibility_score_0_100']:.1f}/100")

        if metrics["abs_lambda2_minus_1"] < 0.01:
            st.success("λ₂ is close to 1: good crystallographic compatibility proxy for recoverable transformation.")
        elif metrics["abs_lambda2_minus_1"] < 0.03:
            st.info("λ₂ is moderately close to 1: possible compatibility, but expect sensitivity to composition/heat treatment.")
        else:
            st.warning("λ₂ is far from 1: higher crystallographic mismatch risk. Validate against cyclic strain and DSC.")

    st.subheader("Transformation-stretch proxy table")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)

    st.subheader("Interactive d-spacing / peak calculator")

    dc1, dc2, dc3, dc4 = st.columns(4)

    phase_for_d = dc1.selectbox("Phase", ["B2 cubic", "B19′ monoclinic"])
    h = dc2.number_input("h", value=1, step=1)
    k = dc3.number_input("k", value=1, step=1)
    l = dc4.number_input("l", value=0, step=1)

    wavelength_calc = st.number_input("Wavelength for calculated 2θ (Å)", value=CU_K_ALPHA_A, step=0.0001, format="%.4f")

    if phase_for_d == "B2 cubic":
        d_calc = cubic_d_spacing(b2_a, int(h), int(k), int(l))
    else:
        d_calc = monoclinic_d_spacing(b19_a, b19_b, b19_c, b19_beta, int(h), int(k), int(l))

    tt_calc = two_theta_from_d(d_calc, wavelength_calc)

    d1, d2 = st.columns(2)
    d1.metric("d-spacing", f"{d_calc:.4f} Å" if np.isfinite(d_calc) else "not valid")
    d2.metric("Calculated 2θ", f"{tt_calc:.3f}°" if np.isfinite(tt_calc) else "not valid")

    st.subheader("Generated NiTi reference peaks from current lattice parameters")

    ref_table = expected_niti_refs(
        b2_a,
        b19_a,
        b19_b,
        b19_c,
        b19_beta,
        wavelength_calc,
    )

    st.dataframe(ref_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Why this matters for the project**

- The process model predicts composition and transformation-temperature shift.
- XRD tells whether B2/B19′/secondary phases are consistent with that prediction.
- This crystallography tab checks whether the B2→B19′ lattice change is geometrically compatible.
- EBSD/TKD then checks whether orientation, texture and local misorientation support or damage functional reversibility.
"""
    )


# =============================================================================
# EBSD / TKD tab
# =============================================================================

with tab_ebsd:
    st.subheader("EBSD/TKD orientation and local crystallography screening")

    st.markdown(
        """
Upload a CSV containing at least `x`, `y`, and Euler angles: `phi1`, `Phi`, `phi2`.

Optional useful columns: `phase`, `ci`, `iq`, `grain_id`.

This tab adds phase fractions, IPF-Z color proxy, KAM/orientation-gradient proxy, grain statistics and orientation-family maps.
"""
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        demo_ebsd = st.checkbox("Use demo EBSD CSV", value=True, key="ebsd_demo")
        ebsd_file = None if demo_ebsd else st.file_uploader("Upload EBSD/TKD CSV", type=["csv"], key="ebsd_upload")

    with e2:
        ci_min = st.number_input("Minimum CI filter", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    with e3:
        iq_min = st.number_input("Minimum IQ filter", min_value=0.0, value=0.0, step=1.0)

    ebsd_loaded = False

    if demo_ebsd or ebsd_file is not None:
        try:
            ebsd_raw = read_ebsd_csv("data/demo_ebsd.csv" if demo_ebsd else ebsd_file)
            ebsd = ebsd_raw.copy()

            if "ci" in ebsd.columns and ci_min > 0:
                ebsd = ebsd[ebsd["ci"] >= ci_min]

            if "iq" in ebsd.columns and iq_min > 0:
                ebsd = ebsd[ebsd["iq"] >= iq_min]

            ebsd = add_orientation_descriptors(ebsd)
            ebsd = orientation_family_bins(ebsd, n_bins=6)
            summary = ebsd_summary(ebsd)

            ebsd_loaded = True

        except Exception as exc:
            st.error(f"Could not read/process EBSD/TKD file: {exc}")

    if ebsd_loaded:
        st.subheader("Data preview")
        st.dataframe(ebsd.head(50), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Indexed points", f"{len(ebsd):,}")

        if "phase" in ebsd.columns:
            c2.metric("Phases", f"{ebsd['phase'].nunique()}")

        if "grain_id" in ebsd.columns:
            c3.metric("Grains", f"{ebsd['grain_id'].nunique()}")

        if "kam_deg_no_symmetry" in ebsd.columns:
            c4.metric("Mean KAM proxy", f"{ebsd['kam_deg_no_symmetry'].mean():.2f}°")

        if "phase" in ebsd.columns:
            phase_frac = (
                ebsd["phase"]
                .value_counts(normalize=True)
                .rename_axis("phase")
                .reset_index(name="fraction")
            )

            phase_frac["fraction_pct"] = 100.0 * phase_frac["fraction"]

            st.subheader("Phase fraction from indexed points")
            st.dataframe(phase_frac[["phase", "fraction_pct"]], use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(phase_frac, x="phase", y="fraction_pct", title="Indexed phase fraction"), use_container_width=True)

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
            value = st.selectbox("Map value", map_options)
            plot_grid_map(ebsd, value, f"Map: {value}")

        if {"ipf_r", "ipf_g", "ipf_b"}.issubset(ebsd.columns):
            st.subheader("IPF-Z color proxy")

            img_ipf = make_ipf_image(ebsd)

            if img_ipf is not None:
                fig_ipf = go.Figure(go.Image(z=(np.clip(img_ipf, 0, 1) * 255).astype(np.uint8)))
                fig_ipf.update_layout(
                    height=550,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title="IPF-Z proxy image",
                )
                st.plotly_chart(fig_ipf, use_container_width=True)
            else:
                st.info("IPF proxy values were calculated, but x/y grid could not be reconstructed for image plotting.")

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

            high_kam_frac = float((ebsd["kam_deg_no_symmetry"] > 5.0).mean() * 100.0)

            if high_kam_frac > 20:
                st.warning(
                    f"{high_kam_frac:.1f}% of indexed points have KAM proxy > 5°. This may indicate high orientation gradients, strain, poor indexing, or substructure."
                )
            else:
                st.success(
                    f"{high_kam_frac:.1f}% of indexed points have KAM proxy > 5°. Orientation-gradient level is not severe by this simple proxy."
                )

        gstats = grain_statistics(ebsd)

        if not gstats.empty:
            st.subheader("Grain statistics")
            st.dataframe(gstats.head(100), use_container_width=True, hide_index=True)

        with st.expander("Raw EBSD summary"):
            st.json(summary)

        st.markdown(
            """
**Important crystallography note**

The orientation-family map is **not exact B19′ variant indexing**. Exact variant indexing needs a parent B2 orientation, an orientation relationship, crystal symmetry operators, and measured martensite orientations. This tab gives a useful first-pass map for texture, local misorientation and phase consistency.
"""
        )


# =============================================================================
# Evidence report tab
# =============================================================================

with tab_report:
    st.subheader("How to turn characterization into a useful NiTi function argument")

    st.markdown(
        """
Use the page outputs in this order:

1. **Image evidence**: are there pores, cracks, lack-of-fusion features, keyhole pores or preparation artifacts?
2. **XRD evidence**: is the phase state mainly B2/austenite, B19′/martensite, mixed, or contaminated by secondary/oxide peaks?
3. **Crystallography evidence**: is the B2→B19′ lattice correspondence geometrically compatible? Check λ₂ and volume change.
4. **EBSD/TKD evidence**: is phase distribution, texture, local misorientation and grain structure consistent with recoverable transformation?
5. **Process-function link**: compare these observations with your vaporization/composition model and DSC transformation temperatures.
"""
    )

    st.subheader("Copy-ready interpretation template")

    st.code(
        """
The characterization workflow combines microstructure segmentation, XRD phase screening,
B2/B19′ lattice-compatibility metrics and EBSD/TKD orientation mapping. The image analysis
checks whether the process window produced excessive pores/cracks. XRD peak matching tests
whether the room-temperature phase state is consistent with B2/austenite, B19′/martensite or
secondary phases. The crystallography module estimates the B2→B19′ stretch compatibility
through λ2, volume change and principal strain metrics. EBSD/TKD maps provide phase fraction,
texture, local orientation-gradient and grain-scale evidence. Together, these outputs connect
LPBF processing and vaporization-induced composition shift to the final functional NiTi state.
""".strip(),
        language="text",
    )

    st.subheader("What still needs real experimental validation")

    st.markdown(
        """
- DSC for Ms, Mf, As and Af.
- ICP/EDS for powder vs printed-part Ni/Ti composition.
- Rietveld refinement for phase fraction and lattice-parameter refinement.
- SEM cross-section validation for pore/crack morphology.
- EBSD/TKD parent reconstruction and exact B19′ variant indexing.
- Cyclic superelastic or shape-memory testing for final function.
"""
    )
