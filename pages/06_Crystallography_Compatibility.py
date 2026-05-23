import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.constants import NITI_DEFAULTS
except Exception:
    NITI_DEFAULTS = {
        "B2_a_A": 3.015,
        "B19p_a_A": 2.889,
        "B19p_b_A": 4.120,
        "B19p_c_A": 4.622,
        "B19p_beta_deg": 96.80,
        "natural_OR_planes": "Use measured EBSD/TKD OR when available.",
        "natural_OR_directions": "Use measured EBSD/TKD OR when available.",
        "candidate_habit_plane": "Use measured habit trace / literature candidate only as guide.",
    }

try:
    from src.crystallography import (
        generate_simplified_b19_variants,
        variant_pair_table,
        stress_variant_scores,
    )
    HAS_PROJECT_VARIANT_FUNCTIONS = True
except Exception:
    HAS_PROJECT_VARIANT_FUNCTIONS = False
    generate_simplified_b19_variants = None
    variant_pair_table = None
    stress_variant_scores = None


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Crystallography and Compatibility", layout="wide")

st.title("Crystallography and Compatibility")
st.caption(
    "B2→B19′ NiTi lattice compatibility, correspondence-theory screening, "
    "variant-family logic, and XRD support."
)

st.markdown(
    """
This page should answer four questions:

1. **Where do my crystallographic inputs come from?**
2. **Are the B2 and B19′ lattices geometrically compatible?**
3. **What does correspondence theory imply for variants, twins and junctions?**
4. **What can I safely claim, and what evidence is still missing?**

This page is a **screening and explanation layer**. It does not replace full PTMC,
Cayron/GenOVa-style correspondence calculations, TKD/EBSD parent reconstruction,
Rietveld refinement, or DSC.
"""
)


# =============================================================================
# Helper functions
# =============================================================================

CU_K_ALPHA_A = 1.5406


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def source_score(source: str) -> int:
    mapping = {
        "Literature/default value": 1,
        "Estimated from simple peak positions": 2,
        "XRD Pawley/Rietveld refined": 4,
        "Temperature-resolved XRD refined": 5,
        "EBSD/TKD measured": 4,
        "DSC measured": 4,
        "ICP/EDS measured": 4,
        "Unknown/not measured": 0,
    }
    return mapping.get(source, 0)


def quality_label(score: float) -> Tuple[str, str]:
    if score >= 80:
        return "high", "Strong evidence basis. You can discuss trends with reasonable confidence."
    if score >= 55:
        return "medium", "Useful evidence basis, but avoid strong quantitative claims."
    if score >= 30:
        return "low", "Screening only. Treat conclusions as hypotheses."
    return "very low", "Do not make strong crystallography/function claims yet."


def monoclinic_volume(a_A: float, b_A: float, c_A: float, beta_deg: float) -> float:
    return a_A * b_A * c_A * math.sin(math.radians(beta_deg))


def b19_basis(a_A: float, b_A: float, c_A: float, beta_deg: float) -> np.ndarray:
    """
    Monoclinic B19′ basis with unique b axis.

    Columns represent martensite basis vectors:
        a_m = [a, 0, 0]
        b_m = [0, b, 0]
        c_m = [c cos(beta), 0, c sin(beta)]
    """
    beta = math.radians(beta_deg)
    return np.array(
        [
            [a_A, 0.0, c_A * math.cos(beta)],
            [0.0, b_A, 0.0],
            [0.0, 0.0, c_A * math.sin(beta)],
        ],
        dtype=float,
    )


def parent_correspondence_basis(B2a_A: float) -> np.ndarray:
    """
    Simplified B2 correspondence supercell.

    This is a transparent screening basis:
        parent length 1 = a_B2
        parent length 2 = sqrt(2) a_B2
        parent length 3 = sqrt(2) a_B2

    Its volume is 2*a_B2^3, which matches the common B19′ conventional-cell
    normalization used for a first compatibility check.
    """
    return np.diag([B2a_A, math.sqrt(2.0) * B2a_A, math.sqrt(2.0) * B2a_A])


def crystallographic_metrics(
    B2a_A: float,
    b19_a_A: float,
    b19_b_A: float,
    b19_c_A: float,
    beta_deg: float,
) -> Dict[str, float]:
    V_B2_cell = B2a_A ** 3
    V_B2_pair = V_B2_cell

    V_B19_cell = monoclinic_volume(b19_a_A, b19_b_A, b19_c_A, beta_deg)
    V_B19_pair = V_B19_cell / 2.0

    V_parent_corr = 2.0 * V_B2_cell

    raw_cell_volume_change_pct = 100.0 * (V_B19_cell - V_B2_cell) / V_B2_cell
    normalized_volume_change_pct = 100.0 * (V_B19_pair - V_B2_pair) / V_B2_pair
    correspondence_volume_change_pct = 100.0 * (V_B19_cell - V_parent_corr) / V_parent_corr

    Bp = parent_correspondence_basis(B2a_A)
    Bm = b19_basis(b19_a_A, b19_b_A, b19_c_A, beta_deg)

    F = Bm @ np.linalg.inv(Bp)
    C = F.T @ F
    eigvals = np.linalg.eigvalsh(C)
    lambdas = np.sqrt(np.maximum(eigvals, 0.0))
    lambdas = np.sort(lambdas)

    lambda1, lambda2, lambda3 = lambdas
    abs_lambda2_minus_1 = abs(lambda2 - 1.0)

    principal_strains_pct = 100.0 * (lambdas - 1.0)
    max_abs_principal_strain_pct = float(np.max(np.abs(principal_strains_pct)))

    detF = float(np.linalg.det(F))

    compatibility_score = 100.0
    compatibility_score -= 4000.0 * abs_lambda2_minus_1
    compatibility_score -= 200.0 * abs(correspondence_volume_change_pct)
    compatibility_score -= 4.0 * max_abs_principal_strain_pct
    compatibility_score = clamp(compatibility_score, 0.0, 100.0)

    return {
        "B2_cell_volume_A3": V_B2_cell,
        "B2_volume_per_NiTi_A3": V_B2_pair,
        "B19p_cell_volume_A3": V_B19_cell,
        "B19p_volume_per_NiTi_A3": V_B19_pair,
        "parent_correspondence_volume_A3": V_parent_corr,
        "raw_cell_volume_change_pct_wrong_if_used_directly": raw_cell_volume_change_pct,
        "normalized_volume_change_pct": normalized_volume_change_pct,
        "correspondence_volume_change_pct": correspondence_volume_change_pct,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
        "abs_lambda2_minus_1": float(abs_lambda2_minus_1),
        "principal_strain_1_pct": float(principal_strains_pct[0]),
        "principal_strain_2_pct": float(principal_strains_pct[1]),
        "principal_strain_3_pct": float(principal_strains_pct[2]),
        "max_abs_principal_strain_pct": max_abs_principal_strain_pct,
        "detF": detF,
        "compatibility_score_0_100": compatibility_score,
    }


def lambda2_verdict(abs_l2_error: float) -> Tuple[str, str, str]:
    if abs_l2_error < 0.005:
        return (
            "excellent",
            "λ₂ is very close to 1.",
            "Good lattice-compatibility proxy; this supports low-mismatch reversible transformation, if phase state and defects are also good.",
        )
    if abs_l2_error < 0.015:
        return (
            "good/moderate",
            "λ₂ is reasonably close to 1.",
            "Compatibility is plausible, but functional behavior still depends on composition, DSC temperatures, texture, defects and cycling.",
        )
    if abs_l2_error < 0.030:
        return (
            "warning",
            "λ₂ is not very close to 1.",
            "Expect increased mismatch/hysteresis risk. Do not claim strong compatibility without EBSD/TKD and cyclic testing.",
        )
    return (
        "poor",
        "λ₂ is far from 1.",
        "High crystallographic mismatch risk. Treat superelastic reversibility as questionable unless experiments prove otherwise.",
    )


def volume_verdict(vol_pct: float) -> Tuple[str, str]:
    av = abs(vol_pct)
    if av < 0.5:
        return "excellent", "Very small normalized volume change."
    if av < 1.5:
        return "acceptable", "Moderate normalized volume change."
    if av < 3.0:
        return "warning", "Volume mismatch may contribute to transformation strain/hysteresis."
    return "poor", "Large volume mismatch warning; check lattice parameters and phase refinement."


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


def reference_peaks(
    B2a_A: float,
    b19_a_A: float,
    b19_b_A: float,
    b19_c_A: float,
    beta_deg: float,
    wavelength_A: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    b2_hkls = [(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0), (3, 1, 0), (2, 2, 2)]
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

    for h, k, l in b2_hkls:
        d = cubic_d_spacing(B2a_A, h, k, l)
        tt = two_theta_from_d(d, wavelength_A)
        if np.isfinite(tt):
            rows.append(
                {
                    "phase": "B2 austenite",
                    "hkl": f"({h}{k}{l})",
                    "d_A": d,
                    "two_theta_deg": tt,
                    "meaning": "B2 peak position from selected austenite lattice parameter.",
                }
            )

    for h, k, l in b19_hkls:
        d = monoclinic_d_spacing(b19_a_A, b19_b_A, b19_c_A, beta_deg, h, k, l)
        tt = two_theta_from_d(d, wavelength_A)
        if np.isfinite(tt) and 20 <= tt <= 100:
            rows.append(
                {
                    "phase": "B19′ martensite",
                    "hkl": f"({h}{k}{l})",
                    "d_A": d,
                    "two_theta_deg": tt,
                    "meaning": "B19′ peak position from selected monoclinic lattice parameters.",
                }
            )

    secondary = [
        ("Ni4Ti3 / Ni-rich precipitate check", "warning", 43.5),
        ("Ti2Ni / Ti-rich secondary check", "warning", 39.2),
        ("Ti2Ni / Ti-rich secondary check", "warning", 42.2),
        ("TiO2 / oxide check", "warning", 27.4),
        ("TiO2 / oxide check", "warning", 36.1),
    ]

    for phase, hkl, tt in secondary:
        rows.append(
            {
                "phase": phase,
                "hkl": hkl,
                "d_A": bragg_d_spacing(tt, wavelength_A),
                "two_theta_deg": tt,
                "meaning": "Approximate warning position only. Confirm with full refinement.",
            }
        )

    return pd.DataFrame(rows).sort_values("two_theta_deg").reset_index(drop=True)


def cayron_operator_table() -> pd.DataFrame:
    """
    Educational Cayron-style correspondence-theory table.

    This is not a full GenOVa output. It summarizes the usable idea for the app:
    B2→B19′ has 12 variants and 7 operator classes. Ambivalent operators can be
    treated through inherited mirror/rotation symmetry; polar operators may need
    weak-plane logic.
    """
    rows = [
        {
            "operator": "O0",
            "class": "neutral",
            "variant relation": "same variant / identity",
            "junction idea": "No variant-pair junction needed.",
            "what to check in EBSD/TKD": "Same orientation family.",
            "safe interpretation": "Reference case only.",
        },
        {
            "operator": "O1",
            "class": "ambivalent",
            "variant relation": "compound-like / 180°-class relation",
            "junction idea": "Type-I or Type-II twin element can be inherited from parent mirror/2-fold symmetry.",
            "what to check in EBSD/TKD": "Strong paired lath relation with repeatable misorientation.",
            "safe interpretation": "Potential compatibility twin family.",
        },
        {
            "operator": "O2",
            "class": "ambivalent",
            "variant relation": "120°-class relation",
            "junction idea": "Junction element can be predicted by correspondence from parent symmetry.",
            "what to check in EBSD/TKD": "Variant-pair boundary trace and misorientation cluster.",
            "safe interpretation": "Potential twin/junction family; verify trace.",
        },
        {
            "operator": "O3",
            "class": "ambivalent",
            "variant relation": "120°-class relation",
            "junction idea": "Related to another inherited parent-symmetry operator.",
            "what to check in EBSD/TKD": "Compare with O2-like family and trace statistics.",
            "safe interpretation": "Potential twin/junction family; not proof alone.",
        },
        {
            "operator": "O4",
            "class": "ambivalent",
            "variant relation": "90°-class relation",
            "junction idea": "Correspondence predicts rational/near-rational junction elements.",
            "what to check in EBSD/TKD": "Boundary trace should repeat across laths if real.",
            "safe interpretation": "Useful for lath-pair classification.",
        },
        {
            "operator": "O5",
            "class": "polar",
            "variant relation": "polar pair",
            "junction idea": "Normal invariant-plane PTMC solution may fail; weak-plane concept can explain junctions.",
            "what to check in EBSD/TKD": "Weak-plane traces and non-equivalent plane correspondences.",
            "safe interpretation": "Do not force simple twin interpretation.",
        },
        {
            "operator": "O6",
            "class": "polar",
            "variant relation": "complementary polar pair",
            "junction idea": "Weak-plane / weak-twin logic may be needed.",
            "what to check in EBSD/TKD": "Trace consistency, local distortion, parent reconstruction.",
            "safe interpretation": "Advanced CT/EBSD analysis required.",
        },
    ]
    return pd.DataFrame(rows)


def fallback_variant_pairs() -> pd.DataFrame:
    rows = []
    for i in range(1, 13):
        for j in range(i + 1, 13):
            diff = abs(i - j)
            if diff == 0:
                op = "O0"
            elif diff in [1, 11]:
                op = "O1"
            elif diff in [2, 10]:
                op = "O2"
            elif diff in [3, 9]:
                op = "O3"
            elif diff in [4, 8]:
                op = "O4"
            elif diff in [5, 7]:
                op = "O5"
            else:
                op = "O6"

            base_score = {
                "O1": 90,
                "O2": 78,
                "O3": 76,
                "O4": 70,
                "O5": 45,
                "O6": 45,
            }.get(op, 50)

            rows.append(
                {
                    "variant_i": f"V{i:02d}",
                    "variant_j": f"V{j:02d}",
                    "operator_family": op,
                    "compatibility_proxy": base_score,
                    "meaning": "Placeholder CT-family ranking. Replace with measured TKD/EBSD variant IDs for final work.",
                }
            )
    return pd.DataFrame(rows)


def stress_axis_proxy_table(axis_name: str) -> pd.DataFrame:
    """
    Educational stress/build-axis proxy.

    This does not calculate true interaction work. It shows how variant families
    could be ranked if their transformation direction/habit alignment were known.
    """
    axis_vectors = {
        "Z/build": np.array([0.0, 0.0, 1.0]),
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Custom diagonal [111]": np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
    }
    axis = axis_vectors[axis_name]

    candidate_normals = [
        ("{110}-A", np.array([1.0, 1.0, 0.0])),
        ("{110}-B", np.array([1.0, -1.0, 0.0])),
        ("{101}-A", np.array([1.0, 0.0, 1.0])),
        ("{101}-B", np.array([1.0, 0.0, -1.0])),
        ("{011}-A", np.array([0.0, 1.0, 1.0])),
        ("{011}-B", np.array([0.0, 1.0, -1.0])),
        ("{100}", np.array([1.0, 0.0, 0.0])),
        ("{010}", np.array([0.0, 1.0, 0.0])),
        ("{001}", np.array([0.0, 0.0, 1.0])),
        ("{111}-A", np.array([1.0, 1.0, 1.0])),
        ("{111}-B", np.array([1.0, -1.0, 1.0])),
        ("{111}-C", np.array([-1.0, 1.0, 1.0])),
    ]

    rows = []
    for idx, (name, n) in enumerate(candidate_normals, start=1):
        n = n / np.linalg.norm(n)
        alignment = abs(float(np.dot(axis, n)))
        rows.append(
            {
                "variant_family_proxy": f"VF{idx:02d}",
                "plane/direction proxy": name,
                "axis_alignment_0_1": alignment,
                "screening_rank_score": 100.0 * alignment,
                "meaning": "High means this family is geometrically aligned with selected build/loading axis. Not true interaction work.",
            }
        )

    return pd.DataFrame(rows).sort_values("screening_rank_score", ascending=False)


def build_final_verdict(
    quality_pct: float,
    metrics: Dict[str, float],
    phase_evidence: str,
    dsc_evidence: str,
    defect_evidence: str,
) -> Tuple[str, str, pd.DataFrame]:
    l2_state, l2_short, l2_long = lambda2_verdict(metrics["abs_lambda2_minus_1"])
    vol_state, vol_text = volume_verdict(metrics["normalized_volume_change_pct"])

    missing = []
    positives = []
    warnings = []

    if quality_pct < 55:
        warnings.append("Data-quality score is not high enough for strong claims.")

    if l2_state in ["excellent", "good/moderate"]:
        positives.append(l2_long)
    else:
        warnings.append(l2_long)

    if vol_state in ["excellent", "acceptable"]:
        positives.append(vol_text)
    else:
        warnings.append(vol_text)

    if "B2" in phase_evidence:
        positives.append("Phase evidence supports austenite/B2 presence.")
    elif "B19" in phase_evidence:
        warnings.append("Phase evidence suggests B19′/martensite at the checked condition.")
    elif "mixed" in phase_evidence.lower():
        warnings.append("Mixed phase evidence: service response may be unstable or temperature-sensitive.")
    else:
        missing.append("Reliable XRD/Rietveld phase evidence.")

    if "measured" in dsc_evidence.lower():
        positives.append("DSC transformation temperatures are available.")
    else:
        missing.append("DSC Ms/Mf/As/Af measurement.")

    if "low" in defect_evidence.lower():
        positives.append("Defect evidence does not dominate the interpretation.")
    elif "high" in defect_evidence.lower():
        warnings.append("High defect/crack/porosity evidence may dominate over crystallographic compatibility.")
    else:
        missing.append("Quantified SEM/optical defect fraction.")

    if quality_pct >= 70 and len(warnings) == 0:
        headline = "Strong compatibility candidate"
        explanation = (
            "The lattice compatibility metrics and evidence quality are favorable. "
            "You can argue that the selected B2/B19′ lattice set is a plausible functional NiTi candidate, "
            "but still validate with DSC and cyclic superelastic/shape-memory testing."
        )
    elif quality_pct >= 45 and l2_state in ["excellent", "good/moderate"]:
        headline = "Promising but not proven"
        explanation = (
            "The crystallographic metrics are promising, but the evidence basis is incomplete. "
            "Use this as a screening result, not as a final claim."
        )
    elif l2_state in ["warning", "poor"]:
        headline = "Crystallographic mismatch warning"
        explanation = (
            "The λ₂/volume metrics suggest mismatch risk. Check the lattice parameters, temperature, "
            "composition, XRD refinement and actual phase state before using this as a superelastic design."
        )
    else:
        headline = "Insufficient evidence"
        explanation = (
            "The page cannot support a strong conclusion. Add measured lattice parameters, phase evidence, "
            "DSC and EBSD/TKD before making a crystallographic claim."
        )

    action_rows = []
    for item in positives:
        action_rows.append({"type": "supports claim", "item": item})
    for item in warnings:
        action_rows.append({"type": "warning", "item": item})
    for item in missing:
        action_rows.append({"type": "missing evidence", "item": item})

    return headline, explanation, pd.DataFrame(action_rows)


# =============================================================================
# Sidebar / global inputs
# =============================================================================

with st.sidebar:
    st.header("Global function context")

    effective_ni = st.number_input(
        "Effective Ni after printing (at.%)",
        min_value=45.0,
        max_value=55.0,
        value=51.00,
        step=0.01,
        format="%.2f",
        key="global_effective_ni",
        help="Use nominal powder Ni only if ICP/EDS or vaporization-corrected Ni is not available.",
    )

    service_temperature = st.number_input(
        "Service / test temperature (°C)",
        min_value=-200.0,
        max_value=300.0,
        value=25.0,
        step=1.0,
        key="global_service_temperature",
    )

    phase_evidence = st.selectbox(
        "Phase evidence at service/check temperature",
        [
            "Unknown/not measured",
            "Mostly B2/austenite",
            "Mostly B19′/martensite",
            "Mixed B2 + B19′",
            "Secondary phases/oxide suspected",
        ],
        key="global_phase_evidence",
    )

    dsc_evidence = st.selectbox(
        "DSC evidence",
        [
            "Not measured",
            "Measured Ms/Mf/As/Af",
            "Only literature/default transformation temperatures",
            "Only estimated from composition",
        ],
        key="global_dsc_evidence",
    )

    defect_evidence = st.selectbox(
        "Defect evidence",
        [
            "Unknown/not measured",
            "Low defect fraction",
            "Moderate defect fraction",
            "High cracks/pores/lack-of-fusion",
        ],
        key="global_defect_evidence",
    )


# =============================================================================
# Main tabs
# =============================================================================

tab_inputs, tab_metrics, tab_ct, tab_xrd, tab_decision = st.tabs(
    [
        "1. Inputs and evidence quality",
        "2. Lattice compatibility metrics",
        "3. Cayron correspondence / variants",
        "4. XRD support from lattice parameters",
        "5. Final decision report",
    ]
)


# =============================================================================
# Tab 1: Inputs
# =============================================================================

with tab_inputs:
    st.subheader("Step 1 — Enter lattice parameters and say where they came from")

    st.markdown(
        """
The same NiTi alloy can give different lattice parameters depending on composition, temperature,
stress state, heat treatment and whether the measured state is B2, R-phase, B19′ or mixed.
So the app must know whether your values are **measured** or just **defaults**.
"""
    )

    col_lattice, col_source = st.columns([1.0, 1.2])

    with col_lattice:
        st.markdown("### Lattice parameters")

        B2a = st.number_input(
            "B2 a₀ (Å)",
            value=float(NITI_DEFAULTS["B2_a_A"]),
            step=0.0005,
            format="%.4f",
            key="inp_B2a",
            help="Best source: XRD refinement of B2/austenite at the temperature relevant to service/testing.",
        )

        b19_a = st.number_input(
            "B19′ aₘ (Å)",
            value=float(NITI_DEFAULTS["B19p_a_A"]),
            step=0.0005,
            format="%.4f",
            key="inp_b19_a",
            help="Best source: XRD refinement of martensite/B19′.",
        )

        b19_b = st.number_input(
            "B19′ bₘ (Å)",
            value=float(NITI_DEFAULTS["B19p_b_A"]),
            step=0.0005,
            format="%.4f",
            key="inp_b19_b",
        )

        b19_c = st.number_input(
            "B19′ cₘ (Å)",
            value=float(NITI_DEFAULTS["B19p_c_A"]),
            step=0.0005,
            format="%.4f",
            key="inp_b19_c",
        )

        beta = st.number_input(
            "B19′ β (deg)",
            value=float(NITI_DEFAULTS["B19p_beta_deg"]),
            step=0.01,
            format="%.3f",
            key="inp_b19_beta",
            help="β is important because it changes the monoclinic metric, volume and compatibility.",
        )

    with col_source:
        st.markdown("### Data source / reliability")

        B2_source = st.selectbox(
            "Source for B2 a₀",
            [
                "Literature/default value",
                "Estimated from simple peak positions",
                "XRD Pawley/Rietveld refined",
                "Temperature-resolved XRD refined",
                "Unknown/not measured",
            ],
            key="src_B2",
        )

        B19_source = st.selectbox(
            "Source for B19′ a,b,c,β",
            [
                "Literature/default value",
                "Estimated from simple peak positions",
                "XRD Pawley/Rietveld refined",
                "Temperature-resolved XRD refined",
                "Unknown/not measured",
            ],
            key="src_B19",
        )

        ni_source = st.selectbox(
            "Source for effective Ni at.%",
            [
                "Literature/default value",
                "ICP/EDS measured",
                "Estimated from vaporization model",
                "Nominal powder only",
                "Unknown/not measured",
            ],
            key="src_Ni",
        )

        phase_source = st.selectbox(
            "Source for phase state",
            [
                "Unknown/not measured",
                "XRD peak screening only",
                "XRD Pawley/Rietveld refined",
                "EBSD/TKD measured",
                "DSC measured",
            ],
            key="src_phase",
        )

        ebsd_source = st.selectbox(
            "Source for variant/orientation evidence",
            [
                "Unknown/not measured",
                "EBSD/TKD measured",
                "Literature/default value",
            ],
            key="src_ebsd",
        )

    max_score = 5 + 5 + 4 + 5 + 4
    actual_score = (
        source_score(B2_source)
        + source_score(B19_source)
        + source_score(ni_source)
        + source_score(phase_source)
        + source_score(ebsd_source)
    )
    quality_pct = 100.0 * actual_score / max_score
    q_label, q_text = quality_label(quality_pct)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Evidence quality", f"{quality_pct:.0f}/100")
    q2.metric("Quality label", q_label)
    q3.metric("Effective Ni", f"{effective_ni:.2f} at.%")
    q4.metric("Service T", f"{service_temperature:.1f} °C")

    if q_label in ["high", "medium"]:
        st.success(q_text)
    elif q_label == "low":
        st.warning(q_text)
    else:
        st.error(q_text)

    st.subheader("Input guide — how to obtain each input")

    guide = pd.DataFrame(
        [
            {
                "input": "B2 a₀",
                "best source": "XRD Pawley/Rietveld refinement of B2/austenite",
                "backup": "literature default",
                "meaning": "Parent austenite lattice size; controls parent reference metric.",
                "warning": "Should match temperature and composition of your sample.",
            },
            {
                "input": "B19′ aₘ,bₘ,cₘ,β",
                "best source": "XRD refinement of martensitic state",
                "backup": "literature default",
                "meaning": "Daughter martensite metric; controls λ₁/λ₂/λ₃ and d-spacings.",
                "warning": "β strongly affects monoclinic volume and compatibility.",
            },
            {
                "input": "Effective Ni at.%",
                "best source": "ICP-OES, EDS/WDS, or calibrated vaporization model",
                "backup": "nominal powder composition",
                "meaning": "Ni loss shifts transformation temperatures and phase state.",
                "warning": "LPBF can reduce Ni by selective evaporation.",
            },
            {
                "input": "Phase state",
                "best source": "XRD + DSC + EBSD/TKD",
                "backup": "XRD peak screening",
                "meaning": "Tells whether the checked condition is B2, B19′ or mixed.",
                "warning": "B2/B19′ peak overlap is common.",
            },
            {
                "input": "Variant/orientation evidence",
                "best source": "EBSD/TKD with parent reconstruction",
                "backup": "none",
                "meaning": "Needed for true variant-pair and junction-plane analysis.",
                "warning": "Without this, variant analysis is only a proxy.",
            },
        ]
    )

    st.dataframe(guide, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 2: Lattice metrics
# =============================================================================

with tab_metrics:
    st.subheader("Step 2 — Lattice compatibility metrics with meaning")

    metrics = crystallographic_metrics(B2a, b19_a, b19_b, b19_c, beta)

    l2_state, l2_short, l2_long = lambda2_verdict(metrics["abs_lambda2_minus_1"])
    vol_state, vol_text = volume_verdict(metrics["normalized_volume_change_pct"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("λ₂", f"{metrics['lambda2']:.5f}")
    m2.metric("|λ₂ − 1|", f"{metrics['abs_lambda2_minus_1']:.5f}")
    m3.metric("Normalized ΔV", f"{metrics['normalized_volume_change_pct']:+.2f}%")
    m4.metric("Max principal strain", f"{metrics['max_abs_principal_strain_pct']:.2f}%")
    m5.metric("Compatibility score", f"{metrics['compatibility_score_0_100']:.1f}/100")

    if l2_state in ["excellent", "good/moderate"]:
        st.success(f"{l2_short} {l2_long}")
    elif l2_state == "warning":
        st.warning(f"{l2_short} {l2_long}")
    else:
        st.error(f"{l2_short} {l2_long}")

    if vol_state in ["excellent", "acceptable"]:
        st.success(vol_text)
    elif vol_state == "warning":
        st.warning(vol_text)
    else:
        st.error(vol_text)

    st.markdown(
        """
### Why the volume is normalized

Do **not** directly compare `B2 a₀³` with the full B19′ monoclinic cell volume.
For a fair screening comparison, this page compares volume **per NiTi formula unit**
and also checks the B2 correspondence supercell volume against the B19′ cell.
"""
    )

    metric_table = pd.DataFrame(
        [
            {
                "metric": "B2 cell volume",
                "value": metrics["B2_cell_volume_A3"],
                "unit": "Å³",
                "meaning": "Conventional B2 cubic cell volume.",
                "interpretation": "Raw parent cell value. Not directly enough for transformation volume claim.",
            },
            {
                "metric": "B2 volume per NiTi",
                "value": metrics["B2_volume_per_NiTi_A3"],
                "unit": "Å³ / NiTi",
                "meaning": "Normalized parent volume per formula unit.",
                "interpretation": "Use this for fair B2/B19′ volume comparison.",
            },
            {
                "metric": "B19′ cell volume",
                "value": metrics["B19p_cell_volume_A3"],
                "unit": "Å³",
                "meaning": "Monoclinic B19′ conventional cell volume.",
                "interpretation": "Usually compared after normalizing per formula unit.",
            },
            {
                "metric": "B19′ volume per NiTi",
                "value": metrics["B19p_volume_per_NiTi_A3"],
                "unit": "Å³ / NiTi",
                "meaning": "B19′ volume normalized by two formula units.",
                "interpretation": "Use this against B2 volume per NiTi.",
            },
            {
                "metric": "Normalized volume change",
                "value": metrics["normalized_volume_change_pct"],
                "unit": "%",
                "meaning": "Change in volume per NiTi unit from B2 to B19′.",
                "interpretation": vol_text,
            },
            {
                "metric": "λ₁, λ₂, λ₃",
                "value": np.nan,
                "unit": "dimensionless",
                "meaning": "Principal stretches of the simplified transformation stretch.",
                "interpretation": "λ₂ close to 1 is the key compatibility proxy.",
            },
            {
                "metric": "|λ₂ − 1|",
                "value": metrics["abs_lambda2_minus_1"],
                "unit": "dimensionless",
                "meaning": "Distance from the cofactor/compatibility target λ₂ = 1.",
                "interpretation": l2_long,
            },
            {
                "metric": "det(F)",
                "value": metrics["detF"],
                "unit": "dimensionless",
                "meaning": "Volume ratio of the simplified correspondence deformation.",
                "interpretation": "Should be near the correspondence-volume ratio.",
            },
        ]
    )

    st.dataframe(metric_table, use_container_width=True, hide_index=True)

    strain_df = pd.DataFrame(
        {
            "axis": ["λ₁", "λ₂", "λ₃"],
            "stretch": [metrics["lambda1"], metrics["lambda2"], metrics["lambda3"]],
            "principal_strain_pct": [
                metrics["principal_strain_1_pct"],
                metrics["principal_strain_2_pct"],
                metrics["principal_strain_3_pct"],
            ],
        }
    )

    st.subheader("Principal transformation stretch / strain")
    st.plotly_chart(
        px.bar(
            strain_df,
            x="axis",
            y="principal_strain_pct",
            hover_data=["stretch"],
            title="Principal metric strain from simplified B2→B19′ correspondence",
        ),
        use_container_width=True,
    )
    st.dataframe(strain_df, use_container_width=True, hide_index=True)

    with st.expander("Formula and assumption used here"):
        st.markdown(
            """
The simplified correspondence basis is:

- parent B2 basis: `diag(a₀, √2a₀, √2a₀)`
- martensite B19′ basis: `[aₘ, bₘ, cₘ, β]`
- transformation gradient: `F = B19′ · inverse(B2_correspondence)`
- stretch eigenvalues: eigenvalues of `sqrt(FᵀF)`

This is useful for screening λ₂ and metric strain, but it is not a full PTMC or full correspondence-theory solver.
"""
        )


# =============================================================================
# Tab 3: Cayron correspondence / variants
# =============================================================================

with tab_ct:
    st.subheader("Step 3 — Cayron correspondence-theory layer")

    st.markdown(
        """
Cayron's correspondence theory is useful here because it does not treat variants as random labels.
For B2→B19′ NiTi, the theory organizes the transformation through:

- **parent B2 symmetry**
- **daughter B19′ symmetry**
- **correspondence variants**
- **orientation variants**
- **distortion variants**
- **operators between variants**
- **junction/twin elements inherited from parent symmetry**

For this dashboard, the important practical message is:

> Use correspondence theory to decide what variant pairs and junctions are crystallographically meaningful, but do not claim exact variant indexing unless EBSD/TKD data and parent reconstruction are available.
"""
    )

    ct_summary = pd.DataFrame(
        [
            {
                "concept": "Parent B2 symmetry",
                "value used in CT literature": "48 symmetry operations",
                "meaning in app": "The parent symmetry generates possible daughter variants.",
                "what user needs": "Parent orientation from EBSD/TKD for real indexing.",
            },
            {
                "concept": "Intersection subgroup",
                "value used in CT literature": "4 common symmetries",
                "meaning in app": "Controls number of variants.",
                "what user needs": "Full symmetry/correspondence calculation for exact work.",
            },
            {
                "concept": "Number of variants",
                "value used in CT literature": "12",
                "meaning in app": "B2→B19′ has 12 correspondence/orientation/distortion variant families.",
                "what user needs": "Measured martensite orientations to assign V1–V12.",
            },
            {
                "concept": "Number of operators",
                "value used in CT literature": "7",
                "meaning in app": "Variant pairs fall into 7 operator families.",
                "what user needs": "Misorientation clusters and boundary traces.",
            },
            {
                "concept": "Ambivalent operators",
                "value used in CT literature": "type-I/type-II/compound twin logic possible",
                "meaning in app": "Junction elements can be inherited from parent mirror or 2-fold symmetry.",
                "what user needs": "EBSD/TKD trace validation.",
            },
            {
                "concept": "Polar operators",
                "value used in CT literature": "weak-plane logic may be needed",
                "meaning in app": "Do not force normal PTMC twin solution.",
                "what user needs": "Advanced CT/TKD analysis.",
            },
        ]
    )

    st.dataframe(ct_summary, use_container_width=True, hide_index=True)

    st.subheader("Operator-family table for interpretation")
    op_df = cayron_operator_table()
    st.dataframe(op_df, use_container_width=True, hide_index=True)

    st.subheader("Natural orientation relationship / habit-plane notes from project defaults")

    st.code(
        f"{NITI_DEFAULTS.get('natural_OR_planes', 'No OR plane default available.')}\n"
        f"{NITI_DEFAULTS.get('natural_OR_directions', 'No OR direction default available.')}\n"
        f"Candidate habit plane: {NITI_DEFAULTS.get('candidate_habit_plane', 'No habit-plane default available.')}",
        language="text",
    )

    st.warning(
        "These OR/habit-plane notes are only useful as a guide. For a real claim, extract the OR, parent B2 orientation, "
        "daughter B19′ orientations and boundary traces from EBSD/TKD."
    )

    st.subheader("Variant-pair screening")

    if HAS_PROJECT_VARIANT_FUNCTIONS:
        try:
            variants = generate_simplified_b19_variants()
            pair_df = pd.DataFrame(variant_pair_table(variants))
            st.success("Using project src.crystallography variant functions.")
        except Exception as exc:
            pair_df = fallback_variant_pairs()
            st.warning(f"Project variant functions failed, using fallback educational table. Error: {exc}")
    else:
        pair_df = fallback_variant_pairs()
        st.info("Project variant functions not available. Using fallback educational CT-family table.")

    if "compatibility_proxy" in pair_df.columns:
        pair_show = pair_df.sort_values("compatibility_proxy", ascending=False)
    else:
        pair_show = pair_df

    c_left, c_right = st.columns([1.0, 1.0])

    with c_left:
        st.markdown("#### Top variant-pair families")
        st.dataframe(pair_show.head(20), use_container_width=True, hide_index=True)

    with c_right:
        if {"misorientation_deg", "compatibility_proxy"}.issubset(pair_df.columns):
            fig_pair = px.scatter(
                pair_df,
                x="misorientation_deg",
                y="compatibility_proxy",
                hover_data=[c for c in ["variant_i", "variant_j"] if c in pair_df.columns],
                title="Variant-pair proxy map",
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        elif "compatibility_proxy" in pair_df.columns:
            fig_pair = px.histogram(
                pair_df,
                x="compatibility_proxy",
                nbins=20,
                title="Distribution of variant-pair compatibility proxy",
            )
            st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("No numeric pair score available for plotting.")

    st.subheader("Build/loading-axis alignment proxy")

    axis_choice = st.selectbox(
        "Axis for build/loading proxy",
        ["Z/build", "X", "Y", "Custom diagonal [111]"],
        key="ct_axis_choice",
    )

    if HAS_PROJECT_VARIANT_FUNCTIONS:
        try:
            axis_map = {
                "Z/build": (0, 0, 1),
                "X": (1, 0, 0),
                "Y": (0, 1, 0),
                "Custom diagonal [111]": (1, 1, 1),
            }
            variants = generate_simplified_b19_variants()
            stress_df = pd.DataFrame(stress_variant_scores(variants, axis_map[axis_choice]))
            st.success("Using project stress_variant_scores().")
        except Exception as exc:
            stress_df = stress_axis_proxy_table(axis_choice)
            st.warning(f"Project stress score failed, using geometric proxy. Error: {exc}")
    else:
        stress_df = stress_axis_proxy_table(axis_choice)

    st.dataframe(stress_df, use_container_width=True, hide_index=True)

    st.info(
        "This is not true transformation-work ranking. True stress-assisted variant selection needs a stress tensor, "
        "parent grain orientation, transformation strain tensor, and measured variant orientations."
    )


# =============================================================================
# Tab 4: XRD support
# =============================================================================

with tab_xrd:
    st.subheader("Step 4 — XRD support from selected lattice parameters")

    st.markdown(
        """
This section calculates approximate B2 and B19′ peak positions from the lattice parameters.
Use it to compare against your measured XRD pattern.

It does **not** perform Rietveld refinement. It only tells you where major B2/B19′ peaks
should roughly appear.
"""
    )

    wavelength = st.number_input(
        "X-ray wavelength λ (Å)",
        value=CU_K_ALPHA_A,
        step=0.0001,
        format="%.4f",
        key="xrd_support_wavelength",
    )

    refs = reference_peaks(B2a, b19_a, b19_b, b19_c, beta, wavelength)

    phase_filter = st.multiselect(
        "Show phases",
        sorted(refs["phase"].unique()),
        default=sorted(refs["phase"].unique()),
        key="xrd_phase_filter",
    )

    refs_view = refs[refs["phase"].isin(phase_filter)].copy()

    st.dataframe(refs_view, use_container_width=True, hide_index=True)

    fig_xrd = px.scatter(
        refs_view,
        x="two_theta_deg",
        y="phase",
        hover_data=["hkl", "d_A", "meaning"],
        title="Calculated B2/B19′/warning peak positions",
    )
    st.plotly_chart(fig_xrd, use_container_width=True)

    st.subheader("Single hkl calculator")

    c1, c2, c3, c4, c5 = st.columns(5)
    calc_phase = c1.selectbox("Phase", ["B2 cubic", "B19′ monoclinic"], key="single_calc_phase")
    h = c2.number_input("h", value=1, step=1, key="single_h")
    k = c3.number_input("k", value=1, step=1, key="single_k")
    l = c4.number_input("l", value=0, step=1, key="single_l")
    calc_lambda = c5.number_input("λ calc (Å)", value=wavelength, step=0.0001, format="%.4f", key="single_lambda")

    if calc_phase == "B2 cubic":
        d_calc = cubic_d_spacing(B2a, int(h), int(k), int(l))
    else:
        d_calc = monoclinic_d_spacing(b19_a, b19_b, b19_c, beta, int(h), int(k), int(l))

    tt_calc = two_theta_from_d(d_calc, calc_lambda)

    dcol, ttcol = st.columns(2)
    dcol.metric("d-spacing", f"{d_calc:.4f} Å" if np.isfinite(d_calc) else "invalid")
    ttcol.metric("2θ", f"{tt_calc:.3f}°" if np.isfinite(tt_calc) else "invalid")

    st.markdown(
        """
### How to use this in the project

- If XRD shows mostly B2 peaks at service temperature, that supports superelastic design.
- If B19′ peaks dominate at room/service temperature, connect that to Ni loss, Af/Ms shift or heat treatment.
- If secondary/oxide warning peaks appear, check oxygen, Ti-rich/Ni-rich intermetallics and EDS.
- If peaks overlap, do not overclaim phase fraction from this page alone.
"""
    )


# =============================================================================
# Tab 5: Final decision
# =============================================================================

with tab_decision:
    st.subheader("Step 5 — Final crystallography decision report")

    metrics = crystallographic_metrics(B2a, b19_a, b19_b, b19_c, beta)

    headline, explanation, evidence_table = build_final_verdict(
        quality_pct,
        metrics,
        phase_evidence,
        dsc_evidence,
        defect_evidence,
    )

    st.markdown(f"## {headline}")

    if "Strong" in headline:
        st.success(explanation)
    elif "Promising" in headline:
        st.info(explanation)
    elif "warning" in headline.lower():
        st.warning(explanation)
    else:
        st.error(explanation)

    report_metrics = pd.DataFrame(
        [
            {"quantity": "Evidence quality", "value": f"{quality_pct:.0f}/100", "meaning": q_text},
            {"quantity": "Effective Ni", "value": f"{effective_ni:.2f} at.%", "meaning": "Connect this to vaporization-induced composition shift."},
            {"quantity": "Service temperature", "value": f"{service_temperature:.1f} °C", "meaning": "Compare with DSC Af/Ms."},
            {"quantity": "Phase evidence", "value": phase_evidence, "meaning": "Determines whether B2/B19′ state supports intended function."},
            {"quantity": "λ₂", "value": f"{metrics['lambda2']:.5f}", "meaning": "Middle principal stretch; closer to 1 is better compatibility proxy."},
            {"quantity": "|λ₂ − 1|", "value": f"{metrics['abs_lambda2_minus_1']:.5f}", "meaning": lambda2_verdict(metrics["abs_lambda2_minus_1"])[2]},
            {"quantity": "Normalized volume change", "value": f"{metrics['normalized_volume_change_pct']:+.2f}%", "meaning": volume_verdict(metrics["normalized_volume_change_pct"])[1]},
            {"quantity": "Max principal strain", "value": f"{metrics['max_abs_principal_strain_pct']:.2f}%", "meaning": "High strain can increase mismatch/hysteresis risk."},
            {"quantity": "Compatibility score", "value": f"{metrics['compatibility_score_0_100']:.1f}/100", "meaning": "Combined screening score, not a measured property."},
        ]
    )

    st.subheader("Report metrics with meaning")
    st.dataframe(report_metrics, use_container_width=True, hide_index=True)

    st.subheader("Evidence interpretation")
    st.dataframe(evidence_table, use_container_width=True, hide_index=True)

    st.subheader("Copy-ready paragraph")

    copy_text = f"""
The crystallography screening used B2 a₀ = {B2a:.4f} Å and B19′ lattice parameters
aₘ = {b19_a:.4f} Å, bₘ = {b19_b:.4f} Å, cₘ = {b19_c:.4f} Å and β = {beta:.3f}°.
The normalized B2→B19′ volume change was {metrics['normalized_volume_change_pct']:+.2f}%.
The simplified transformation-stretch calculation gave λ₂ = {metrics['lambda2']:.5f},
with |λ₂ − 1| = {metrics['abs_lambda2_minus_1']:.5f}. This gives the interpretation:
{lambda2_verdict(metrics['abs_lambda2_minus_1'])[2]} The evidence-quality score was
{quality_pct:.0f}/100, so the result should be treated as {q_label} confidence. The
correspondence-theory section should be used as a variant-family guide only unless EBSD/TKD
parent reconstruction and measured B19′ variant orientations are available.
""".strip()

    st.code(copy_text, language="text")

    st.subheader("Next experimental actions")

    next_actions = pd.DataFrame(
        [
            {
                "priority": 1,
                "action": "Refine B2 and B19′ lattice parameters by XRD/Rietveld or Pawley fitting.",
                "why": "Defaults are not enough for strong λ₂ or volume-change claims.",
            },
            {
                "priority": 2,
                "action": "Measure Ms, Mf, As and Af by DSC.",
                "why": "Phase state at service temperature controls superelastic/shape-memory behavior.",
            },
            {
                "priority": 3,
                "action": "Measure printed-part Ni/Ti by ICP/EDS/WDS.",
                "why": "LPBF Ni evaporation shifts transformation temperatures and lattice state.",
            },
            {
                "priority": 4,
                "action": "Acquire EBSD/TKD maps and reconstruct parent B2 orientation.",
                "why": "Needed for real Cayron-style variant/operator and junction-plane analysis.",
            },
            {
                "priority": 5,
                "action": "Compare boundary traces with predicted correspondence operator families.",
                "why": "This converts the page from a proxy into real variant-pair evidence.",
            },
            {
                "priority": 6,
                "action": "Run cyclic superelastic or shape-memory tests.",
                "why": "Compatibility metrics support the claim, but function must be mechanically validated.",
            },
        ]
    )

    st.dataframe(next_actions, use_container_width=True, hide_index=True)

    export = pd.concat(
        [
            pd.DataFrame([metrics]),
            pd.DataFrame(
                [
                    {
                        "effective_Ni_at_pct": effective_ni,
                        "service_temperature_C": service_temperature,
                        "evidence_quality_pct": quality_pct,
                        "quality_label": q_label,
                        "headline": headline,
                    }
                ]
            ),
        ],
        axis=1,
    )

    st.download_button(
        "Download crystallography screening result as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="niti_crystallography_compatibility_result.csv",
        mime="text/csv",
    )
