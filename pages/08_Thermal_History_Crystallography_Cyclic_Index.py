import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="Thermal History – Crystallography – Cyclic Function Index",
    layout="wide",
)

st.title("Thermal History – Crystallography – Cyclic Function Index")
st.caption(
    "A mechanism-based NiTi-AM index linking LPBF thermal history, B2/B19′ phase state, "
    "crystallographic compatibility, residual strain, martensite stabilization, and cyclic superelastic stability."
)

st.markdown(
    """
This page is built around one key NiTi-AM idea:

> **Good NiTi is not only dense NiTi. Good NiTi needs the right thermal history, phase state, crystallography, and cyclic reversibility.**

The interlayer-remelting study on EBF³ NiTi shows a general mechanism that is also useful for LPBF:
repeated thermal exposure changes microstructure, crystallographic orientation, grain boundaries,
residual strain, B2/B19′ transformation behavior, and superelastic degradation.  
For LPBF, the equivalent thermal-history sources are **rescanning, hatch overlap, contour remelting,
layer reheating, local heat accumulation, oxygen pickup, and powder reuse**.

This page gives a transparent screening index. It does **not** replace DSC, XRD/Rietveld, ICP/EDS,
EBSD/TKD, or cyclic mechanical testing.
"""
)


# =============================================================================
# Helper functions
# =============================================================================

def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def risk_between(x: float, low_good: float, high_bad: float) -> float:
    """
    0 = good/low-risk, 100 = high-risk.
    """
    if high_bad == low_good:
        return 0.0
    return clamp(100.0 * (x - low_good) / (high_bad - low_good))


def inverse_score_from_risk(risk: float) -> float:
    return clamp(100.0 - risk)


def bell_score(x: float, best_low: float, best_high: float, too_low: float, too_high: float) -> float:
    """
    Score is high inside [best_low, best_high], decreases outside.
    Used for stress-induced martensite temperature window.
    """
    if best_low <= x <= best_high:
        return 100.0
    if x < best_low:
        return clamp(100.0 * (x - too_low) / (best_low - too_low))
    return clamp(100.0 * (too_high - x) / (too_high - best_high))


def label_risk(risk: float) -> Tuple[str, str]:
    if risk < 25:
        return "low", "This factor is not currently dominating the risk."
    if risk < 50:
        return "moderate", "This factor may matter; verify with measurement."
    if risk < 75:
        return "high", "This factor is a serious warning."
    return "critical", "This factor can dominate or destroy the functional response."


def label_score(score: float) -> Tuple[str, str]:
    if score >= 80:
        return "strong candidate", "The combined evidence supports a potentially stable functional NiTi condition."
    if score >= 65:
        return "promising", "The condition is promising, but experimental confirmation is still needed."
    if score >= 45:
        return "mixed / unstable", "The condition may work, but the risk of unstable or degraded superelasticity is significant."
    return "poor / high-risk", "The condition is unlikely to give reliable cyclic superelasticity without correction."


def evidence_weight(source: str) -> int:
    values = {
        "Measured directly": 100,
        "XRD/Rietveld or DSC measured": 90,
        "EBSD/TKD measured": 90,
        "ICP/EDS/WDS measured": 90,
        "Measured but approximate": 70,
        "Estimated from model": 55,
        "Literature/default": 35,
        "Assumed / unknown": 15,
    }
    return values.get(source, 15)


def quality_label(q: float) -> Tuple[str, str]:
    if q >= 80:
        return "high", "Most inputs are measured or strongly supported."
    if q >= 60:
        return "medium", "The index is useful for screening, but some key inputs still need measurement."
    if q >= 40:
        return "low", "Use this only as a hypothesis generator."
    return "very low", "Do not make strong claims from this result."


def compute_ved(power_W: float, speed_mm_s: float, hatch_mm: float, layer_mm: float) -> float:
    if speed_mm_s <= 0 or hatch_mm <= 0 or layer_mm <= 0:
        return np.nan
    return power_W / (speed_mm_s * hatch_mm * layer_mm)


def compute_led(power_W: float, speed_mm_s: float) -> float:
    if speed_mm_s <= 0:
        return np.nan
    return power_W / speed_mm_s


def compute_indices(inputs: Dict[str, float]) -> Dict[str, float]:
    P = inputs["power_W"]
    v = inputs["speed_mm_s"]
    h = inputs["hatch_mm"]
    t = inputs["layer_mm"]

    ved = compute_ved(P, v, h, t)
    led = compute_led(P, v)

    # -------------------------------------------------------------------------
    # 1. Thermal Cycle Severity Index
    # -------------------------------------------------------------------------
    ved_thermal = risk_between(ved, 55.0, 170.0)
    led_thermal = risk_between(led, 0.10, 0.45)
    remelt_risk = risk_between(inputs["remelt_passes"], 0.0, 4.0)
    overlap_risk = risk_between(inputs["hatch_overlap_pct"], 10.0, 70.0)
    reheat_risk = inputs["layer_reheat_severity_pct"]
    buildplate_risk = risk_between(inputs["build_plate_C"], 50.0, 450.0)
    layer_thin_risk = risk_between(0.080 - inputs["layer_mm"], 0.000, 0.060)

    thermal_cycle_index = clamp(
        0.26 * ved_thermal
        + 0.18 * led_thermal
        + 0.20 * remelt_risk
        + 0.13 * overlap_risk
        + 0.11 * reheat_risk
        + 0.06 * buildplate_risk
        + 0.06 * layer_thin_risk
    )

    # Lack-of-fusion is a separate low-energy warning.
    lack_of_fusion_risk = clamp(
        0.60 * risk_between(55.0 - ved, 0.0, 35.0)
        + 0.40 * risk_between(0.12 - led, 0.0, 0.10)
    )

    # -------------------------------------------------------------------------
    # 2. Oxygen / oxy-precipitate risk
    # -------------------------------------------------------------------------
    oxygen_risk = risk_between(inputs["oxygen_ppm"], 50.0, 800.0)
    reuse_risk = risk_between(inputs["powder_reuse_cycles"], 0.0, 8.0)
    secondary_phase_risk = inputs["secondary_phase_signal_pct"]

    oxide_precipitate_risk = clamp(
        0.42 * oxygen_risk
        + 0.18 * reuse_risk
        + 0.18 * secondary_phase_risk
        + 0.14 * thermal_cycle_index
        + 0.08 * risk_between(inputs["defect_fraction_pct"], 0.2, 3.0)
    )

    # -------------------------------------------------------------------------
    # 3. Phase suitability and martensite-at-service risk
    # -------------------------------------------------------------------------
    b2_fraction = inputs["b2_fraction_pct"]
    b19_fraction = inputs["b19_fraction_pct"]
    retained_b19 = inputs["retained_b19_after_cycling_pct"]

    service_T = inputs["service_temperature_C"]
    Ms = inputs["Ms_C"]
    Af = inputs["Af_C"]

    if service_T < Ms:
        martensite_at_service_risk = 100.0
    elif service_T < Af:
        martensite_at_service_risk = 65.0
    else:
        martensite_at_service_risk = risk_between(Af - service_T, -80.0, 0.0)

    phase_suitability_score = clamp(
        0.55 * b2_fraction
        + 0.25 * inverse_score_from_risk(b19_fraction)
        + 0.20 * inverse_score_from_risk(martensite_at_service_risk)
    )

    # -------------------------------------------------------------------------
    # 4. Residual strain / dislocation / local distortion risk
    # -------------------------------------------------------------------------
    residual_strain_risk = risk_between(inputs["residual_strain_after_cycles_pct"], 0.2, 4.0)
    kam_mean_risk = risk_between(inputs["mean_KAM_deg"], 0.8, 5.0)
    high_kam_risk = inputs["high_KAM_fraction_pct"]
    defect_risk = risk_between(inputs["defect_fraction_pct"], 0.2, 4.0)

    residual_distortion_risk = clamp(
        0.34 * residual_strain_risk
        + 0.22 * kam_mean_risk
        + 0.18 * high_kam_risk
        + 0.14 * retained_b19
        + 0.12 * defect_risk
    )

    # -------------------------------------------------------------------------
    # 5. B19′ stabilization risk
    # -------------------------------------------------------------------------
    dsc_width_risk = risk_between(inputs["DSC_peak_width_C"], 8.0, 55.0)
    hysteresis_risk = risk_between(inputs["hysteresis_width_C"], 15.0, 75.0)

    b19_stabilization_risk = clamp(
        0.22 * b19_fraction
        + 0.22 * retained_b19
        + 0.17 * residual_distortion_risk
        + 0.15 * dsc_width_risk
        + 0.12 * thermal_cycle_index
        + 0.12 * martensite_at_service_risk
    )

    # -------------------------------------------------------------------------
    # 6. Transformation broadening risk
    # -------------------------------------------------------------------------
    ni_deviation_risk = risk_between(abs(inputs["effective_Ni_at_pct"] - 50.8), 0.05, 0.80)

    transformation_broadening_risk = clamp(
        0.32 * dsc_width_risk
        + 0.24 * hysteresis_risk
        + 0.16 * thermal_cycle_index
        + 0.12 * residual_distortion_risk
        + 0.10 * ni_deviation_risk
        + 0.06 * oxide_precipitate_risk
    )

    # -------------------------------------------------------------------------
    # 7. Stress-induced martensite window / sigma_Ms proxy
    # -------------------------------------------------------------------------
    delta_T_Ms = service_T - Ms
    sigma_slope = inputs["clausius_clapeyron_slope_MPa_C"]
    sigma_Ms_proxy = inputs["sigma_Ms_at_Ms_MPa"] + sigma_slope * max(0.0, delta_T_Ms)

    # In superelastic design, service temperature should usually be above Af,
    # but not so far above Ms that sigma_Ms becomes too high and plasticity risk increases.
    stress_window_score = bell_score(delta_T_Ms, best_low=25.0, best_high=110.0, too_low=-20.0, too_high=220.0)

    if service_T < Af:
        stress_window_score *= 0.55

    plasticity_during_loading_risk = clamp(
        0.55 * risk_between(sigma_Ms_proxy, 350.0, 850.0)
        + 0.25 * residual_distortion_risk
        + 0.20 * defect_risk
    )

    # -------------------------------------------------------------------------
    # 8. Crystallographic compatibility / Cayron-style correspondence support
    # -------------------------------------------------------------------------
    lambda2_error = inputs["lambda2_error"]
    lambda2_risk = risk_between(lambda2_error, 0.005, 0.035)
    volume_mismatch_risk = risk_between(abs(inputs["normalized_volume_change_pct"]), 0.5, 3.0)
    variant_alignment_score = inputs["texture_variant_alignment_pct"]

    crystallography_score = clamp(
        0.42 * inverse_score_from_risk(lambda2_risk)
        + 0.25 * inverse_score_from_risk(volume_mismatch_risk)
        + 0.20 * variant_alignment_score
        + 0.13 * inverse_score_from_risk(kam_mean_risk)
    )

    # -------------------------------------------------------------------------
    # 9. Cyclic degradation risk
    # -------------------------------------------------------------------------
    cyclic_degradation_risk = clamp(
        0.26 * b19_stabilization_risk
        + 0.23 * residual_distortion_risk
        + 0.16 * transformation_broadening_risk
        + 0.13 * plasticity_during_loading_risk
        + 0.10 * oxide_precipitate_risk
        + 0.08 * defect_risk
        + 0.04 * lack_of_fusion_risk
    )

    # -------------------------------------------------------------------------
    # 10. Final functional index
    # -------------------------------------------------------------------------
    cyclic_function_score = clamp(
        0.19 * phase_suitability_score
        + 0.17 * stress_window_score
        + 0.16 * crystallography_score
        + 0.15 * inverse_score_from_risk(cyclic_degradation_risk)
        + 0.12 * inverse_score_from_risk(b19_stabilization_risk)
        + 0.10 * inverse_score_from_risk(transformation_broadening_risk)
        + 0.07 * inverse_score_from_risk(oxide_precipitate_risk)
        + 0.04 * inverse_score_from_risk(lack_of_fusion_risk)
    )

    return {
        "VED_J_mm3": ved,
        "LED_J_mm": led,
        "thermal_cycle_index": thermal_cycle_index,
        "lack_of_fusion_risk": lack_of_fusion_risk,
        "oxide_precipitate_risk": oxide_precipitate_risk,
        "phase_suitability_score": phase_suitability_score,
        "martensite_at_service_risk": martensite_at_service_risk,
        "residual_distortion_risk": residual_distortion_risk,
        "b19_stabilization_risk": b19_stabilization_risk,
        "transformation_broadening_risk": transformation_broadening_risk,
        "delta_T_service_minus_Ms_C": delta_T_Ms,
        "sigma_Ms_proxy_MPa": sigma_Ms_proxy,
        "stress_window_score": stress_window_score,
        "plasticity_during_loading_risk": plasticity_during_loading_risk,
        "lambda2_risk": lambda2_risk,
        "volume_mismatch_risk": volume_mismatch_risk,
        "crystallography_score": crystallography_score,
        "cyclic_degradation_risk": cyclic_degradation_risk,
        "cyclic_function_score": cyclic_function_score,
    }


def build_factor_table(inputs: Dict[str, float], outputs: Dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "factor / index": "VED",
            "value": f"{outputs['VED_J_mm3']:.1f} J/mm³",
            "what it means": "Volumetric energy density. Useful first descriptor, but not enough alone for NiTi.",
            "how to get input": "From LPBF parameters: P / (v × hatch × layer thickness).",
            "if high": "Can increase remelting, heat accumulation, keyhole/evaporation and Ni loss risk.",
            "if low": "Can cause lack of fusion, poor bonding and porosity.",
            "current outcome": label_risk(risk_between(outputs["VED_J_mm3"], 55, 170))[0],
        },
        {
            "factor / index": "Thermal Cycle Severity Index",
            "value": f"{outputs['thermal_cycle_index']:.1f}/100",
            "what it means": "Repeated thermal exposure from energy input, rescanning, overlap, layer reheating and thin layers.",
            "how to get input": "Use scan strategy, remelt/rescan count, hatch overlap, layer thickness and build temperature.",
            "if high": "Higher risk of residual strain, precipitate/oxide evolution, retained B19′ and cyclic degradation.",
            "if low": "Less thermal damage, but may still suffer lack of fusion if energy is too low.",
            "current outcome": label_risk(outputs["thermal_cycle_index"])[0],
        },
        {
            "factor / index": "Oxy-precipitate risk",
            "value": f"{outputs['oxide_precipitate_risk']:.1f}/100",
            "what it means": "Risk that oxygen, powder reuse and thermal exposure promote oxide/oxy-precipitate effects such as Ti₄Ni₂Oₓ-like particles.",
            "how to get input": "Measure chamber oxygen, powder oxygen, powder reuse, XRD/EDS secondary signals.",
            "if high": "Transformation may be pinned; matrix composition may shift; cyclic recovery may degrade.",
            "if low": "Cleaner matrix and lower precipitation/oxide-related transformation disturbance.",
            "current outcome": label_risk(outputs["oxide_precipitate_risk"])[0],
        },
        {
            "factor / index": "B19′ stabilization risk",
            "value": f"{outputs['b19_stabilization_risk']:.1f}/100",
            "what it means": "Risk that martensite remains or becomes stabilized after thermal/cyclic history.",
            "how to get input": "Use XRD/EBSD B19′ fraction, retained B19′ after unloading, DSC width and KAM/residual strain.",
            "if high": "Poor superelastic recovery; residual strain; broad or incomplete reverse transformation.",
            "if low": "Better chance of reversible B2 ↔ B19′ transformation.",
            "current outcome": label_risk(outputs["b19_stabilization_risk"])[0],
        },
        {
            "factor / index": "Residual distortion risk",
            "value": f"{outputs['residual_distortion_risk']:.1f}/100",
            "what it means": "Risk from dislocation pile-ups, plastic deformation, local misorientation and retained strain.",
            "how to get input": "Cyclic residual strain, EBSD/TKD KAM, high-KAM fraction and defect fraction.",
            "if high": "Lamellar martensite can broaden/stabilize; superelastic loops deteriorate.",
            "if low": "Lower internal strain barrier for reversible transformation.",
            "current outcome": label_risk(outputs["residual_distortion_risk"])[0],
        },
        {
            "factor / index": "Transformation broadening risk",
            "value": f"{outputs['transformation_broadening_risk']:.1f}/100",
            "what it means": "Risk that transformation is spread over a wide temperature/stress range due to heterogeneity.",
            "how to get input": "DSC peak width, hysteresis width, composition variation, KAM and phase spread.",
            "if high": "Less sharp reversible transformation; unstable phase response and larger hysteresis.",
            "if low": "Cleaner and more reversible transformation behavior.",
            "current outcome": label_risk(outputs["transformation_broadening_risk"])[0],
        },
        {
            "factor / index": "σMs proxy",
            "value": f"{outputs['sigma_Ms_proxy_MPa']:.0f} MPa",
            "what it means": "Estimated stress needed to trigger stress-induced martensite at service temperature.",
            "how to get input": "Use Ms, service temperature and an approximate Clausius–Clapeyron slope.",
            "if high": "Plasticity may occur before clean stress-induced transformation.",
            "if low": "Martensite may form easily, but if too close to Ms the phase state may be unstable.",
            "current outcome": f"ΔT = {outputs['delta_T_service_minus_Ms_C']:.1f} °C",
        },
        {
            "factor / index": "Crystallography score",
            "value": f"{outputs['crystallography_score']:.1f}/100",
            "what it means": "Compatibility score from λ₂, normalized volume mismatch, texture/variant alignment and KAM.",
            "how to get input": "Use XRD-refined lattice parameters, λ₂ from crystallography page, EBSD/TKD texture and KAM.",
            "if high": "Better B2/B19′ compatibility and variant accommodation.",
            "if low": "Mismatch, poor variant accommodation and hysteresis risk.",
            "current outcome": label_score(outputs["crystallography_score"])[0],
        },
        {
            "factor / index": "Cyclic degradation risk",
            "value": f"{outputs['cyclic_degradation_risk']:.1f}/100",
            "what it means": "Combined risk that repeated loading/unloading will degrade superelasticity.",
            "how to get input": "Needs cyclic mechanical testing, retained martensite, KAM, DSC broadening and defect data.",
            "if high": "Expect residual strain, reduced recoverable strain and stabilized martensite.",
            "if low": "Better chance of stable superelastic loops.",
            "current outcome": label_risk(outputs["cyclic_degradation_risk"])[0],
        },
        {
            "factor / index": "Final cyclic function score",
            "value": f"{outputs['cyclic_function_score']:.1f}/100",
            "what it means": "Overall estimate of functional NiTi suitability after processing and cycling.",
            "how to get input": "Combines process, phase, DSC, crystallography, EBSD/TKD, defects and cyclic evidence.",
            "if high": "Strong superelastic/shape-memory candidate.",
            "if low": "High chance of unstable or degraded functional response.",
            "current outcome": label_score(outputs["cyclic_function_score"])[0],
        },
    ]

    return pd.DataFrame(rows)


def build_action_table(outputs: Dict[str, float]) -> pd.DataFrame:
    actions = []

    if outputs["lack_of_fusion_risk"] > 55:
        actions.append(
            {
                "priority": "high",
                "problem": "Low-energy / lack-of-fusion risk",
                "what to do": "Increase energy input carefully, reduce hatch spacing, or improve overlap. Verify with cross-section porosity.",
                "measurement to confirm": "SEM/optical cross-section, density, lack-of-fusion morphology.",
            }
        )

    if outputs["thermal_cycle_index"] > 65:
        actions.append(
            {
                "priority": "high",
                "problem": "Thermal-cycle severity is high",
                "what to do": "Reduce unnecessary rescans/remelting, optimize hatch overlap, avoid excessive heat accumulation.",
                "measurement to confirm": "XRD/DSC phase stability, EBSD KAM, residual strain after cycling.",
            }
        )

    if outputs["oxide_precipitate_risk"] > 55:
        actions.append(
            {
                "priority": "high",
                "problem": "Oxygen / oxy-precipitate risk",
                "what to do": "Check powder oxygen, chamber oxygen, powder reuse, and Ti-rich/oxide signals.",
                "measurement to confirm": "EDS/WDS oxygen mapping, XRD secondary peaks, TEM/SEM particle analysis.",
            }
        )

    if outputs["b19_stabilization_risk"] > 50:
        actions.append(
            {
                "priority": "high",
                "problem": "B19′ stabilization risk",
                "what to do": "Check whether martensite remains after unloading/cycling; tune Ni content and heat treatment.",
                "measurement to confirm": "XRD before/after cycling, EBSD/TKD phase maps, DSC after cycling.",
            }
        )

    if outputs["transformation_broadening_risk"] > 50:
        actions.append(
            {
                "priority": "medium",
                "problem": "Broad transformation risk",
                "what to do": "Reduce composition/thermal heterogeneity; measure DSC peak widths and hysteresis.",
                "measurement to confirm": "DSC Ms/Mf/As/Af and peak width, XRD phase fraction vs temperature.",
            }
        )

    if outputs["residual_distortion_risk"] > 50:
        actions.append(
            {
                "priority": "medium",
                "problem": "Residual strain / local misorientation risk",
                "what to do": "Modify heat treatment or process to reduce dislocation pile-ups and local strain.",
                "measurement to confirm": "EBSD/TKD KAM, cyclic residual strain, microstructure after cycling.",
            }
        )

    if outputs["crystallography_score"] < 55:
        actions.append(
            {
                "priority": "medium",
                "problem": "Weak crystallographic compatibility",
                "what to do": "Refine lattice parameters, calculate λ₂ carefully, and check variant accommodation by EBSD/TKD.",
                "measurement to confirm": "XRD/Rietveld lattice parameters, EBSD/TKD parent reconstruction.",
            }
        )

    if not actions:
        actions.append(
            {
                "priority": "continue validation",
                "problem": "No single factor dominates the risk",
                "what to do": "Proceed to validation: DSC, XRD, composition, EBSD/TKD and cyclic mechanical testing.",
                "measurement to confirm": "Cyclic superelastic loop stability and recoverable strain.",
            }
        )

    return pd.DataFrame(actions)


# =============================================================================
# Sidebar inputs
# =============================================================================

with st.sidebar:
    st.header("LPBF process inputs")

    power_W = st.slider("Laser power P (W)", 40, 500, 140, 5, key="idx_power")
    speed_mm_s = st.slider("Scan speed v (mm/s)", 100, 2500, 900, 10, key="idx_speed")
    hatch_mm = st.slider("Hatch spacing h (mm)", 0.03, 0.18, 0.08, 0.005, key="idx_hatch")
    layer_mm = st.slider("Layer thickness t (mm)", 0.01, 0.08, 0.03, 0.005, key="idx_layer")

    st.header("Thermal-history inputs")

    remelt_passes = st.slider("Remelt/rescan passes", 0, 6, 1, 1, key="idx_remelt")
    hatch_overlap_pct = st.slider("Hatch/track overlap (%)", 0, 90, 35, 1, key="idx_overlap")
    layer_reheat_severity_pct = st.slider("Layer reheating severity (%)", 0, 100, 35, 1, key="idx_reheat")
    build_plate_C = st.slider("Build plate temperature (°C)", 20, 600, 100, 10, key="idx_buildplate")

    st.header("Composition / atmosphere")

    effective_Ni_at_pct = st.slider("Effective Ni after printing (at.%)", 49.0, 52.5, 51.0, 0.01, key="idx_ni")
    oxygen_ppm = st.slider("Oxygen level / powder oxygen proxy (ppm)", 1, 1200, 120, 1, key="idx_oxygen")
    powder_reuse_cycles = st.slider("Powder reuse cycles", 0, 12, 1, 1, key="idx_reuse")
    secondary_phase_signal_pct = st.slider("Secondary/oxide phase signal (%)", 0, 100, 15, 1, key="idx_secondary")

    st.header("Transformation inputs")

    service_temperature_C = st.slider("Service/test temperature (°C)", -100, 200, 25, 1, key="idx_service")
    Ms_C = st.slider("Ms (°C)", -120, 150, -20, 1, key="idx_Ms")
    Af_C = st.slider("Af (°C)", -100, 180, 5, 1, key="idx_Af")
    DSC_peak_width_C = st.slider("DSC peak width / transformation spread (°C)", 2, 100, 25, 1, key="idx_dscwidth")
    hysteresis_width_C = st.slider("Transformation hysteresis width (°C)", 5, 120, 35, 1, key="idx_hysteresis")

    st.header("Phase / cycling evidence")

    b2_fraction_pct = st.slider("B2/austenite fraction at service (%)", 0, 100, 80, 1, key="idx_b2")
    b19_fraction_pct = st.slider("B19′ martensite fraction at service (%)", 0, 100, 20, 1, key="idx_b19")
    retained_b19_after_cycling_pct = st.slider("Retained B19′ after unloading/cycling (%)", 0, 100, 15, 1, key="idx_retained")

    st.header("EBSD/TKD + mechanical evidence")

    mean_KAM_deg = st.slider("Mean KAM / local misorientation proxy (deg)", 0.0, 10.0, 1.8, 0.1, key="idx_kam")
    high_KAM_fraction_pct = st.slider("High-KAM fraction (%)", 0, 100, 20, 1, key="idx_highkam")
    residual_strain_after_cycles_pct = st.slider("Residual strain after cycles (%)", 0.0, 8.0, 1.0, 0.1, key="idx_residual")
    defect_fraction_pct = st.slider("Pore/crack/defect fraction (%)", 0.0, 8.0, 0.8, 0.1, key="idx_defect")

    st.header("Crystallography inputs")

    lambda2_error = st.slider("|λ₂ − 1| from crystallography page", 0.000, 0.060, 0.012, 0.001, key="idx_l2")
    normalized_volume_change_pct = st.slider("Normalized B2→B19′ volume change (%)", -6.0, 6.0, 0.8, 0.1, key="idx_vol")
    texture_variant_alignment_pct = st.slider("Texture / variant alignment score (%)", 0, 100, 60, 1, key="idx_texture")

    st.header("σMs proxy settings")

    sigma_Ms_at_Ms_MPa = st.slider("σMs at Ms proxy (MPa)", 0, 300, 80, 5, key="idx_sig0")
    clausius_clapeyron_slope_MPa_C = st.slider("dσ/dT proxy (MPa/°C)", 2.0, 12.0, 6.0, 0.5, key="idx_ccslope")


inputs = {
    "power_W": power_W,
    "speed_mm_s": speed_mm_s,
    "hatch_mm": hatch_mm,
    "layer_mm": layer_mm,
    "remelt_passes": remelt_passes,
    "hatch_overlap_pct": hatch_overlap_pct,
    "layer_reheat_severity_pct": layer_reheat_severity_pct,
    "build_plate_C": build_plate_C,
    "effective_Ni_at_pct": effective_Ni_at_pct,
    "oxygen_ppm": oxygen_ppm,
    "powder_reuse_cycles": powder_reuse_cycles,
    "secondary_phase_signal_pct": secondary_phase_signal_pct,
    "service_temperature_C": service_temperature_C,
    "Ms_C": Ms_C,
    "Af_C": Af_C,
    "DSC_peak_width_C": DSC_peak_width_C,
    "hysteresis_width_C": hysteresis_width_C,
    "b2_fraction_pct": b2_fraction_pct,
    "b19_fraction_pct": b19_fraction_pct,
    "retained_b19_after_cycling_pct": retained_b19_after_cycling_pct,
    "mean_KAM_deg": mean_KAM_deg,
    "high_KAM_fraction_pct": high_KAM_fraction_pct,
    "residual_strain_after_cycles_pct": residual_strain_after_cycles_pct,
    "defect_fraction_pct": defect_fraction_pct,
    "lambda2_error": lambda2_error,
    "normalized_volume_change_pct": normalized_volume_change_pct,
    "texture_variant_alignment_pct": texture_variant_alignment_pct,
    "sigma_Ms_at_Ms_MPa": sigma_Ms_at_Ms_MPa,
    "clausius_clapeyron_slope_MPa_C": clausius_clapeyron_slope_MPa_C,
}

outputs = compute_indices(inputs)
factor_table = build_factor_table(inputs, outputs)
action_table = build_action_table(outputs)


# =============================================================================
# Evidence-quality controls
# =============================================================================

with st.expander("Data source / evidence quality settings", expanded=False):
    st.markdown(
        """
Tell the app where the inputs came from. This does not change the physics score;
it changes how strongly the result should be trusted.
"""
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        process_source = st.selectbox(
            "Process parameters",
            ["Measured directly", "Measured but approximate", "Literature/default", "Assumed / unknown"],
            key="q_process",
        )
        composition_source = st.selectbox(
            "Effective Ni / composition",
            ["ICP/EDS/WDS measured", "Estimated from model", "Literature/default", "Assumed / unknown"],
            key="q_composition",
        )

    with q2:
        transformation_source = st.selectbox(
            "Ms/Af and DSC width",
            ["XRD/Rietveld or DSC measured", "Estimated from model", "Literature/default", "Assumed / unknown"],
            key="q_transformation",
        )
        phase_source = st.selectbox(
            "B2/B19′ fraction",
            ["XRD/Rietveld or DSC measured", "Measured but approximate", "Estimated from model", "Assumed / unknown"],
            key="q_phase",
        )

    with q3:
        ebsd_source = st.selectbox(
            "KAM / texture / variant alignment",
            ["EBSD/TKD measured", "Measured but approximate", "Estimated from model", "Assumed / unknown"],
            key="q_ebsd",
        )
        cyclic_source = st.selectbox(
            "Residual strain / retained B19′",
            ["Measured directly", "Measured but approximate", "Estimated from model", "Assumed / unknown"],
            key="q_cyclic",
        )

    with q4:
        oxygen_source = st.selectbox(
            "Oxygen / secondary phases",
            ["ICP/EDS/WDS measured", "XRD/Rietveld or DSC measured", "Estimated from model", "Assumed / unknown"],
            key="q_oxygen",
        )
        crystallography_source = st.selectbox(
            "λ₂ / volume mismatch",
            ["XRD/Rietveld or DSC measured", "Estimated from model", "Literature/default", "Assumed / unknown"],
            key="q_crystal",
        )

quality_values = [
    evidence_weight(process_source),
    evidence_weight(composition_source),
    evidence_weight(transformation_source),
    evidence_weight(phase_source),
    evidence_weight(ebsd_source),
    evidence_weight(cyclic_source),
    evidence_weight(oxygen_source),
    evidence_weight(crystallography_source),
]

evidence_quality_pct = float(np.mean(quality_values))
evidence_label, evidence_text = quality_label(evidence_quality_pct)


# =============================================================================
# Main results
# =============================================================================

score_label, score_text = label_score(outputs["cyclic_function_score"])

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Final cyclic function score", f"{outputs['cyclic_function_score']:.1f}/100")
m2.metric("Verdict", score_label)
m3.metric("Evidence quality", f"{evidence_quality_pct:.0f}/100")
m4.metric("VED", f"{outputs['VED_J_mm3']:.1f} J/mm³")
m5.metric("σMs proxy", f"{outputs['sigma_Ms_proxy_MPa']:.0f} MPa")
m6.metric("Thermal cycle index", f"{outputs['thermal_cycle_index']:.1f}/100")

if outputs["cyclic_function_score"] >= 80:
    st.success(score_text)
elif outputs["cyclic_function_score"] >= 65:
    st.info(score_text)
elif outputs["cyclic_function_score"] >= 45:
    st.warning(score_text)
else:
    st.error(score_text)

if evidence_quality_pct >= 60:
    st.success(f"Evidence quality: {evidence_label}. {evidence_text}")
elif evidence_quality_pct >= 40:
    st.warning(f"Evidence quality: {evidence_label}. {evidence_text}")
else:
    st.error(f"Evidence quality: {evidence_label}. {evidence_text}")


# =============================================================================
# Tabs
# =============================================================================

tab_overview, tab_meanings, tab_plots, tab_crystal, tab_actions, tab_report = st.tabs(
    [
        "1. What this index does",
        "2. Factor meanings and outcomes",
        "3. Risk maps and plots",
        "4. Crystallography / correspondence link",
        "5. Actions and measurements",
        "6. Copy-ready report",
    ]
)


# =============================================================================
# Tab 1
# =============================================================================

with tab_overview:
    st.subheader("The index logic")

    st.markdown(
        """
The page calculates a **cyclic functional stability index** for NiTi-AM.

It does not ask only:  
`Is the density high?`

It asks:

1. Did the LPBF process create excessive or insufficient thermal exposure?
2. Did remelting/reheating create residual strain or retained B19′?
3. Is the alloy B2/austenite at service temperature?
4. Is stress-induced B19′ possible without plastic deformation?
5. Is transformation sharp or broad?
6. Is λ₂ / lattice compatibility acceptable?
7. Are texture, KAM and variant alignment favorable?
8. Is the sample likely to keep superelasticity after repeated cycling?
"""
    )

    logic = pd.DataFrame(
        [
            {
                "stage": "LPBF thermal history",
                "inputs": "P, v, hatch, layer thickness, remelts, overlap, reheating",
                "calculated factor": "Thermal Cycle Severity Index",
                "meaning": "How much repeated heat exposure the material experiences.",
                "main risk": "Residual strain, precipitates/oxides, retained martensite, transformation broadening.",
            },
            {
                "stage": "Composition and oxygen",
                "inputs": "effective Ni at.%, oxygen ppm, powder reuse, secondary phases",
                "calculated factor": "Oxy-precipitate risk + Ni deviation risk",
                "meaning": "Whether chemistry and oxygen disturb B2/B19′ transformation.",
                "main risk": "Ti₄Ni₂Oₓ-like particles, oxide signals, shifted Ms/Af, pinned transformation.",
            },
            {
                "stage": "Phase state",
                "inputs": "B2 fraction, B19′ fraction, service temperature, Ms, Af",
                "calculated factor": "Phase suitability score",
                "meaning": "Whether the alloy is likely B2/austenite and transformable at service.",
                "main risk": "Martensite already present at service or unstable mixed phase.",
            },
            {
                "stage": "Cyclic damage",
                "inputs": "retained B19′, residual strain, KAM, DSC width, hysteresis",
                "calculated factor": "B19′ stabilization + cyclic degradation risk",
                "meaning": "Whether cycling will stabilize martensite and reduce recovery.",
                "main risk": "Dislocation pile-ups, plasticity, lamellar martensite stabilization.",
            },
            {
                "stage": "Crystallography",
                "inputs": "λ₂ error, normalized volume change, texture alignment, KAM",
                "calculated factor": "Crystallography score",
                "meaning": "Whether B2→B19′ transformation is geometrically compatible.",
                "main risk": "High mismatch, poor variant accommodation, higher hysteresis.",
            },
            {
                "stage": "Final function",
                "inputs": "all above",
                "calculated factor": "Cyclic Function Score",
                "meaning": "Overall screening estimate for stable superelastic/shape-memory function.",
                "main risk": "Looks good initially but degrades after cycling.",
            },
        ]
    )

    st.dataframe(logic, use_container_width=True, hide_index=True)

    st.subheader("Good vs bad NiTi-AM behavior")

    routes = pd.DataFrame(
        [
            {
                "route": "Good route",
                "condition": "B2 at service → stress-induced B19′ during loading → reverse B19′→B2 on unloading",
                "expected result": "High recoverable strain, low residual strain, stable superelastic loops.",
            },
            {
                "route": "Mixed/unstable route",
                "condition": "B2 + residual B19′ near service temperature",
                "expected result": "Transformation is possible but unstable; hysteresis and residual strain may grow.",
            },
            {
                "route": "Bad route",
                "condition": "Retained/stabilized B19′ + dislocation pile-ups + broad transformation",
                "expected result": "Superelastic degradation, residual martensite, lower recoverable strain.",
            },
            {
                "route": "Defect-dominated route",
                "condition": "Cracks, pores, lack of fusion, keyhole pores or oxide particles dominate",
                "expected result": "Mechanical failure or poor cyclic durability even if phase state looks acceptable.",
            },
        ]
    )

    st.dataframe(routes, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 2
# =============================================================================

with tab_meanings:
    st.subheader("Factor meanings, how to find inputs, and possible outcomes")

    st.dataframe(factor_table, use_container_width=True, hide_index=True)

    st.subheader("Input guide")

    input_guide = pd.DataFrame(
        [
            {
                "input": "Power, speed, hatch, layer thickness",
                "best source": "LPBF machine parameter file",
                "backup": "manual process log",
                "why it matters": "Controls VED, linear energy density and thermal exposure.",
            },
            {
                "input": "Remelt/rescan passes",
                "best source": "scan strategy file",
                "backup": "process description",
                "why it matters": "Repeated thermal exposure can alter residual strain, B19′ stability and precipitate state.",
            },
            {
                "input": "Hatch overlap",
                "best source": "track spacing and melt-pool width",
                "backup": "estimated from hatch spacing",
                "why it matters": "High overlap can cause reheating/remelting; low overlap can cause lack of fusion.",
            },
            {
                "input": "Effective Ni at.%",
                "best source": "ICP-OES, WDS/EDS or calibrated vaporization model",
                "backup": "powder nominal Ni",
                "why it matters": "Small Ni changes strongly shift transformation temperatures.",
            },
            {
                "input": "Ms, Af, DSC peak width, hysteresis",
                "best source": "DSC",
                "backup": "literature or composition model",
                "why it matters": "Defines whether service temperature is B2, B19′ or mixed, and whether transformation is sharp.",
            },
            {
                "input": "B2/B19′ fraction",
                "best source": "XRD/Rietveld, EBSD/TKD, temperature-resolved XRD",
                "backup": "XRD peak screening",
                "why it matters": "Direct phase evidence for superelastic readiness.",
            },
            {
                "input": "Retained B19′ after cycling",
                "best source": "XRD/EBSD before and after cyclic test",
                "backup": "mechanical residual strain proxy",
                "why it matters": "Retained martensite is a key sign of cyclic superelastic degradation.",
            },
            {
                "input": "KAM / local misorientation",
                "best source": "EBSD/TKD",
                "backup": "none",
                "why it matters": "High KAM suggests residual strain, dislocation density or local distortion.",
            },
            {
                "input": "λ₂ and volume mismatch",
                "best source": "XRD-refined B2/B19′ lattice parameters",
                "backup": "literature lattice constants",
                "why it matters": "Crystallographic compatibility controls transformation mismatch and hysteresis tendency.",
            },
            {
                "input": "Texture / variant alignment",
                "best source": "EBSD/TKD parent/daughter orientation analysis",
                "backup": "build-direction texture proxy",
                "why it matters": "Variant selection depends on grain orientation and loading/build direction.",
            },
        ]
    )

    st.dataframe(input_guide, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 3
# =============================================================================

with tab_plots:
    st.subheader("Risk and score plots")

    risk_rows = [
        {"name": "Thermal cycle severity", "value": outputs["thermal_cycle_index"], "type": "risk"},
        {"name": "Lack-of-fusion", "value": outputs["lack_of_fusion_risk"], "type": "risk"},
        {"name": "Oxy-precipitate", "value": outputs["oxide_precipitate_risk"], "type": "risk"},
        {"name": "B19′ stabilization", "value": outputs["b19_stabilization_risk"], "type": "risk"},
        {"name": "Residual distortion", "value": outputs["residual_distortion_risk"], "type": "risk"},
        {"name": "Transformation broadening", "value": outputs["transformation_broadening_risk"], "type": "risk"},
        {"name": "Plasticity during loading", "value": outputs["plasticity_during_loading_risk"], "type": "risk"},
        {"name": "Cyclic degradation", "value": outputs["cyclic_degradation_risk"], "type": "risk"},
    ]

    score_rows = [
        {"name": "Phase suitability", "value": outputs["phase_suitability_score"], "type": "score"},
        {"name": "Stress window", "value": outputs["stress_window_score"], "type": "score"},
        {"name": "Crystallography", "value": outputs["crystallography_score"], "type": "score"},
        {"name": "Final cyclic function", "value": outputs["cyclic_function_score"], "type": "score"},
    ]

    plot_df = pd.DataFrame(risk_rows + score_rows)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            pd.DataFrame(risk_rows).sort_values("value", ascending=True),
            x="value",
            y="name",
            orientation="h",
            title="Risk factors: higher is worse",
            text="value",
        )
        fig.update_traces(texttemplate="%{text:.1f}")
        fig.update_layout(xaxis_range=[0, 100], height=520)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            pd.DataFrame(score_rows).sort_values("value", ascending=True),
            x="value",
            y="name",
            orientation="h",
            title="Function scores: higher is better",
            text="value",
        )
        fig.update_traces(texttemplate="%{text:.1f}")
        fig.update_layout(xaxis_range=[0, 100], height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Radar view")

    radar_names = [
        "Phase",
        "Stress window",
        "Crystallography",
        "Low B19′ stabilization",
        "Low residual distortion",
        "Low broadening",
        "Low oxide risk",
        "Low cyclic degradation",
    ]

    radar_values = [
        outputs["phase_suitability_score"],
        outputs["stress_window_score"],
        outputs["crystallography_score"],
        inverse_score_from_risk(outputs["b19_stabilization_risk"]),
        inverse_score_from_risk(outputs["residual_distortion_risk"]),
        inverse_score_from_risk(outputs["transformation_broadening_risk"]),
        inverse_score_from_risk(outputs["oxide_precipitate_risk"]),
        inverse_score_from_risk(outputs["cyclic_degradation_risk"]),
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_names + [radar_names[0]],
            fill="toself",
            name="current condition",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=580,
        title="Functional balance map",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("σMs temperature proxy")

    temp_grid = np.linspace(Ms_C - 30, Ms_C + 220, 150)
    sigma_grid = sigma_Ms_at_Ms_MPa + clausius_clapeyron_slope_MPa_C * np.maximum(0, temp_grid - Ms_C)

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=temp_grid, y=sigma_grid, mode="lines", name="σMs proxy"))
    fig_sig.add_vline(x=Ms_C, line_dash="dot", annotation_text="Ms")
    fig_sig.add_vline(x=Af_C, line_dash="dash", annotation_text="Af")
    fig_sig.add_vline(x=service_temperature_C, line_dash="solid", annotation_text="service T")
    fig_sig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="σMs proxy (MPa)",
        title="Critical stress trend from service temperature relative to Ms",
        height=500,
    )
    st.plotly_chart(fig_sig, use_container_width=True)


# =============================================================================
# Tab 4
# =============================================================================

with tab_crystal:
    st.subheader("Crystallography and correspondence-theory connection")

    st.markdown(
        """
This index uses crystallography in a practical way:

- **λ₂ close to 1** means the middle transformation stretch is compatible with a low-mismatch interface.
- **Small normalized volume change** means the B2→B19′ transformation has less volumetric mismatch.
- **Texture/variant alignment** estimates whether the build/loading direction supports favorable transformation strain.
- **KAM/local misorientation** warns that residual strain may block clean reversible transformation.

For a Cayron-style correspondence-theory analysis, variant pairs should not be treated as random.
B2→B19′ NiTi is normally discussed using parent symmetry, daughter symmetry, correspondence variants,
orientation variants, distortion variants and operators between variants. The important practical message is:
**exact variant claims require EBSD/TKD parent reconstruction and measured martensite orientations.**
"""
    )

    crystal_table = pd.DataFrame(
        [
            {
                "crystallographic item": "λ₂",
                "your input": f"{1 + lambda2_error:.5f} or |λ₂−1|={lambda2_error:.5f}",
                "meaning": "Middle principal stretch of the transformation. λ₂≈1 is better for compatibility.",
                "good outcome": "|λ₂−1| below about 0.005–0.015.",
                "bad outcome": "Large mismatch can increase hysteresis and residual strain.",
            },
            {
                "crystallographic item": "Normalized volume change",
                "your input": f"{normalized_volume_change_pct:+.2f}%",
                "meaning": "Volume change per comparable B2/B19′ formula-unit basis.",
                "good outcome": "Small absolute volume change.",
                "bad outcome": "Large mismatch can create internal stress and unstable transformation.",
            },
            {
                "crystallographic item": "Texture / variant alignment",
                "your input": f"{texture_variant_alignment_pct:.0f}%",
                "meaning": "Whether build/loading direction aligns with favorable transformation strain families.",
                "good outcome": "Favorable grains/variants are aligned with the loading direction.",
                "bad outcome": "Poor alignment makes transformation anisotropic or hard to activate.",
            },
            {
                "crystallographic item": "KAM / local misorientation",
                "your input": f"{mean_KAM_deg:.1f}°",
                "meaning": "Proxy for local residual strain, dislocation density or poor orientation quality.",
                "good outcome": "Low KAM supports cleaner reversible transformation.",
                "bad outcome": "High KAM supports martensite stabilization and loop degradation.",
            },
            {
                "crystallographic item": "Retained B19′ after cycling",
                "your input": f"{retained_b19_after_cycling_pct:.0f}%",
                "meaning": "Direct sign that stress-induced martensite did not fully reverse.",
                "good outcome": "Near zero retained martensite after unloading.",
                "bad outcome": "High retained B19′ means superelasticity is degrading.",
            },
        ]
    )

    st.dataframe(crystal_table, use_container_width=True, hide_index=True)

    st.subheader("Cayron-style correspondence logic — how to use it safely")

    ct_table = pd.DataFrame(
        [
            {
                "level": "Level 1: lattice compatibility",
                "what this page can do": "Use λ₂, volume change and strain proxies.",
                "what it tells you": "Whether B2 and B19′ metrics are geometrically favorable.",
                "what it cannot prove": "Exact habit plane or exact variant pairs.",
            },
            {
                "level": "Level 2: variant-family screening",
                "what this page can do": "Use texture/variant alignment and KAM as proxies.",
                "what it tells you": "Whether orientation evidence is favorable or risky.",
                "what it cannot prove": "True V1–V12 variant identity.",
            },
            {
                "level": "Level 3: correspondence operators",
                "what this page can do": "Explain that B2→B19′ variant pairs should follow operator families.",
                "what it tells you": "Which pairs are crystallographically meaningful in principle.",
                "what it cannot prove": "Operator assignment without EBSD/TKD reconstruction.",
            },
            {
                "level": "Level 4: real Cayron/CT analysis",
                "what this page can do": "Prepare the inputs and evidence checklist.",
                "what it tells you": "What data are missing.",
                "what it cannot prove": "Full GenOVa/ARPGE-style parent reconstruction or weak-plane/twin solution.",
            },
        ]
    )

    st.dataframe(ct_table, use_container_width=True, hide_index=True)

    st.info(
        "Use this page to decide whether crystallography supports or weakens your LPBF NiTi function claim. "
        "For a real variant claim, export EBSD/TKD orientations and perform parent B2 reconstruction plus variant indexing."
    )


# =============================================================================
# Tab 5
# =============================================================================

with tab_actions:
    st.subheader("Recommended actions and measurements")

    st.dataframe(action_table, use_container_width=True, hide_index=True)

    st.subheader("Minimum evidence checklist")

    checklist = pd.DataFrame(
        [
            {
                "evidence": "DSC Ms/Mf/As/Af and peak width",
                "needed because": "Defines service phase state and transformation broadening.",
                "minimum acceptable": "At least one DSC heating/cooling cycle.",
                "best case": "DSC before and after cyclic mechanical loading.",
            },
            {
                "evidence": "XRD phase state",
                "needed because": "Checks B2, B19′, retained martensite and secondary phases.",
                "minimum acceptable": "Peak-screening comparison.",
                "best case": "Rietveld/Pawley refinement with lattice parameters.",
            },
            {
                "evidence": "Effective Ni/Ti composition",
                "needed because": "Ni loss shifts transformation temperatures.",
                "minimum acceptable": "EDS semi-quantitative check.",
                "best case": "ICP-OES/WDS calibrated composition.",
            },
            {
                "evidence": "EBSD/TKD KAM and texture",
                "needed because": "Residual strain and texture control variant accommodation.",
                "minimum acceptable": "Phase/orientation maps.",
                "best case": "Parent reconstruction and B19′ variant indexing.",
            },
            {
                "evidence": "Cyclic superelastic loops",
                "needed because": "Final function must be mechanically validated.",
                "minimum acceptable": "Loading/unloading loop with residual strain.",
                "best case": "Multi-cycle recoverable strain, residual strain, hysteresis and retained B19′ check.",
            },
            {
                "evidence": "SEM/optical defect analysis",
                "needed because": "Cracks/pores can dominate failure regardless of crystallography.",
                "minimum acceptable": "Cross-section image and pore/crack area fraction.",
                "best case": "3D porosity or multiple section statistics.",
            },
        ]
    )

    st.dataframe(checklist, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 6
# =============================================================================

with tab_report:
    st.subheader("Copy-ready report paragraph")

    risk_summary = pd.DataFrame(
        [
            {"item": "final cyclic function score", "value": f"{outputs['cyclic_function_score']:.1f}/100", "interpretation": score_label},
            {"item": "evidence quality", "value": f"{evidence_quality_pct:.0f}/100", "interpretation": evidence_label},
            {"item": "thermal cycle index", "value": f"{outputs['thermal_cycle_index']:.1f}/100", "interpretation": label_risk(outputs["thermal_cycle_index"])[0]},
            {"item": "B19′ stabilization risk", "value": f"{outputs['b19_stabilization_risk']:.1f}/100", "interpretation": label_risk(outputs["b19_stabilization_risk"])[0]},
            {"item": "residual distortion risk", "value": f"{outputs['residual_distortion_risk']:.1f}/100", "interpretation": label_risk(outputs["residual_distortion_risk"])[0]},
            {"item": "transformation broadening risk", "value": f"{outputs['transformation_broadening_risk']:.1f}/100", "interpretation": label_risk(outputs["transformation_broadening_risk"])[0]},
            {"item": "crystallography score", "value": f"{outputs['crystallography_score']:.1f}/100", "interpretation": label_score(outputs["crystallography_score"])[0]},
            {"item": "cyclic degradation risk", "value": f"{outputs['cyclic_degradation_risk']:.1f}/100", "interpretation": label_risk(outputs["cyclic_degradation_risk"])[0]},
        ]
    )

    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

    report_text = f"""
The thermal-history–crystallography–cyclic-function index evaluates the selected LPBF NiTi condition by combining process thermal exposure, oxygen/secondary-phase risk, B2/B19′ phase suitability, residual strain indicators, transformation broadening, σMs tendency, crystallographic compatibility and cyclic degradation risk. For the current condition, VED is {outputs['VED_J_mm3']:.1f} J/mm³ and the Thermal Cycle Severity Index is {outputs['thermal_cycle_index']:.1f}/100. The predicted B19′ stabilization risk is {outputs['b19_stabilization_risk']:.1f}/100, residual distortion risk is {outputs['residual_distortion_risk']:.1f}/100, and transformation broadening risk is {outputs['transformation_broadening_risk']:.1f}/100. The crystallography score, based on λ₂ mismatch, normalized volume change, texture/variant alignment and KAM, is {outputs['crystallography_score']:.1f}/100. The final cyclic function score is {outputs['cyclic_function_score']:.1f}/100, giving the interpretation: {score_label}. The evidence-quality score is {evidence_quality_pct:.0f}/100, so the conclusion should be treated as {evidence_label} confidence. This result should be validated using DSC, XRD/Rietveld, composition measurement, EBSD/TKD and cyclic superelastic testing.
""".strip()

    st.code(report_text, language="text")

    st.subheader("Export")

    export = pd.concat(
        [
            pd.DataFrame([inputs]),
            pd.DataFrame([outputs]),
            pd.DataFrame(
                [
                    {
                        "evidence_quality_pct": evidence_quality_pct,
                        "evidence_label": evidence_label,
                        "final_verdict": score_label,
                    }
                ]
            ),
        ],
        axis=1,
    )

    st.download_button(
        "Download index result as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="niti_thermal_history_crystallography_cyclic_index.csv",
        mime="text/csv",
    )
