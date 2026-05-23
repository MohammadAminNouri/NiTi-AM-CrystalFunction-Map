"""
Physics-informed vaporization / composition-shift model for LPBF NiTi.

This is a screening surrogate, not a full CFD/CALPHAD model.

Purpose:
LPBF parameters
-> hot-surface temperature proxy
-> Ni/Ti vaporization tendency
-> final effective Ni at.%
-> transformation-temperature shift
-> functional risk map

Important correction:
The evaporation temperature must represent the molten-pool free surface / hot
spot, not the average melt-pool temperature. If the estimated surface
temperature is too low, vapor pressure becomes almost zero and the dashboard
shows no composition consequence, which is not useful for LPBF NiTi screening.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

R_GAS = 8.314462618
ATM_PA = 101325.0

MOLAR_MASS_KG_MOL = {
    "Ni": 58.6934e-3,
    "Ti": 47.867e-3,
}

MOLAR_MASS_G_MOL = {
    "Ni": 58.6934,
    "Ti": 47.867,
}

# Approximate elemental boiling points
BOILING_K = {
    "Ni": 3186.0,
    "Ti": 3560.0,
}

# Approximate vaporization enthalpies
DELTA_H_VAP_J_MOL = {
    "Ni": 377_000.0,
    "Ti": 425_000.0,
}

NITI_DENSITY_KG_M3 = 6450.0


@dataclass
class VaporizationInput:
    laser_power_W: float
    scan_speed_mm_s: float
    hatch_spacing_mm: float
    layer_thickness_mm: float
    powder_Ni_at_pct: float = 51.30

    beam_diameter_um: float = 80.0
    absorptivity: float = 0.38
    build_plate_C: float = 80.0

    remelt_passes: int = 0
    oxygen_ppm: float = 70.0

    # Keep this small. This is not the same as saying the physical
    # accommodation is exactly this value; it is a lumped screening factor.
    accommodation_coeff: float = 0.08

    # Use this only for calibration against measured EDS/ICP.
    calibration_scale: float = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def volumetric_energy_density(
    laser_power_W: float,
    scan_speed_mm_s: float,
    hatch_spacing_mm: float,
    layer_thickness_mm: float,
) -> float:
    if scan_speed_mm_s <= 0 or hatch_spacing_mm <= 0 or layer_thickness_mm <= 0:
        return float("nan")
    return laser_power_W / (scan_speed_mm_s * hatch_spacing_mm * layer_thickness_mm)


def linear_energy_density(laser_power_W: float, scan_speed_mm_s: float) -> float:
    if scan_speed_mm_s <= 0:
        return float("nan")
    return laser_power_W / scan_speed_mm_s


def at_percent_to_wt_fractions(ni_at_pct: float) -> Dict[str, float]:
    ni_at_pct = _clamp(ni_at_pct, 0.0, 100.0)
    ti_at_pct = 100.0 - ni_at_pct

    ni_mass = ni_at_pct * MOLAR_MASS_G_MOL["Ni"]
    ti_mass = ti_at_pct * MOLAR_MASS_G_MOL["Ti"]
    total = ni_mass + ti_mass

    if total <= 0:
        return {"Ni": 0.0, "Ti": 0.0}

    return {"Ni": ni_mass / total, "Ti": ti_mass / total}


def wt_fractions_to_ni_at_percent(ni_wt: float, ti_wt: float) -> float:
    ni_wt = max(0.0, ni_wt)
    ti_wt = max(0.0, ti_wt)

    ni_moles = ni_wt / MOLAR_MASS_G_MOL["Ni"]
    ti_moles = ti_wt / MOLAR_MASS_G_MOL["Ti"]
    total_moles = ni_moles + ti_moles

    if total_moles <= 0:
        return float("nan")

    return 100.0 * ni_moles / total_moles


def estimate_hot_surface_temperature_K(p: VaporizationInput) -> float:
    """
    Hot free-surface temperature proxy.

    This is the important corrected part.

    For evaporation, we need the laser-side molten-pool surface/hot spot,
    not the average melt-pool temperature. Evaporation becomes significant
    only when the local surface temperature approaches the boiling range.

    The function is intentionally monotonic:
    - higher power increases T
    - slower scan increases T
    - higher absorptivity increases T
    - thinner layers increase T
    - smaller beam diameter increases intensity
    - remelting adds repeated thermal exposure
    """
    ved = volumetric_energy_density(
        p.laser_power_W,
        p.scan_speed_mm_s,
        p.hatch_spacing_mm,
        p.layer_thickness_mm,
    )
    led = linear_energy_density(p.laser_power_W, p.scan_speed_mm_s)

    if math.isnan(ved) or math.isnan(led):
        return float("nan")

    beam_factor = (80.0 / max(p.beam_diameter_um, 20.0)) ** 0.35
    layer_factor = (0.030 / max(p.layer_thickness_mm, 0.005)) ** 0.25
    speed_factor = (800.0 / max(p.scan_speed_mm_s, 50.0)) ** 0.30
    power_factor = (max(p.laser_power_W, 1.0) / 100.0) ** 0.55
    absorptivity_factor = p.absorptivity / 0.38

    # Baseline near melting plus laser hot-spot superheat.
    # The nonlinear term makes high-power/slow-speed cases move toward
    # vaporization temperature, while moderate superelastic windows stay lower.
    hot_T = (
        1850.0
        + 600.0 * power_factor * speed_factor * beam_factor * layer_factor * absorptivity_factor
        + 4.0 * ved
        + 900.0 * max(0.0, led - 0.12)
        + 0.10 * (p.build_plate_C - 80.0)
    )

    # Remelting means repeated exposure. It should not simply explode T,
    # but it should increase effective evaporation severity.
    hot_T += 60.0 * max(0, p.remelt_passes)

    # Keep within physically reasonable screening bounds.
    return _clamp(hot_T, 1750.0, 3850.0)


def estimate_melt_pool_geometry_um(p: VaporizationInput) -> Dict[str, float]:
    ved = volumetric_energy_density(
        p.laser_power_W,
        p.scan_speed_mm_s,
        p.hatch_spacing_mm,
        p.layer_thickness_mm,
    )

    if math.isnan(ved):
        return {
            "width_um": float("nan"),
            "depth_um": float("nan"),
            "length_um": float("nan"),
            "top_area_m2": float("nan"),
            "cross_section_area_m2": float("nan"),
        }

    layer_um = p.layer_thickness_mm * 1000.0
    beam = p.beam_diameter_um

    width_um = beam * (0.90 + 0.0045 * ved + 0.0010 * (p.laser_power_W - 100.0))
    depth_um = layer_um * (1.15 + 0.0100 * ved + 0.0008 * (p.laser_power_W - 100.0))
    length_um = width_um * (1.60 + 0.0025 * ved)

    width_um = _clamp(width_um, 45.0, 500.0)
    depth_um = _clamp(depth_um, max(12.0, 0.7 * layer_um), 400.0)
    length_um = _clamp(length_um, 70.0, 1300.0)

    width_m = width_um * 1e-6
    depth_m = depth_um * 1e-6
    length_m = length_um * 1e-6

    top_area_m2 = math.pi * (length_m / 2.0) * (width_m / 2.0)
    cross_section_area_m2 = 0.5 * math.pi * (width_m / 2.0) * depth_m

    return {
        "width_um": width_um,
        "depth_um": depth_um,
        "length_um": length_um,
        "top_area_m2": top_area_m2,
        "cross_section_area_m2": cross_section_area_m2,
    }


def pure_vapor_pressure_pa(element: str, temperature_K: float) -> float:
    """
    Clausius-Clapeyron vapor-pressure proxy.

    Anchored by P(Tb) = 1 atm.
    """
    tb = BOILING_K[element]
    dh = DELTA_H_VAP_J_MOL[element]

    exponent = (dh / R_GAS) * ((1.0 / tb) - (1.0 / temperature_K))
    exponent = _clamp(exponent, -80.0, 25.0)

    return ATM_PA * math.exp(exponent)


def langmuir_flux_mol_m2_s(
    partial_pressure_pa: float,
    molar_mass_kg_mol: float,
    temperature_K: float,
    accommodation_coeff: float,
) -> float:
    if partial_pressure_pa <= 0 or molar_mass_kg_mol <= 0 or temperature_K <= 0:
        return 0.0

    lam = _clamp(accommodation_coeff, 0.0, 1.0)
    denominator = math.sqrt(2.0 * math.pi * molar_mass_kg_mol * R_GAS * temperature_K)

    return lam * partial_pressure_pa / denominator


def remelting_multiplier(layer_thickness_mm: float, remelt_passes: int) -> float:
    """
    Thinner layers and explicit remelting increase repeated thermal exposure.
    """
    layer_thickness_mm = max(layer_thickness_mm, 0.005)
    layer_factor = (0.030 / layer_thickness_mm) ** 0.45
    pass_factor = 1.0 + 0.85 * max(0, remelt_passes)

    return _clamp(layer_factor * pass_factor, 0.40, 7.00)


def compute_vaporization_composition(p: VaporizationInput) -> Dict[str, float]:
    ved = volumetric_energy_density(
        p.laser_power_W,
        p.scan_speed_mm_s,
        p.hatch_spacing_mm,
        p.layer_thickness_mm,
    )
    led = linear_energy_density(p.laser_power_W, p.scan_speed_mm_s)
    temp_K = estimate_hot_surface_temperature_K(p)
    geom = estimate_melt_pool_geometry_um(p)

    if math.isnan(ved) or math.isnan(temp_K):
        raise ValueError("Invalid LPBF input parameters.")

    initial_wt = at_percent_to_wt_fractions(p.powder_Ni_at_pct)

    x_ni = _clamp(p.powder_Ni_at_pct / 100.0, 0.0, 1.0)
    x_ti = 1.0 - x_ni

    # First-pass activity approximation.
    activity_ni = x_ni
    activity_ti = x_ti

    pure_p_ni = pure_vapor_pressure_pa("Ni", temp_K)
    pure_p_ti = pure_vapor_pressure_pa("Ti", temp_K)

    partial_p_ni = pure_p_ni * activity_ni
    partial_p_ti = pure_p_ti * activity_ti

    flux_ni = langmuir_flux_mol_m2_s(
        partial_p_ni,
        MOLAR_MASS_KG_MOL["Ni"],
        temp_K,
        p.accommodation_coeff,
    )
    flux_ti = langmuir_flux_mol_m2_s(
        partial_p_ti,
        MOLAR_MASS_KG_MOL["Ti"],
        temp_K,
        p.accommodation_coeff,
    )

    top_area_m2 = geom["top_area_m2"]
    cross_section_area_m2 = geom["cross_section_area_m2"]
    scan_speed_m_s = p.scan_speed_mm_s / 1000.0

    deposit_volume_rate_m3_s = cross_section_area_m2 * scan_speed_m_s
    deposit_mass_rate_kg_s = deposit_volume_rate_m3_s * NITI_DENSITY_KG_M3

    initial_ni_mass_rate = deposit_mass_rate_kg_s * initial_wt["Ni"]
    initial_ti_mass_rate = deposit_mass_rate_kg_s * initial_wt["Ti"]

    cycle_factor = remelting_multiplier(p.layer_thickness_mm, p.remelt_passes)
    scale = max(0.0, p.calibration_scale)

    ni_loss_rate = (
        flux_ni
        * MOLAR_MASS_KG_MOL["Ni"]
        * top_area_m2
        * cycle_factor
        * scale
    )

    ti_loss_rate = (
        flux_ti
        * MOLAR_MASS_KG_MOL["Ti"]
        * top_area_m2
        * cycle_factor
        * scale
    )

    # Keep extreme slider combinations numerically safe.
    ni_loss_rate = min(ni_loss_rate, 0.65 * initial_ni_mass_rate)
    ti_loss_rate = min(ti_loss_rate, 0.65 * initial_ti_mass_rate)

    final_ni_mass_rate = max(0.0, initial_ni_mass_rate - ni_loss_rate)
    final_ti_mass_rate = max(0.0, initial_ti_mass_rate - ti_loss_rate)
    final_total = final_ni_mass_rate + final_ti_mass_rate

    final_ni_wt = final_ni_mass_rate / final_total
    final_ti_wt = final_ti_mass_rate / final_total
    final_ni_at = wt_fractions_to_ni_at_percent(final_ni_wt, final_ti_wt)

    delta_ni_at = final_ni_at - p.powder_Ni_at_pct

    # Ni loss increases transformation temperatures.
    # Rule of thumb: 0.1 at.% Ni loss can shift transformation temperatures upward strongly.
    predicted_transform_shift_C = -100.0 * delta_ni_at

    cvi = abs(predicted_transform_shift_C)

    if cvi < 3.0:
        risk_label = "low composition-shift risk"
    elif cvi < 15.0:
        risk_label = "moderate composition-shift risk"
    else:
        risk_label = "high Ni-loss / transformation-shift risk"

    if delta_ni_at < -0.15:
        action = (
            "Ni loss is meaningful. Reduce thermal exposure: lower power, increase scan speed, "
            "avoid unnecessary remelting/rescanning, increase hatch efficiency, or recalibrate "
            "with measured EDS/ICP and DSC."
        )
    elif delta_ni_at < -0.03:
        action = (
            "Small but visible Ni loss is predicted. This may still shift Af/Ms. "
            "Check with DSC and composition measurement."
        )
    else:
        action = (
            "Predicted Ni loss is small for this condition. This is acceptable for a moderate "
            "superelastic process window, but still verify with EDS/ICP and DSC."
        )

    return {
        "VED_J_mm3": ved,
        "linear_energy_density_J_mm": led,
        "peak_temperature_K": temp_K,
        "peak_temperature_C": temp_K - 273.15,
        "melt_pool_width_um": geom["width_um"],
        "melt_pool_depth_um": geom["depth_um"],
        "melt_pool_length_um": geom["length_um"],
        "top_surface_area_m2": top_area_m2,
        "cross_section_area_m2": cross_section_area_m2,
        "remelting_multiplier": cycle_factor,
        "initial_Ni_at_pct": p.powder_Ni_at_pct,
        "initial_Ni_wt_pct": 100.0 * initial_wt["Ni"],
        "initial_Ti_wt_pct": 100.0 * initial_wt["Ti"],
        "pure_vapor_pressure_Ni_Pa": pure_p_ni,
        "pure_vapor_pressure_Ti_Pa": pure_p_ti,
        "partial_pressure_Ni_Pa": partial_p_ni,
        "partial_pressure_Ti_Pa": partial_p_ti,
        "flux_Ni_mol_m2_s": flux_ni,
        "flux_Ti_mol_m2_s": flux_ti,
        "deposit_mass_rate_kg_s": deposit_mass_rate_kg_s,
        "Ni_loss_rate_kg_s": ni_loss_rate,
        "Ti_loss_rate_kg_s": ti_loss_rate,
        "Ni_loss_percent_of_initial_Ni_mass": 100.0 * ni_loss_rate / max(initial_ni_mass_rate, 1e-30),
        "Ti_loss_percent_of_initial_Ti_mass": 100.0 * ti_loss_rate / max(initial_ti_mass_rate, 1e-30),
        "final_Ni_wt_pct": 100.0 * final_ni_wt,
        "final_Ti_wt_pct": 100.0 * final_ti_wt,
        "final_Ni_at_pct": final_ni_at,
        "delta_Ni_at_pct": delta_ni_at,
        "predicted_transformation_shift_C": predicted_transform_shift_C,
        "composition_vulnerability_index_C": cvi,
        "risk_label": risk_label,
        "recommended_action": action,
    }


def sweep_power_speed(
    base: VaporizationInput,
    powers_W: List[float],
    speeds_mm_s: List[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for power in powers_W:
        for speed in speeds_mm_s:
            p2 = VaporizationInput(
                laser_power_W=float(power),
                scan_speed_mm_s=float(speed),
                hatch_spacing_mm=base.hatch_spacing_mm,
                layer_thickness_mm=base.layer_thickness_mm,
                powder_Ni_at_pct=base.powder_Ni_at_pct,
                beam_diameter_um=base.beam_diameter_um,
                absorptivity=base.absorptivity,
                build_plate_C=base.build_plate_C,
                remelt_passes=base.remelt_passes,
                oxygen_ppm=base.oxygen_ppm,
                accommodation_coeff=base.accommodation_coeff,
                calibration_scale=base.calibration_scale,
            )
            out = compute_vaporization_composition(p2)
            out["laser_power_W"] = power
            out["scan_speed_mm_s"] = speed
            rows.append(out)

    return rows
