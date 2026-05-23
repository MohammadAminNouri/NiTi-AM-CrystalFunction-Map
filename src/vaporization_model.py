"""
Physics-informed vaporization / composition-shift model for LPBF NiTi.

This module is a lightweight screening surrogate inspired by integrated
LPBF vaporization models. It is NOT a full heat-fluid-flow solver and it
does NOT replace DSC, ICP/EDS, XRD, SEM, EBSD/TKD, or mechanical testing.

Purpose in this repository:
    LPBF parameters
    -> melt-pool thermal proxy
    -> Ni/Ti vaporization tendency
    -> final effective Ni at.%
    -> transformation-temperature shift
    -> functional risk map

Main assumptions:
    1. Binary Ni-Ti alloy.
    2. Ideal first-pass activity model: partial pressure = x_i * P_i^pure.
    3. Peak temperature and pool dimensions are empirical/screening proxies.
    4. Remelting is represented by a transparent multiplier.
    5. Outputs are meant for process-window comparison, not certification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

R_GAS = 8.314462618  # J/(mol K)
ATM_PA = 101325.0

# Atomic / molar data
MOLAR_MASS_KG_MOL = {
    "Ni": 58.6934e-3,
    "Ti": 47.867e-3,
}

MOLAR_MASS_G_MOL = {
    "Ni": 58.6934,
    "Ti": 47.867,
}

# Approximate boiling points and vaporization enthalpies.
# These are used only for a transparent Clausius-Clapeyron vapor-pressure proxy.
BOILING_K = {
    "Ni": 3186.0,
    "Ti": 3560.0,
}

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
    absorptivity: float = 0.35
    build_plate_C: float = 80.0

    remelt_passes: int = 0
    oxygen_ppm: float = 70.0

    # Langmuir accommodation/calibration parameter.
    # Higher value = stronger evaporation.
    # Keep this adjustable because real values depend on gas flow, pressure,
    # plume, surface condition, and calibration against measured composition.
    accommodation_coeff: float = 0.25

    # Extra global calibration factor. Keep at 1.0 unless you have measured
    # ICP/EDS data and want to calibrate the surrogate.
    calibration_scale: float = 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def volumetric_energy_density(
    laser_power_W: float,
    scan_speed_mm_s: float,
    hatch_spacing_mm: float,
    layer_thickness_mm: float,
) -> float:
    """VED in J/mm^3."""
    if scan_speed_mm_s <= 0 or hatch_spacing_mm <= 0 or layer_thickness_mm <= 0:
        return float("nan")
    return laser_power_W / (scan_speed_mm_s * hatch_spacing_mm * layer_thickness_mm)


def linear_energy_density(laser_power_W: float, scan_speed_mm_s: float) -> float:
    """Linear energy density in J/mm."""
    if scan_speed_mm_s <= 0:
        return float("nan")
    return laser_power_W / scan_speed_mm_s


def at_percent_to_wt_fractions(ni_at_pct: float) -> Dict[str, float]:
    """
    Convert Ni at.% to Ni/Ti weight fractions.

    Input:
        ni_at_pct = atomic percent Ni, e.g. 51.3

    Output:
        {"Ni": wt_fraction_Ni, "Ti": wt_fraction_Ti}
    """
    ni_at_pct = _clamp(ni_at_pct, 0.0, 100.0)
    ti_at_pct = 100.0 - ni_at_pct

    ni_mass = ni_at_pct * MOLAR_MASS_G_MOL["Ni"]
    ti_mass = ti_at_pct * MOLAR_MASS_G_MOL["Ti"]
    total = ni_mass + ti_mass

    if total <= 0:
        return {"Ni": 0.0, "Ti": 0.0}

    return {"Ni": ni_mass / total, "Ti": ti_mass / total}


def wt_fractions_to_ni_at_percent(ni_wt: float, ti_wt: float) -> float:
    """Convert Ni/Ti weight fractions back to Ni at.%."""
    ni_wt = max(0.0, ni_wt)
    ti_wt = max(0.0, ti_wt)

    ni_moles = ni_wt / MOLAR_MASS_G_MOL["Ni"]
    ti_moles = ti_wt / MOLAR_MASS_G_MOL["Ti"]
    total_moles = ni_moles + ti_moles

    if total_moles <= 0:
        return float("nan")

    return 100.0 * ni_moles / total_moles


def estimate_peak_surface_temperature_K(p: VaporizationInput) -> float:
    """
    Screening proxy for top-surface peak temperature.

    It intentionally uses P, v, hatch, layer thickness, absorptivity and build
    plate temperature, not VED alone. This is not CFD; it is a monotonic proxy
    for comparing nearby LPBF conditions.
    """
    ved = volumetric_energy_density(
        p.laser_power_W,
        p.scan_speed_mm_s,
        p.hatch_spacing_mm,
        p.layer_thickness_mm,
    )

    if math.isnan(ved):
        return float("nan")

    layer_um = p.layer_thickness_mm * 1000.0

    # Baseline near NiTi melting range, increased by energy input.
    # Coefficients are deliberately conservative and transparent.
    temp_K = (
        1540.0
        + 5.5 * ved
        + 0.45 * (p.laser_power_W - 100.0)
        - 0.018 * (p.scan_speed_mm_s - 800.0)
        + 550.0 * (p.absorptivity - 0.35)
        - 0.55 * (layer_um - 30.0)
        + 0.15 * (p.build_plate_C - 80.0)
    )

    # Oxygen/plume effects are not solved. We add only a mild penalty because
    # high oxygen often accompanies less stable processing.
    temp_K += max(0.0, p.oxygen_ppm - 70.0) * 0.03

    return _clamp(temp_K, 1550.0, 3400.0)


def estimate_melt_pool_geometry_um(p: VaporizationInput) -> Dict[str, float]:
    """
    Estimate pool width, depth, length, surface area, and cross-section area.

    The cross-section is treated as a half ellipse.
    The top surface is treated as an ellipse.
    """
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

    width_um = beam * (0.80 + 0.0048 * ved + 0.0012 * (p.laser_power_W - 100.0))
    depth_um = layer_um * (1.00 + 0.0100 * ved + 0.0010 * (p.laser_power_W - 100.0))
    length_um = width_um * (1.55 + 0.0025 * ved)

    width_um = _clamp(width_um, 45.0, 450.0)
    depth_um = _clamp(depth_um, max(12.0, 0.6 * layer_um), 350.0)
    length_um = _clamp(length_um, 70.0, 1200.0)

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

    P(Tb) = 1 atm is used as the anchor point.
    """
    if element not in BOILING_K:
        raise ValueError(f"Unknown element: {element}")

    if temperature_K <= 0:
        return 0.0

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
    """
    Langmuir evaporation flux in mol/(m^2 s).
    """
    if partial_pressure_pa <= 0 or molar_mass_kg_mol <= 0 or temperature_K <= 0:
        return 0.0

    lam = _clamp(accommodation_coeff, 0.0, 1.0)
    denominator = math.sqrt(2.0 * math.pi * molar_mass_kg_mol * R_GAS * temperature_K)

    return lam * partial_pressure_pa / denominator


def remelting_multiplier(layer_thickness_mm: float, remelt_passes: int) -> float:
    """
    Transparent remelting correction.

    Thinner layers get more repeated thermal exposure for the same build height.
    Explicit remelt/rescan passes are added on top.
    """
    layer_thickness_mm = max(layer_thickness_mm, 0.005)

    layer_factor = (0.030 / layer_thickness_mm) ** 0.45
    explicit_pass_factor = 1.0 + max(0, remelt_passes)

    return _clamp(layer_factor * explicit_pass_factor, 0.40, 8.00)


def compute_vaporization_composition(p: VaporizationInput) -> Dict[str, float]:
    """
    Main calculation.

    Returns a dictionary with process descriptors, thermal proxy, vaporization
    outputs, final Ni at.%, transformation-shift proxy, and risk labels.
    """
    ved = volumetric_energy_density(
        p.laser_power_W,
        p.scan_speed_mm_s,
        p.hatch_spacing_mm,
        p.layer_thickness_mm,
    )
    led = linear_energy_density(p.laser_power_W, p.scan_speed_mm_s)
    temp_K = estimate_peak_surface_temperature_K(p)
    geom = estimate_melt_pool_geometry_um(p)

    if math.isnan(ved) or math.isnan(temp_K):
        raise ValueError("Invalid LPBF input parameters.")

    initial_wt = at_percent_to_wt_fractions(p.powder_Ni_at_pct)
    x_ni = _clamp(p.powder_Ni_at_pct / 100.0, 0.0, 1.0)
    x_ti = 1.0 - x_ni

    # Ideal first-pass activity approximation.
    # Later, replace this with CALPHAD/JMatPro/pycalphad values if available.
    activity = {
        "Ni": x_ni,
        "Ti": x_ti,
    }

    pure_p_ni = pure_vapor_pressure_pa("Ni", temp_K)
    pure_p_ti = pure_vapor_pressure_pa("Ti", temp_K)

    partial_p_ni = pure_p_ni * activity["Ni"]
    partial_p_ti = pure_p_ti * activity["Ti"]

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

    # Avoid impossible losses in extreme slider combinations.
    ni_loss_rate = min(ni_loss_rate, 0.50 * initial_ni_mass_rate)
    ti_loss_rate = min(ti_loss_rate, 0.50 * initial_ti_mass_rate)

    final_ni_mass_rate = max(0.0, initial_ni_mass_rate - ni_loss_rate)
    final_ti_mass_rate = max(0.0, initial_ti_mass_rate - ti_loss_rate)
    final_total = final_ni_mass_rate + final_ti_mass_rate

    if final_total <= 0:
        final_ni_wt = float("nan")
        final_ti_wt = float("nan")
        final_ni_at = float("nan")
    else:
        final_ni_wt = final_ni_mass_rate / final_total
        final_ti_wt = final_ti_mass_rate / final_total
        final_ni_at = wt_fractions_to_ni_at_percent(final_ni_wt, final_ti_wt)

    delta_ni_at = final_ni_at - p.powder_Ni_at_pct

    # Current app uses a strong simplified relation:
    # 0.1 at.% Ni increase roughly lowers transformation temperatures.
    # Therefore Ni loss raises Af/Ms. This is a screening sensitivity only.
    predicted_transform_shift_C = -100.0 * delta_ni_at

    composition_vulnerability_index = abs(predicted_transform_shift_C)

    if composition_vulnerability_index < 5.0:
        risk_label = "low composition-shift risk"
    elif composition_vulnerability_index < 20.0:
        risk_label = "moderate composition-shift risk"
    else:
        risk_label = "high Ni-loss / transformation-shift risk"

    if delta_ni_at < -0.20:
        action = (
            "Reduce peak thermal exposure: lower laser power, increase scan speed, "
            "avoid unnecessary remelting, or test a thicker layer if density remains acceptable."
        )
    elif delta_ni_at > 0.05:
        action = (
            "Mass balance predicts slight Ni enrichment, which is unusual for NiTi; "
            "check calibration, composition data, and vapor-pressure assumptions."
        )
    else:
        action = (
            "Composition shift is small in this screening model. Verify with EDS/ICP and DSC."
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
        "composition_vulnerability_index_C": composition_vulnerability_index,
        "risk_label": risk_label,
        "recommended_action": action,
    }


def sweep_power_speed(
    base: VaporizationInput,
    powers_W: List[float],
    speeds_mm_s: List[float],
) -> List[Dict[str, float]]:
    """
    Create a power-speed map while keeping hatch, layer thickness, composition,
    remelting, beam diameter, absorptivity, and calibration fixed.
    """
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
