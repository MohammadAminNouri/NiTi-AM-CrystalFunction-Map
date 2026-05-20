from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.constants import PROCESS_RULES


@dataclass
class ProcessInput:
    laser_power_W: float
    scan_speed_mm_s: float
    hatch_spacing_mm: float
    layer_thickness_mm: float
    powder_Ni_at_pct: float
    measured_Ni_at_pct: Optional[float]
    build_plate_C: float
    oxygen_ppm: float
    remelt_passes: int
    heat_treatment_C: float
    heat_treatment_min: float
    service_temperature_C: float


def volumetric_energy_density(p: ProcessInput) -> float:
    if p.scan_speed_mm_s <= 0:
        return float("nan")

    if p.hatch_spacing_mm <= 0:
        return float("nan")

    if p.layer_thickness_mm <= 0:
        return float("nan")

    return (
        p.laser_power_W
        / (p.scan_speed_mm_s * p.hatch_spacing_mm * p.layer_thickness_mm)
    )


def linear_energy_density(p: ProcessInput) -> float:
    if p.scan_speed_mm_s <= 0:
        return float("nan")

    return p.laser_power_W / p.scan_speed_mm_s


def process_window_label(ved: float) -> str:
    if np.isnan(ved):
        return "invalid"

    if ved < PROCESS_RULES["low_energy_lack_of_fusion_J_mm3"]:
        return "low-energy / lack-of-fusion or cracking risk"

    if (
        PROCESS_RULES["low_energy_superelastic_min_J_mm3"]
        <= ved
        <= PROCESS_RULES["low_energy_superelastic_max_J_mm3"]
    ):
        return "reported low-energy superelastic window"

    if ved >= PROCESS_RULES["high_energy_poor_superelastic_min_J_mm3"]:
        return "high-energy Ni-loss / poor-superelasticity risk"

    if ved >= PROCESS_RULES["keyhole_onset_reported_J_mm3"]:
        return "transition to keyhole / partial response risk"

    return "intermediate screening window"


def ni_evaporation_risk(
    ved: float,
    remelt_passes: int = 0,
    oxygen_ppm: float = 70.0,
) -> float:
    if np.isnan(ved):
        return 1.0

    x = (
        (ved - 116.7) / 35.0
        + 0.35 * remelt_passes
        + max(0.0, oxygen_ppm - 70.0) / 400.0
    )

    return float(1.0 / (1.0 + np.exp(-x)))


def defect_risks(ved: float) -> dict:
    if np.isnan(ved):
        return {
            "lack_of_fusion": 1.0,
            "keyhole": 1.0,
            "cracking": 1.0,
            "stable_window": 0.0,
        }

    lack_of_fusion = float(1.0 / (1.0 + np.exp((ved - 55.0) / 8.0)))
    keyhole = float(1.0 / (1.0 + np.exp(-(ved - 133.3) / 12.0)))
    cracking = float(1.0 / (1.0 + np.exp((ved - 60.0) / 7.0)))

    stable_window = float(
        max(
            0.0,
            1.0 - max(lack_of_fusion, keyhole, 0.5 * cracking),
        )
    )

    return {
        "lack_of_fusion": lack_of_fusion,
        "keyhole": keyhole,
        "cracking": cracking,
        "stable_window": stable_window,
    }


def estimate_composition_shift(
    powder_Ni_at_pct: float,
    ni_loss_risk: float,
    measured_Ni_at_pct: Optional[float] = None,
) -> float:
    if measured_Ni_at_pct is not None and not np.isnan(measured_Ni_at_pct):
        return float(measured_Ni_at_pct)

    max_loss_atpct = 0.75

    return float(powder_Ni_at_pct - max_loss_atpct * ni_loss_risk)


def transformation_temperature_rule(
    effective_Ni_at_pct: float,
    heat_treatment_C: float,
    heat_treatment_min: float,
) -> dict:
    """
    Screening rule only.

    Around near-equiatomic NiTi, transformation temperature can be extremely
    sensitive to Ni content. This simplified model is used only to make the app
    interactive until enough row-level DSC data are added.
    """

    dNi_tenths = (effective_Ni_at_pct - 50.0) / 0.1

    heat_factor = (
        np.clip((heat_treatment_C - 450.0) / 200.0, -1.0, 1.0)
        * np.log1p(max(heat_treatment_min, 0.0))
        / np.log(121.0)
    )

    Ms = 45.0 - 10.0 * dNi_tenths - 6.0 * heat_factor
    Mf = Ms - 12.0
    As = Ms + 18.0
    Af = As + 35.0

    return {
        "Ms_C": float(Ms),
        "Mf_C": float(Mf),
        "As_C": float(As),
        "Af_C": float(Af),
        "hysteresis_K": float(Af - Ms),
    }


def functional_score(
    temps: dict,
    service_temperature_C: float,
    risks: dict,
    target: str = "superelastic",
) -> dict:
    Af = temps["Af_C"]
    Ms = temps["Ms_C"]
    hysteresis = temps["hysteresis_K"]

    defect_penalty = max(
        risks["lack_of_fusion"],
        risks["keyhole"],
        risks["cracking"],
    )

    if target == "superelastic":
        temperature_score = 1.0 / (
            1.0 + np.exp(-(service_temperature_C - Af) / 8.0)
        )
    else:
        temperature_score = np.exp(-abs(service_temperature_C - Af) / 35.0)

    hysteresis_score = 1.0 / (1.0 + np.exp((hysteresis - 35.0) / 8.0))

    score = float(
        np.clip(
            0.50 * temperature_score
            + 0.25 * hysteresis_score
            + 0.25 * (1.0 - defect_penalty),
            0.0,
            1.0,
        )
    )

    if service_temperature_C < Ms and target == "superelastic":
        label = "high martensite-at-service risk"
    elif score > 0.72:
        label = "strong functional candidate"
    elif score > 0.48:
        label = "conditional candidate; verify experimentally"
    else:
        label = "high functional risk"

    return {
        "functional_score": score,
        "functional_label": label,
    }
