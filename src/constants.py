APP_TITLE = "NiTi-AM CrystalFunction Map"
APP_SUBTITLE = "Benchmark-oriented process-crystallography-function platform for additively manufactured NiTi"

NITI_DEFAULTS = {
    "B2_a_A": 3.015,
    "B19p_a_A": 2.89,
    "B19p_b_A": 4.12,
    "B19p_c_A": 4.62,
    "B19p_beta_deg": 96.8,
    "natural_OR_planes": "(010)B19′ // (110)B2",
    "natural_OR_directions": "[101]B19′ // [111]B2",
    "candidate_habit_plane": "(112)B2 // (101)B19′",
    "reference_Ms_C": 45.0,
    "reference_Mf_C": 35.0,
    "reference_As_C": 60.0,
    "reference_Af_C": 90.0,
}

APPLICATION_TARGETS = {
    "Biomedical superelastic implant": {
        "service_temperature_C": 37.0,
        "Af_rule": "Af should generally be below or close to body temperature for stable superelasticity.",
        "priorities": ["low porosity", "low hysteresis", "stable austenite at service", "fatigue resistance"]
    },
    "Thermal actuator": {
        "service_temperature_C": 60.0,
        "Af_rule": "Af should be near the intended activation temperature.",
        "priorities": ["repeatable thermal actuation", "controlled hysteresis", "good recovery strain"]
    },
    "Damping lattice": {
        "service_temperature_C": 25.0,
        "Af_rule": "Transformation should occur near the mechanical service window.",
        "priorities": ["stress-induced transformation", "energy dissipation", "texture control"]
    },
    "Aerospace deployable component": {
        "service_temperature_C": -20.0,
        "Af_rule": "Transformation window must match deployment environment.",
        "priorities": ["predictability", "low defect density", "cycling stability"]
    },
}

PROCESS_RULES = {
    "low_energy_lack_of_fusion_J_mm3": 55.0,
    "low_energy_superelastic_min_J_mm3": 66.7,
    "low_energy_superelastic_max_J_mm3": 116.7,
    "keyhole_onset_reported_J_mm3": 133.3,
    "high_energy_poor_superelastic_min_J_mm3": 166.7,
    "high_energy_poor_superelastic_max_J_mm3": 233.3,
    "oxygen_ppm_reference": 70.0,
}
