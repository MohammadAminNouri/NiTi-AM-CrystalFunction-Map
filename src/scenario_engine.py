import pandas as pd
from pathlib import Path


def load_scenarios():
    return pd.read_csv(Path("data/scenario_rules.csv"))


def scenario_for_ved(ved: float):
    scenarios = load_scenarios()
    numeric = scenarios.dropna(subset=["VED_min_J_mm3", "VED_max_J_mm3"]).copy()

    hit = numeric[
        (numeric["VED_min_J_mm3"].astype(float) <= ved)
        & (ved <= numeric["VED_max_J_mm3"].astype(float))
    ]

    if len(hit):
        return hit.iloc[0].to_dict()

    if ved < 66.7:
        return numeric.iloc[(numeric["VED_max_J_mm3"].astype(float) - ved).abs().argmin()].to_dict()

    if ved > 155.6:
        return numeric.iloc[(numeric["VED_min_J_mm3"].astype(float) - ved).abs().argmin()].to_dict()

    return {
        "scenario": "Between benchmark windows",
        "metallurgical_logic": "This process sits between extracted benchmark windows. Treat the output as an interpolation zone, not a validated window.",
        "expected_observation": "Measure density, Ni composition, DSC and cyclic response before assigning functional class.",
        "functional_consequence": "Conditional; closest literature rows should be inspected.",
        "example_case": "No direct exact row is assigned to this specific interval.",
        "source": "screening_rule",
    }


def format_scenario_markdown(scenario: dict) -> str:
    return f"""
### {scenario.get("scenario", "Scenario")}

**Metallurgical logic**  
{scenario.get("metallurgical_logic", "")}

**Expected observation**  
{scenario.get("expected_observation", "")}

**Functional consequence**  
{scenario.get("functional_consequence", "")}

**Example / evidence**  
{scenario.get("example_case", "")}
"""
