import streamlit as st
import pandas as pd
from src.process_models import (
    ProcessInput, volumetric_energy_density, defect_risks, ni_evaporation_risk,
    estimate_composition_shift, transformation_temperature_rule, functional_score,
    process_window_label
)
from src.data_utils import load_exact_process_rows
from src.scenario_engine import scenario_for_ved, format_scenario_markdown

st.title("Prediction Studio")
st.caption("Input a process condition and obtain a structured screening report with benchmark comparison.")

st.markdown("""
This page is for a user who wants to test a new LPBF NiTi condition before printing or before interpreting experimental results.
It reports **screening-level** outputs and tells the user which measurements are required to validate the prediction.
""")

with st.form("prediction_form"):
    st.subheader("Process and chemistry")
    c1, c2, c3, c4 = st.columns(4)
    P = c1.number_input("Laser power W", value=100.0)
    v = c2.number_input("Scan speed mm/s", value=800.0)
    h = c3.number_input("Hatch spacing mm", value=0.05, step=0.005, format="%.4f")
    t = c4.number_input("Layer thickness mm", value=0.03, step=0.005, format="%.4f")

    c5, c6, c7, c8 = st.columns(4)
    powder_Ni = c5.number_input("Powder Ni at.%", value=51.3, step=0.01)
    measured_Ni = c5.number_input("Measured/EDS Ni at.% (optional; set 0 to ignore)", value=0.0, step=0.01)
    oxygen = c6.number_input("Oxygen ppm", value=70.0)
    remelt = c6.number_input("Remelt passes", value=0, min_value=0)
    heat_T = c7.number_input("Heat treatment T °C", value=500.0)
    heat_min = c7.number_input("Heat treatment min", value=30.0)
    service_T = c8.number_input("Service temperature °C", value=37.0)
    target = c8.selectbox("Function target", ["superelastic", "thermal actuation"])
    submitted = st.form_submit_button("Run screening")

if submitted:
    mNi = None if measured_Ni == 0 else measured_Ni
    p = ProcessInput(P, v, h, t, powder_Ni, mNi, 80, oxygen, remelt, heat_T, heat_min, service_T)
    ved = volumetric_energy_density(p)
    risks = defect_risks(ved)
    evap = ni_evaporation_risk(ved, remelt, oxygen)
    eff_Ni = estimate_composition_shift(powder_Ni, evap, mNi)
    temps = transformation_temperature_rule(eff_Ni, heat_T, heat_min)
    func = functional_score(temps, service_T, risks, target)
    scenario = scenario_for_ved(ved)

    st.subheader("Result summary")
    report = pd.DataFrame([
        ["VED_J_mm3", ved, "Calculated as P/(v·h·t). Useful but not sufficient alone."],
        ["process_window", process_window_label(ved), "Nearest known process-window label."],
        ["effective_Ni_at_pct", eff_Ni, "Measured Ni used if supplied; otherwise screening estimate from Ni-loss risk."],
        ["Ni_loss_risk", evap, "Increases with high VED, remelting and oxygen."],
        ["lack_of_fusion_risk", risks["lack_of_fusion"], "Dominant at low energy."],
        ["keyhole_risk", risks["keyhole"], "Dominant at high energy."],
        ["cracking_risk", risks["cracking"], "Screening risk at insufficient bonding."],
        ["Ms_C", temps["Ms_C"], "Start of martensite formation on cooling."],
        ["Mf_C", temps["Mf_C"], "Finish of martensite formation on cooling."],
        ["As_C", temps["As_C"], "Start of austenite recovery on heating."],
        ["Af_C", temps["Af_C"], "Finish of austenite recovery on heating."],
        ["hysteresis_K", temps["hysteresis_K"], "Wide hysteresis can reduce cycling stability."],
        ["functional_score", func["functional_score"], "0-1 screening score, not certification."],
        ["functional_label", func["functional_label"], "Qualitative screening label."],
    ], columns=["metric", "value", "meaning"])
    st.dataframe(report, use_container_width=True)
    st.download_button("Download report CSV", report.to_csv(index=False), "niti_prediction_report.csv")

    st.subheader("Scenario interpretation")
    st.markdown(format_scenario_markdown(scenario))

    st.subheader("What should be measured next?")
    st.markdown("""
1. **Density/porosity** from polished cross-sections or CT.  
2. **Composition** by EDS/ICP, especially Ni loss relative to powder.  
3. **DSC** for Ms/Mf/As/Af and hysteresis.  
4. **XRD** for B2/B19′ and secondary phases.  
5. **EBSD/TKD** if variant selection, texture or parent-grain inheritance matters.  
6. **Cyclic compression/tension** to confirm recoverable strain and residual strain.
""")

    bench = load_exact_process_rows()
    bench["distance"] = ((bench["laser_power_W"] - P) / 100) ** 2 + ((bench["scan_speed_mm_s"] - v) / 500) ** 2 + ((bench["VED_J_mm3"] - ved) / 80) ** 2
    st.subheader("Closest exact benchmark rows")
    st.dataframe(
        bench.sort_values("distance").head(8)[
            ["sample_id", "laser_power_W", "scan_speed_mm_s", "VED_J_mm3", "functional_class", "functional_observation", "microstructure_observation", "notes"]
        ],
        use_container_width=True
    )
