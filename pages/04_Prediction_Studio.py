import streamlit as st
import pandas as pd
from src.process_models import ProcessInput, volumetric_energy_density, defect_risks, ni_evaporation_risk, estimate_composition_shift, transformation_temperature_rule, functional_score, process_window_label
from src.data_utils import load_exact_process_rows, load_paper_facts

st.title("Prediction Studio")
st.caption("For users who want to input their own process and obtain a structured screening report.")

with st.form("prediction_form"):
    st.subheader("Process and chemistry")
    c1,c2,c3,c4 = st.columns(4)
    P = c1.number_input("Laser power W", value=100.0)
    v = c2.number_input("Scan speed mm/s", value=800.0)
    h = c3.number_input("Hatch spacing mm", value=0.05, step=0.005, format="%.4f")
    t = c4.number_input("Layer thickness mm", value=0.03, step=0.005, format="%.4f")
    c5,c6,c7,c8 = st.columns(4)
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
    p = ProcessInput(P,v,h,t,powder_Ni,mNi,80,oxygen,remelt,heat_T,heat_min,service_T)
    ved = volumetric_energy_density(p)
    risks = defect_risks(ved)
    evap = ni_evaporation_risk(ved, remelt, oxygen)
    eff_Ni = estimate_composition_shift(powder_Ni, evap, mNi)
    temps = transformation_temperature_rule(eff_Ni, heat_T, heat_min)
    func = functional_score(temps, service_T, risks, target)
    report = pd.DataFrame([
        ["VED_J_mm3", ved],
        ["process_window", process_window_label(ved)],
        ["effective_Ni_at_pct", eff_Ni],
        ["Ni_loss_risk", evap],
        ["lack_of_fusion_risk", risks["lack_of_fusion"]],
        ["keyhole_risk", risks["keyhole"]],
        ["cracking_risk", risks["cracking"]],
        ["Ms_C", temps["Ms_C"]],
        ["Mf_C", temps["Mf_C"]],
        ["As_C", temps["As_C"]],
        ["Af_C", temps["Af_C"]],
        ["hysteresis_K", temps["hysteresis_K"]],
        ["functional_score", func["functional_score"]],
        ["functional_label", func["functional_label"]],
    ], columns=["metric","value"])
    st.dataframe(report, use_container_width=True)
    st.download_button("Download report CSV", report.to_csv(index=False), "niti_prediction_report.csv")

    bench = load_exact_process_rows()
    bench["distance"] = ((bench["laser_power_W"]-P)/100)**2 + ((bench["scan_speed_mm_s"]-v)/500)**2 + ((bench["VED_J_mm3"]-ved)/80)**2
    st.subheader("Closest benchmark rows")
    st.dataframe(bench.sort_values("distance").head(8), use_container_width=True)
