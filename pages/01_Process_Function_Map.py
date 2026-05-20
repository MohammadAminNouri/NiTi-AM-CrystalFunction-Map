import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.constants import APPLICATION_TARGETS, PROCESS_RULES
from src.process_models import ProcessInput, volumetric_energy_density, linear_energy_density, defect_risks, ni_evaporation_risk, estimate_composition_shift, transformation_temperature_rule, functional_score, process_window_label
from src.data_utils import load_exact_process_rows

st.title("Process-Function Map")
st.caption("Input LPBF parameters and evaluate printability, Ni-loss risk and functional window.")

with st.sidebar:
    app = st.selectbox("Target application", list(APPLICATION_TARGETS.keys()))
    service_T = st.number_input("Service temperature (°C)", -120.0, 250.0, APPLICATION_TARGETS[app]["service_temperature_C"])
    target_mode = st.radio("Functional target", ["superelastic", "thermal actuation"], horizontal=False)
    st.info(APPLICATION_TARGETS[app]["Af_rule"])

c1,c2,c3,c4 = st.columns(4)
P = c1.slider("Laser power P (W)", 40, 500, 100, 5)
v = c2.slider("Scan speed v (mm/s)", 100, 2500, 800, 10)
h = c3.slider("Hatch spacing h (mm)", 0.03, 0.18, 0.05, 0.005)
t = c4.slider("Layer thickness t (mm)", 0.01, 0.08, 0.03, 0.005)

c5,c6,c7,c8 = st.columns(4)
powder_Ni = c5.slider("Powder Ni (at.%)", 49.0, 52.0, 51.30, 0.01)
measured_Ni_on = c5.checkbox("Use measured Ni at.%")
measured_Ni = c5.number_input("Measured Ni at.%", 45.0, 55.0, powder_Ni, 0.01) if measured_Ni_on else None
oxygen = c6.slider("Chamber oxygen (ppm)", 1, 1000, 70, 1)
remelt = c6.slider("Remelt/rescan passes", 0, 5, 0, 1)
heat_T = c7.slider("Heat treatment T (°C)", 20, 900, 500, 10)
heat_min = c7.slider("Heat treatment time (min)", 0, 1440, 30, 10)
build_T = c8.slider("Build plate T (°C)", 20, 600, 80, 10)

proc = ProcessInput(P, v, h, t, powder_Ni, measured_Ni, build_T, oxygen, remelt, heat_T, heat_min, service_T)
ved = volumetric_energy_density(proc)
led = linear_energy_density(proc)
risks = defect_risks(ved)
evap = ni_evaporation_risk(ved, remelt, oxygen)
eff_Ni = estimate_composition_shift(powder_Ni, evap, measured_Ni)
temps = transformation_temperature_rule(eff_Ni, heat_T, heat_min)
func = functional_score(temps, service_T, risks, target_mode)

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("VED", f"{ved:.1f} J/mm³")
m2.metric("Linear ED", f"{led:.3f} J/mm")
m3.metric("Process label", process_window_label(ved))
m4.metric("Ni-loss risk", f"{evap:.2f}")
m5.metric("Af estimate", f"{temps['Af_C']:.1f} °C")
m6.metric("Function score", f"{func['functional_score']:.2f}")

st.subheader("Risk map around current hatch/layer setting")
powers = list(range(60, 421, 20))
speeds = list(range(250, 2001, 75))
z = []
for pp in powers:
    row = []
    for vv in speeds:
        p2 = ProcessInput(pp, vv, h, t, powder_Ni, measured_Ni, build_T, oxygen, remelt, heat_T, heat_min, service_T)
        ved2 = volumetric_energy_density(p2)
        r = defect_risks(ved2)
        e = ni_evaporation_risk(ved2, remelt, oxygen)
        row.append(max(r["lack_of_fusion"], r["keyhole"], r["cracking"], e))
    z.append(row)

fig = go.Figure(data=go.Heatmap(x=speeds, y=powers, z=z, colorbar=dict(title="max risk")))
fig.add_trace(go.Scatter(x=[v], y=[P], mode="markers", marker=dict(size=16, symbol="x"), name="current"))
fig.update_layout(height=520, xaxis_title="Scan speed (mm/s)", yaxis_title="Laser power (W)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Transformation window")
fig2 = go.Figure()
for k, val in temps.items():
    if k.endswith("_C"):
        fig2.add_vline(x=val, line_dash="dash", annotation_text=k.replace("_C",""))
fig2.add_vrect(x0=service_T-2, x1=service_T+2, fillcolor="gray", opacity=0.15, annotation_text="service")
fig2.update_layout(height=250, xaxis_title="Temperature (°C)", yaxis_visible=False)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Nearest exact process rows")
bench = load_exact_process_rows()
bench["distance"] = ((bench["laser_power_W"]-P)/100)**2 + ((bench["scan_speed_mm_s"]-v)/500)**2 + ((bench["VED_J_mm3"]-ved)/80)**2
st.dataframe(bench.sort_values("distance").head(5)[["sample_id","laser_power_W","scan_speed_mm_s","hatch_spacing_mm","layer_thickness_mm","VED_J_mm3","functional_observation","microstructure_observation","source_id"]], use_container_width=True)
