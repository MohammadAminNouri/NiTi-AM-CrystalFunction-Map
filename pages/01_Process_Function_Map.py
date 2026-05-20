import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.constants import APPLICATION_TARGETS
from src.process_models import (
    ProcessInput, volumetric_energy_density, linear_energy_density,
    defect_risks, ni_evaporation_risk, estimate_composition_shift,
    transformation_temperature_rule, functional_score, process_window_label
)
from src.data_utils import load_exact_process_rows
from src.scenario_engine import scenario_for_ved, format_scenario_markdown, load_scenarios

st.title("Process–Function Map")
st.caption("Input LPBF parameters and evaluate printability, Ni-loss risk, transformation window and functional scenario.")

with st.sidebar:
    app = st.selectbox("Target application", list(APPLICATION_TARGETS.keys()))
    service_T = st.number_input("Service temperature (°C)", -120.0, 250.0, APPLICATION_TARGETS[app]["service_temperature_C"])
    target_mode = st.radio("Functional target", ["superelastic", "thermal actuation"], horizontal=False)
    st.info(APPLICATION_TARGETS[app]["Af_rule"])

st.markdown("""
This page should be read as a **screening map**, not as a certified process qualification.  
The logic is metallurgical: LPBF parameters modify melt stability, defects, Ni evaporation, oxygen pickup and cooling history; those changes shift B2/B19′ phase balance and therefore the usable superelastic or shape-memory response.
""")

c1, c2, c3, c4 = st.columns(4)
P = c1.slider("Laser power P (W)", 40, 500, 100, 5)
v = c2.slider("Scan speed v (mm/s)", 100, 2500, 800, 10)
h = c3.slider("Hatch spacing h (mm)", 0.03, 0.18, 0.05, 0.005)
t = c4.slider("Layer thickness t (mm)", 0.01, 0.08, 0.03, 0.005)

c5, c6, c7, c8 = st.columns(4)
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

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("VED", f"{ved:.1f} J/mm³")
m2.metric("Linear ED", f"{led:.3f} J/mm")
m3.metric("Process label", process_window_label(ved))
m4.metric("Estimated effective Ni", f"{eff_Ni:.2f} at.%")
m5.metric("Af estimate", f"{temps['Af_C']:.1f} °C")
m6.metric("Function score", f"{func['functional_score']:.2f}")

st.subheader("Metallurgical scenario")
scenario = scenario_for_ved(ved)
st.markdown(format_scenario_markdown(scenario))

with st.expander("Why this scenario matters"):
    st.markdown("""
- **Low energy** usually fails before functional design starts: lack of fusion or cracking dominates.
- **Moderate energy** can preserve enough Ni and austenite stability to give superelastic response.
- **High energy** may improve melting but increases Ni evaporation. Since NiTi transformation temperatures are extremely composition-sensitive, small Ni loss can move Af/Ms above service temperature.
- **VED is not sufficient alone.** Keep laser power, scan speed, hatch spacing and layer thickness as separate features because equal VED can produce different melt-pool lifetime, cooling rate and evaporation behavior.
""")

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
Mf, Ms, As, Af = temps["Mf_C"], temps["Ms_C"], temps["As_C"], temps["Af_C"]
xmin = min(Mf, Ms, As, Af, service_T) - 35
xmax = max(Mf, Ms, As, Af, service_T) + 35

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=[Mf, Ms], y=[1, 1], mode="lines+markers",
    line=dict(width=18), marker=dict(size=10),
    name="Martensite interval Mf→Ms",
    hovertemplate="Martensite interval<br>%{x:.1f} °C<extra></extra>"
))
fig2.add_trace(go.Scatter(
    x=[As, Af], y=[2, 2], mode="lines+markers",
    line=dict(width=18), marker=dict(size=10),
    name="Austenite interval As→Af",
    hovertemplate="Austenite interval<br>%{x:.1f} °C<extra></extra>"
))
fig2.add_vline(x=service_T, line_dash="dot", line_width=3, annotation_text="service")
for label, val in [("Mf", Mf), ("Ms", Ms), ("As", As), ("Af", Af)]:
    fig2.add_vline(x=val, line_dash="dash", opacity=0.35, annotation_text=label)

fig2.update_layout(
    height=340,
    xaxis_title="Temperature (°C)",
    xaxis=dict(range=[xmin, xmax]),
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2],
        ticktext=["Cooling: martensite formation", "Heating: austenite recovery"],
        range=[0.4, 2.6]
    ),
    margin=dict(l=20, r=20, t=30, b=30),
)
st.plotly_chart(fig2, use_container_width=True)

if target_mode == "superelastic":
    if service_T > Af:
        st.success("Screening interpretation: service temperature is above Af, so the material can be austenitic at service if defects and hysteresis are controlled.")
    elif service_T < Ms:
        st.warning("Screening interpretation: service temperature is below Ms, so the material is likely martensitic at service; superelasticity is unlikely without composition/heat-treatment correction.")
    else:
        st.info("Screening interpretation: service temperature lies inside the transformation interval. Expect mixed phase or unstable functional response unless verified by DSC.")
else:
    st.info("For thermal actuation, Af should be close to the intended activation temperature, while hysteresis should remain acceptable for cycling.")

st.subheader("Nearest exact process rows")
bench = load_exact_process_rows()
bench["distance"] = ((bench["laser_power_W"] - P) / 100) ** 2 + ((bench["scan_speed_mm_s"] - v) / 500) ** 2 + ((bench["VED_J_mm3"] - ved) / 80) ** 2
st.dataframe(
    bench.sort_values("distance").head(8)[
        ["sample_id", "laser_power_W", "scan_speed_mm_s", "hatch_spacing_mm", "layer_thickness_mm", "VED_J_mm3", "functional_class", "functional_observation", "microstructure_observation", "source_id"]
    ],
    use_container_width=True
)

with st.expander("Scenario library"):
    st.dataframe(load_scenarios(), use_container_width=True, height=360)

with st.expander("References used by this page"):
    st.markdown("""
- Xiang et al., 2024, **Metals**, LPBF Ni-rich Ni51.3Ti48.7 process/superelasticity map, DOI: `10.3390/met14090961`.
- Xue et al., 2022, **Acta Materialia**, defect-free LPBF NiTi with tensile superelasticity up to 6%, DOI: `10.1016/j.actamat.2022.117781`.
- Li et al., 2024, **Virtual and Physical Prototyping**, 195-entry / 23-publication ML process optimization, DOI: `10.1080/17452759.2024.2364221`.
- Carcavilla et al., 2026, **Scientific Reports**, large power-speed study emphasizing nonlinear process effects, DOI: `10.1038/s41598-026-44034-1`.
""")
