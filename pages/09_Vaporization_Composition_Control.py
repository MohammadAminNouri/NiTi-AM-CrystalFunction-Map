import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.vaporization_model import (
    VaporizationInput,
    compute_vaporization_composition,
    sweep_power_speed,
)
from src.process_models import transformation_temperature_rule


st.set_page_config(
    page_title="Vaporization–Composition Control",
    layout="wide",
)

st.title("Vaporization–Composition Control")
st.caption(
    "Physics-informed screening page for Ni evaporation, effective Ni/Ti ratio, "
    "transformation-temperature shift, and functional risk in LPBF NiTi."
)

st.markdown(
    """
This page upgrades the old **Ni-loss risk** idea into a transparent
**vaporization → composition → transformation** calculation.

It is inspired by integrated LPBF vaporization modeling, but this is still a
**screening surrogate**, not a CFD/CALPHAD-certified process model. The useful
thing is that it shows *why* a process condition may move NiTi away from the
desired B2/austenite superelastic window.
"""
)

with st.sidebar:
    st.header("LPBF inputs")

    P = st.slider("Laser power P (W)", 40, 500, 100, 5)
    v = st.slider("Scan speed v (mm/s)", 100, 2500, 800, 10)
    h = st.slider("Hatch spacing h (mm)", 0.03, 0.18, 0.05, 0.005)
    t = st.slider("Layer thickness t (mm)", 0.01, 0.08, 0.03, 0.005)

    st.header("NiTi and machine inputs")
    powder_Ni = st.slider("Powder Ni (at.%)", 49.00, 52.00, 51.30, 0.01)
    beam = st.slider("Beam diameter (µm)", 40, 160, 80, 5)
    absorptivity = st.slider("Effective absorptivity", 0.15, 0.70, 0.35, 0.01)
    build_T = st.slider("Build plate temperature (°C)", 20, 600, 80, 10)

    st.header("Atmosphere / remelting")
    oxygen = st.slider("Oxygen level (ppm)", 1, 1000, 70, 1)
    remelt = st.slider("Remelt/rescan passes", 0, 5, 0, 1)

    st.header("Model calibration")
    accommodation = st.slider(
        "Accommodation coefficient λ",
        0.01,
        1.00,
        0.25,
        0.01,
        help=(
            "Higher λ means stronger evaporation. Keep adjustable because "
            "real LPBF evaporation depends on shielding gas, pressure, plume, "
            "surface condition, and calibration against measured composition."
        ),
    )
    calibration = st.slider(
        "Global calibration scale",
        0.10,
        5.00,
        1.00,
        0.05,
        help=(
            "Use 1.0 by default. Change only when calibrating against measured "
            "ICP/EDS composition data."
        ),
    )

    st.header("Transformation check")
    heat_T = st.slider("Heat treatment T (°C)", 20, 900, 500, 10)
    heat_min = st.slider("Heat treatment time (min)", 0, 1440, 30, 10)
    service_T = st.slider("Service temperature (°C)", -120, 250, 25, 1)


inp = VaporizationInput(
    laser_power_W=P,
    scan_speed_mm_s=v,
    hatch_spacing_mm=h,
    layer_thickness_mm=t,
    powder_Ni_at_pct=powder_Ni,
    beam_diameter_um=beam,
    absorptivity=absorptivity,
    build_plate_C=build_T,
    remelt_passes=remelt,
    oxygen_ppm=oxygen,
    accommodation_coeff=accommodation,
    calibration_scale=calibration,
)

out = compute_vaporization_composition(inp)

st.subheader("Main screening result")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("VED", f"{out['VED_J_mm3']:.1f} J/mm³")
m2.metric("Linear ED", f"{out['linear_energy_density_J_mm']:.3f} J/mm")
m3.metric("Peak T proxy", f"{out['peak_temperature_C']:.0f} °C")
m4.metric("Final Ni", f"{out['final_Ni_at_pct']:.3f} at.%")
m5.metric("ΔNi", f"{out['delta_Ni_at_pct']:+.3f} at.%")
m6.metric("Predicted TT shift", f"{out['predicted_transformation_shift_C']:+.1f} °C")

if "high" in out["risk_label"]:
    st.error(out["risk_label"])
elif "moderate" in out["risk_label"]:
    st.warning(out["risk_label"])
else:
    st.success(out["risk_label"])

st.info(out["recommended_action"])

st.divider()

left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("What the model is doing")

    st.markdown(
        """
The calculation follows this chain:

1. Calculate VED and linear energy density.
2. Estimate a peak top-surface temperature from P, v, h, layer thickness,
   absorptivity and build-plate temperature.
3. Estimate melt-pool width, depth and length.
4. Estimate Ni and Ti vapor pressures.
5. Use a Langmuir-style evaporation flux.
6. Convert vaporized Ni/Ti mass into final Ni at.% by mass balance.
7. Convert the Ni shift into a transformation-temperature shift proxy.
"""
    )

    summary = pd.DataFrame(
        [
            ["Initial Ni", f"{out['initial_Ni_at_pct']:.3f} at.%"],
            ["Final Ni", f"{out['final_Ni_at_pct']:.3f} at.%"],
            ["Ni change", f"{out['delta_Ni_at_pct']:+.3f} at.%"],
            ["Initial Ni", f"{out['initial_Ni_wt_pct']:.3f} wt.%"],
            ["Final Ni", f"{out['final_Ni_wt_pct']:.3f} wt.%"],
            ["Ni loss", f"{out['Ni_loss_percent_of_initial_Ni_mass']:.4f}% of initial Ni mass"],
            ["Ti loss", f"{out['Ti_loss_percent_of_initial_Ti_mass']:.4f}% of initial Ti mass"],
            ["Remelting multiplier", f"{out['remelting_multiplier']:.2f}×"],
            ["Composition vulnerability index", f"{out['composition_vulnerability_index_C']:.1f} °C equivalent"],
        ],
        columns=["Output", "Value"],
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

with right:
    st.subheader("Melt-pool and vaporization descriptors")

    desc = pd.DataFrame(
        [
            ["Pool width", f"{out['melt_pool_width_um']:.1f} µm"],
            ["Pool depth", f"{out['melt_pool_depth_um']:.1f} µm"],
            ["Pool length", f"{out['melt_pool_length_um']:.1f} µm"],
            ["Pure vapor pressure Ni", f"{out['pure_vapor_pressure_Ni_Pa']:.3e} Pa"],
            ["Pure vapor pressure Ti", f"{out['pure_vapor_pressure_Ti_Pa']:.3e} Pa"],
            ["Partial pressure Ni", f"{out['partial_pressure_Ni_Pa']:.3e} Pa"],
            ["Partial pressure Ti", f"{out['partial_pressure_Ti_Pa']:.3e} Pa"],
            ["Ni flux", f"{out['flux_Ni_mol_m2_s']:.3e} mol/m²/s"],
            ["Ti flux", f"{out['flux_Ti_mol_m2_s']:.3e} mol/m²/s"],
        ],
        columns=["Descriptor", "Value"],
    )
    st.dataframe(desc, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Power–speed composition-shift map")

powers = list(range(60, 421, 20))
speeds = list(range(250, 2051, 75))

rows = sweep_power_speed(inp, powers, speeds)
df_map = pd.DataFrame(rows)

z_delta = []
z_cvi = []

for power in powers:
    row_delta = []
    row_cvi = []
    for speed in speeds:
        hit = df_map[
            (df_map["laser_power_W"] == power)
            & (df_map["scan_speed_mm_s"] == speed)
        ].iloc[0]
        row_delta.append(hit["delta_Ni_at_pct"])
        row_cvi.append(hit["composition_vulnerability_index_C"])
    z_delta.append(row_delta)
    z_cvi.append(row_cvi)

tab1, tab2 = st.tabs(["ΔNi at.% map", "Transformation-shift risk map"])

with tab1:
    fig = go.Figure(
        data=go.Heatmap(
            x=speeds,
            y=powers,
            z=z_delta,
            colorbar=dict(title="ΔNi at.%"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[v],
            y=[P],
            mode="markers",
            marker=dict(size=16, symbol="x"),
            name="current condition",
        )
    )
    fig.update_layout(
        height=560,
        xaxis_title="Scan speed (mm/s)",
        yaxis_title="Laser power (W)",
        title="Predicted Ni composition change",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure(
        data=go.Heatmap(
            x=speeds,
            y=powers,
            z=z_cvi,
            colorbar=dict(title="°C equivalent"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[v],
            y=[P],
            mode="markers",
            marker=dict(size=16, symbol="x"),
            name="current condition",
        )
    )
    fig.update_layout(
        height=560,
        xaxis_title="Scan speed (mm/s)",
        yaxis_title="Laser power (W)",
        title="Composition vulnerability index",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Transformation-window consequence")

temps_initial = transformation_temperature_rule(powder_Ni, heat_T, heat_min)
temps_final = transformation_temperature_rule(out["final_Ni_at_pct"], heat_T, heat_min)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Initial Af", f"{temps_initial['Af_C']:.1f} °C")
c2.metric("Final Af", f"{temps_final['Af_C']:.1f} °C")
c3.metric("Initial Ms", f"{temps_initial['Ms_C']:.1f} °C")
c4.metric("Final Ms", f"{temps_final['Ms_C']:.1f} °C")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=[temps_initial["Mf_C"], temps_initial["Ms_C"]],
        y=[1, 1],
        mode="lines+markers",
        line=dict(width=14),
        name="Initial cooling Mf→Ms",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[temps_initial["As_C"], temps_initial["Af_C"]],
        y=[2, 2],
        mode="lines+markers",
        line=dict(width=14),
        name="Initial heating As→Af",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[temps_final["Mf_C"], temps_final["Ms_C"]],
        y=[3, 3],
        mode="lines+markers",
        line=dict(width=14),
        name="After vaporization cooling Mf→Ms",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[temps_final["As_C"], temps_final["Af_C"]],
        y=[4, 4],
        mode="lines+markers",
        line=dict(width=14),
        name="After vaporization heating As→Af",
    )
)

fig2.add_vline(
    x=service_T,
    line_dash="dot",
    line_width=3,
    annotation_text="service T",
)

xmin = min(
    temps_initial["Mf_C"],
    temps_initial["Ms_C"],
    temps_initial["As_C"],
    temps_initial["Af_C"],
    temps_final["Mf_C"],
    temps_final["Ms_C"],
    temps_final["As_C"],
    temps_final["Af_C"],
    service_T,
) - 35

xmax = max(
    temps_initial["Mf_C"],
    temps_initial["Ms_C"],
    temps_initial["As_C"],
    temps_initial["Af_C"],
    temps_final["Mf_C"],
    temps_final["Ms_C"],
    temps_final["As_C"],
    temps_final["Af_C"],
    service_T,
) + 35

fig2.update_layout(
    height=430,
    xaxis_title="Temperature (°C)",
    xaxis=dict(range=[xmin, xmax]),
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=[
            "Initial cooling",
            "Initial heating",
            "After vaporization cooling",
            "After vaporization heating",
        ],
    ),
    margin=dict(l=20, r=20, t=30, b=30),
)

st.plotly_chart(fig2, use_container_width=True)

if service_T > temps_final["Af_C"]:
    st.success(
        "Screening interpretation: after the estimated composition shift, "
        "service temperature is still above Af. This supports possible "
        "austenitic/superelastic service, provided defects and hysteresis are acceptable."
    )
elif service_T < temps_final["Ms_C"]:
    st.error(
        "Screening interpretation: after the estimated composition shift, "
        "service temperature is below Ms. Martensite-at-service risk is high, "
        "so stable superelasticity is unlikely without process or heat-treatment correction."
    )
else:
    st.warning(
        "Screening interpretation: service temperature lies inside or near the "
        "transformation interval. Expect mixed/unstable phase response unless DSC confirms otherwise."
    )

st.divider()

st.subheader("Export current calculation")

export = pd.DataFrame([out])
st.download_button(
    label="Download current vaporization calculation as CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="niti_vaporization_composition_result.csv",
    mime="text/csv",
)

with st.expander("Scientific limits and how to improve this model later"):
    st.markdown(
        """
This page is intentionally transparent and conservative.

**What is already useful**
- It separates P, v, hatch spacing and layer thickness instead of relying only on VED.
- It makes Ni/Ti mass balance visible.
- It adds remelting and layer-thickness sensitivity.
- It connects composition shift directly to Af/Ms risk.

**What must be added for publication-level prediction**
- ICP/EDS measured powder and printed-part Ni/Ti data.
- DSC-measured Ms, Mf, As and Af.
- Measured melt-pool dimensions or calibrated thermal simulation.
- Better activity data from CALPHAD/JMatPro/pycalphad.
- External validation against multiple LPBF NiTi papers.
- Uncertainty bands around final Ni at.% and transformation temperatures.

**Safe wording**
Use this as a physics-informed screening and hypothesis-generation tool, not as a
certified predictor of final LPBF NiTi performance.
"""
    )
