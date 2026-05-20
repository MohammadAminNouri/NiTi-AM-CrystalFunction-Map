import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(
    page_title="NiTi-AM CrystalFunction Map",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NiTi-AM CrystalFunction Map")
st.caption(
    "Benchmark-oriented process–crystallography–function platform for additively manufactured NiTi"
)

st.markdown(
    "This app is a research screening platform for LPBF / PBF-LB/M NiTi. "
    "It connects process parameters, Ni-loss risk, defect risk, B2↔B19′ transformation windows, "
    "literature benchmark logic, and functional superelastic or shape-memory suitability."
)

st.warning(
    "This is a screening and benchmark tool. It is not a substitute for DSC, XRD, EBSD/TKD, "
    "density measurement, cyclic mechanical testing, or process qualification."
)


def safe_read_csv(path):
    file_path = Path(path)
    if file_path.exists():
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def volumetric_energy_density(power_w, speed_mm_s, hatch_mm, layer_mm):
    if speed_mm_s <= 0 or hatch_mm <= 0 or layer_mm <= 0:
        return 0.0
    return power_w / (speed_mm_s * hatch_mm * layer_mm)


def defect_risk_from_ved(ved):
    import math

    lack_of_fusion = 1.0 / (1.0 + math.exp((ved - 55.0) / 8.0))
    keyhole = 1.0 / (1.0 + math.exp(-(ved - 133.3) / 12.0))
    cracking = 1.0 / (1.0 + math.exp((ved - 60.0) / 7.0))

    return {
        "lack_of_fusion": lack_of_fusion,
        "keyhole": keyhole,
        "cracking": cracking,
        "max_risk": max(lack_of_fusion, keyhole, cracking),
    }


def ni_loss_risk_from_ved(ved, oxygen_ppm, remelt_passes):
    import math

    x = (ved - 116.7) / 35.0
    x += 0.35 * remelt_passes
    x += max(0.0, oxygen_ppm - 70.0) / 400.0

    return 1.0 / (1.0 + math.exp(-x))


def estimate_effective_ni(powder_ni, measured_ni, ni_loss_risk):
    if measured_ni > 0:
        return measured_ni
    return powder_ni - 0.75 * ni_loss_risk


def transformation_rule(effective_ni, heat_t, heat_min):
    import math

    d_ni_tenths = (effective_ni - 50.0) / 0.1

    if heat_min <= 0:
        heat_factor = 0.0
    else:
        heat_factor = ((heat_t - 450.0) / 200.0) * math.log1p(heat_min) / math.log(121.0)

    ms = 45.0 - 10.0 * d_ni_tenths - 6.0 * heat_factor
    mf = ms - 12.0
    a_s = ms + 18.0
    af = a_s + 35.0

    return {
        "Mf_C": mf,
        "Ms_C": ms,
        "As_C": a_s,
        "Af_C": af,
        "hysteresis_K": af - ms,
    }


def process_scenario(ved):
    if ved < 55.0:
        return {
            "title": "Low-energy discontinuity / cracking risk",
            "logic": "The melt pool is likely too small or unstable for reliable interlayer bonding. Lack of fusion, cracks, or discontinuous tracks can dominate before functional NiTi behavior is meaningful.",
            "example": "Comparable to the low-energy cracked condition reported for Ni-rich LPBF NiTi at about 53.3 J/mm³.",
        }

    if 66.7 <= ved <= 116.7:
        return {
            "title": "Functional low-energy superelastic window",
            "logic": "Energy is high enough for bonding but not so high that Ni evaporation strongly shifts the transformation window. This is the most promising screening region for room-temperature superelasticity.",
            "example": "Comparable to reported Ni-rich LPBF NiTi conditions with strong cyclic recovery between about 66.7 and 116.7 J/mm³.",
        }

    if 133.3 <= ved <= 155.6:
        return {
            "title": "Transition zone: keyhole / partial superelasticity",
            "logic": "Higher energy can improve melting but may introduce keyhole defects, local Ni loss, and unstable transformation response.",
            "example": "Comparable to partial-superelasticity cases where recovery decreases and keyhole features may appear.",
        }

    if ved >= 166.7:
        return {
            "title": "High-energy Ni-loss / martensite-at-service risk",
            "logic": "High energy increases Ni evaporation. In NiTi, small Ni loss can strongly raise transformation temperatures and push the material toward martensitic behavior at service temperature.",
            "example": "Comparable to high-energy Ni-rich LPBF NiTi cases with poor or nearly absent superelasticity.",
        }

    return {
        "title": "Intermediate unvalidated process window",
        "logic": "This condition is between extracted benchmark windows. Treat the prediction as interpolation and verify with density, composition, DSC, XRD, and cyclic testing.",
        "example": "No exact benchmark row is assigned to this interval.",
    }


st.divider()

st.header("Process–Function Screening")

c1, c2, c3, c4 = st.columns(4)

power = c1.number_input("Laser power P (W)", min_value=1.0, value=100.0, step=5.0)
speed = c2.number_input("Scan speed v (mm/s)", min_value=1.0, value=800.0, step=50.0)
hatch = c3.number_input("Hatch spacing h (mm)", min_value=0.001, value=0.05, step=0.005, format="%.4f")
layer = c4.number_input("Layer thickness t (mm)", min_value=0.001, value=0.03, step=0.005, format="%.4f")

c5, c6, c7, c8 = st.columns(4)

powder_ni = c5.number_input("Powder Ni (at.%)", min_value=45.0, max_value=55.0, value=51.3, step=0.01)
measured_ni = c5.number_input("Measured Ni at.% (0 = ignore)", min_value=0.0, max_value=55.0, value=0.0, step=0.01)
oxygen = c6.number_input("Oxygen level (ppm)", min_value=0.0, value=70.0, step=10.0)
remelt = c6.number_input("Remelt / rescan passes", min_value=0, value=0, step=1)
heat_t = c7.number_input("Heat treatment temperature (°C)", min_value=0.0, value=500.0, step=10.0)
heat_min = c7.number_input("Heat treatment time (min)", min_value=0.0, value=30.0, step=10.0)
service_t = c8.number_input("Service temperature (°C)", value=37.0, step=1.0)
target = c8.selectbox("Functional target", ["superelastic", "thermal actuation"])

ved = volumetric_energy_density(power, speed, hatch, layer)
linear_ed = power / speed
risks = defect_risk_from_ved(ved)
ni_loss_risk = ni_loss_risk_from_ved(ved, oxygen, remelt)
effective_ni = estimate_effective_ni(powder_ni, measured_ni, ni_loss_risk)
temps = transformation_rule(effective_ni, heat_t, heat_min)
scenario = process_scenario(ved)

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("VED", f"{ved:.1f} J/mm³")
m2.metric("Linear energy", f"{linear_ed:.3f} J/mm")
m3.metric("Defect risk", f"{risks['max_risk']:.2f}")
m4.metric("Ni-loss risk", f"{ni_loss_risk:.2f}")
m5.metric("Effective Ni", f"{effective_ni:.2f} at.%")
m6.metric("Estimated Af", f"{temps['Af_C']:.1f} °C")

st.subheader("Metallurgical scenario")
st.markdown("### " + scenario["title"])
st.markdown("**Logic:** " + scenario["logic"])
st.markdown("**Example:** " + scenario["example"])

st.subheader("Transformation window")

mf = temps["Mf_C"]
ms = temps["Ms_C"]
a_s = temps["As_C"]
af = temps["Af_C"]

xmin = min(mf, ms, a_s, af, service_t) - 35
xmax = max(mf, ms, a_s, af, service_t) + 35

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[mf, ms],
        y=[1, 1],
        mode="lines+markers",
        line=dict(width=18),
        marker=dict(size=10),
        name="Cooling: Mf → Ms",
    )
)

fig.add_trace(
    go.Scatter(
        x=[a_s, af],
        y=[2, 2],
        mode="lines+markers",
        line=dict(width=18),
        marker=dict(size=10),
        name="Heating: As → Af",
    )
)

fig.add_vline(x=service_t, line_dash="dot", line_width=3, annotation_text="service")

for label, value in [("Mf", mf), ("Ms", ms), ("As", a_s), ("Af", af)]:
    fig.add_vline(x=value, line_dash="dash", opacity=0.35, annotation_text=label)

fig.update_layout(
    height=340,
    xaxis_title="Temperature (°C)",
    xaxis=dict(range=[xmin, xmax]),
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2],
        ticktext=["Cooling: martensite formation", "Heating: austenite recovery"],
        range=[0.4, 2.6],
    ),
)

st.plotly_chart(fig, use_container_width=True)

if target == "superelastic":
    if service_t > af:
        st.success("Service temperature is above Af. Austenitic superelastic response is possible if defects and hysteresis are controlled.")
    elif service_t < ms:
        st.warning("Service temperature is below Ms. The material may be martensitic at service, so superelasticity is unlikely.")
    else:
        st.info("Service temperature lies inside the transformation interval. Mixed or unstable response is possible.")
else:
    st.info("For actuation, Af should be close to the intended activation temperature and hysteresis should be acceptable.")

st.subheader("Risk map")

powers = list(range(60, 421, 20))
speeds = list(range(250, 2001, 75))
z = []

for p in powers:
    row = []
    for v in speeds:
        local_ved = volumetric_energy_density(p, v, hatch, layer)
        local_risk = defect_risk_from_ved(local_ved)["max_risk"]
        local_ni_loss = ni_loss_risk_from_ved(local_ved, oxygen, remelt)
        row.append(max(local_risk, local_ni_loss))
    z.append(row)

risk_fig = go.Figure(
    data=go.Heatmap(
        x=speeds,
        y=powers,
        z=z,
        colorbar=dict(title="max risk"),
    )
)

risk_fig.add_trace(
    go.Scatter(
        x=[speed],
        y=[power],
        mode="markers",
        marker=dict(size=16, symbol="x"),
        name="current",
    )
)

risk_fig.update_layout(
    height=500,
    xaxis_title="Scan speed (mm/s)",
    yaxis_title="Laser power (W)",
)

st.plotly_chart(risk_fig, use_container_width=True)

st.divider()

st.header("Benchmark data preview")

exact_rows = safe_read_csv("data/exact_literature_process_rows.csv")
facts = safe_read_csv("data/paper_level_numeric_facts.csv")

if len(exact_rows) > 0:
    st.subheader("Exact literature process rows")
    st.dataframe(exact_rows.head(20), use_container_width=True)

if len(facts) > 0:
    st.subheader("Paper-level numerical facts")
    st.dataframe(facts.head(30), use_container_width=True)

with st.expander("References and interpretation logic"):
    st.markdown(
        "- Xiang et al. 2024, Metals, Ni-rich LPBF NiTi process/superelasticity map, DOI: 10.3390/met14090961.\n"
        "- Xue et al. 2022, Acta Materialia, defect-free LPBF NiTi with tensile superelasticity up to 6%, DOI: 10.1016/j.actamat.2022.117781.\n"
        "- Li et al. 2024, Virtual and Physical Prototyping, ML optimization using 195 entries from 23 publications, DOI: 10.1080/17452759.2024.2364221.\n"
        "- Energy density alone is not enough; power, speed, hatch spacing, layer thickness, chemistry, oxygen, and heat treatment should remain separate descriptors."
    )