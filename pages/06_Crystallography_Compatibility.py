import streamlit as st
import pandas as pd
import plotly.express as px
from src.constants import NITI_DEFAULTS
from src.crystallography import lattice_misfit_summary, generate_simplified_b19_variants, variant_pair_table, stress_variant_scores

st.title("Crystallography and Compatibility")
st.caption("Editable B2/B19′ lattice metrics, variant descriptors and stress-assisted variant selection screening.")

c1,c2,c3,c4,c5 = st.columns(5)
B2a = c1.number_input("B2 a (Å)", value=NITI_DEFAULTS["B2_a_A"], step=0.001, format="%.4f")
a = c2.number_input("B19′ a (Å)", value=NITI_DEFAULTS["B19p_a_A"], step=0.001, format="%.4f")
b = c3.number_input("B19′ b (Å)", value=NITI_DEFAULTS["B19p_b_A"], step=0.001, format="%.4f")
c = c4.number_input("B19′ c (Å)", value=NITI_DEFAULTS["B19p_c_A"], step=0.001, format="%.4f")
beta = c5.number_input("β (deg)", value=NITI_DEFAULTS["B19p_beta_deg"], step=0.01, format="%.3f")

summary = lattice_misfit_summary(B2a, a, b, c, beta)
m1,m2,m3,m4 = st.columns(4)
m1.metric("B2 volume", f"{summary['B2_volume_A3']:.3f} Å³")
m2.metric("B19′ volume", f"{summary['B19p_volume_A3']:.3f} Å³")
m3.metric("Volume change", f"{summary['volume_change_pct']:.2f}%")
m4.metric("λ2 proxy", f"{summary['lambda2_proxy']:.4f}")

st.subheader("Metric strain")
df = pd.DataFrame({"axis":["λ1","λ2","λ3"], "strain":summary["principal_metric_strain"]})
st.plotly_chart(px.bar(df, x="axis", y="strain"), use_container_width=True)

st.subheader("Reference orientation relationship")
st.code(f"{NITI_DEFAULTS['natural_OR_planes']}\n{NITI_DEFAULTS['natural_OR_directions']}\nCandidate habit plane: {NITI_DEFAULTS['candidate_habit_plane']}")

st.subheader("Variant-pair screening")
variants = generate_simplified_b19_variants()
pair_df = pd.DataFrame(variant_pair_table(variants))
axis_choice = st.selectbox("Stress/build axis", ["Z/build", "X", "Y"])
axis = {"Z/build":(0,0,1), "X":(1,0,0), "Y":(0,1,0)}[axis_choice]
stress_df = pd.DataFrame(stress_variant_scores(variants, axis))
cL, cR = st.columns(2)
cL.dataframe(stress_df, use_container_width=True)
cR.dataframe(pair_df.sort_values("compatibility_proxy", ascending=False).head(15), use_container_width=True)
fig = px.scatter(pair_df, x="misorientation_deg", y="compatibility_proxy", hover_data=["variant_i","variant_j"])
st.plotly_chart(fig, use_container_width=True)

st.warning("This module is a transparent screening layer. Full publication-grade variant analysis requires exact correspondence operators and measured EBSD/TKD data.")
