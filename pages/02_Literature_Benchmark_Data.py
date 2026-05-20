import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_exact_process_rows, load_paper_facts
from src.scenario_engine import load_scenarios

st.title("Literature Benchmark Data")
st.caption("Exact extracted rows, paper-level numerical facts and scenario rules. Demo rows are not shown here.")

rows = load_exact_process_rows()
facts = load_paper_facts()
scenarios = load_scenarios()

tab1, tab2, tab3, tab4 = st.tabs(["Exact process rows", "Paper-level facts", "Scenario rules", "Coverage gaps"])

with tab1:
    st.markdown("""
The current exact row-level dataset is strongest for **process-window classification**, because it contains A1–D4 process conditions and functional labels from the Ni-rich LPBF study.
""")
    st.dataframe(rows, use_container_width=True, height=420)
    fig = px.scatter(
        rows, x="scan_speed_mm_s", y="laser_power_W",
        size="VED_J_mm3", color="functional_class",
        hover_data=["sample_id", "VED_J_mm3", "functional_observation", "microstructure_observation"]
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download exact process rows", rows.to_csv(index=False), "exact_literature_process_rows.csv")

with tab2:
    st.markdown("These are exact numerical facts/ranges/constants used to support the app logic.")
    st.dataframe(facts, use_container_width=True, height=520)
    st.download_button("Download paper-level facts", facts.to_csv(index=False), "paper_level_numeric_facts.csv")

with tab3:
    st.markdown("""
Scenario rules translate literature observations into transparent app logic.  
They are not hidden ML; they are readable metallurgical assumptions that can be edited as the dataset grows.
""")
    st.dataframe(scenarios, use_container_width=True, height=520)
    st.download_button("Download scenario rules", scenarios.to_csv(index=False), "scenario_rules.csv")

with tab4:
    st.markdown("""
### Current state

The repository now has:

- 16 exact LPBF process rows from one systematic Ni-rich study;
- 36 paper-level facts from multiple studies;
- scenario rules covering low-energy failure, functional superelastic windows, high-energy Ni-loss risk, defect-free optimization and ML benchmarking.

### What still needs extraction for stronger regression

- row-level density/porosity for each process condition;
- row-level Ms/Mf/As/Af values, preferably from tables or digitized DSC figures;
- row-level UTS, elongation, residual strain, recovery strain and cyclic data;
- oxygen level and measured Ni/Ti composition for every condition;
- heat-treatment rows, not only as-printed rows;
- EBSD/TKD orientation maps for variant-level statistics.

### Why the app is designed this way

A thin but exact database is better than a large fake one.  
The app therefore supports real classification now and regression later as more row-level paper data are added.
""")
