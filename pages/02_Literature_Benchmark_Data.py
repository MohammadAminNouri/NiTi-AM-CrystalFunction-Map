import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_exact_process_rows, load_paper_facts

st.title("Literature Benchmark Data")
st.caption("Exact extracted rows and paper-level numerical facts. Demo rows are not shown here.")

rows = load_exact_process_rows()
facts = load_paper_facts()

tab1, tab2, tab3 = st.tabs(["Exact process rows", "Paper-level facts", "Coverage gaps"])

with tab1:
    st.dataframe(rows, use_container_width=True, height=420)
    fig = px.scatter(rows, x="scan_speed_mm_s", y="laser_power_W", size="VED_J_mm3", color="functional_class",
                     hover_data=["sample_id","VED_J_mm3","functional_observation"])
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download exact process rows", rows.to_csv(index=False), "exact_literature_process_rows.csv")

with tab2:
    st.dataframe(facts, use_container_width=True, height=520)
    st.download_button("Download paper-level facts", facts.to_csv(index=False), "paper_level_numeric_facts.csv")

with tab3:
    st.markdown("""
### Current state

The repository contains exact numbers from accessible tables/text and a structured path to scale the dataset.

### Still needed for a true community benchmark

- Full row-level extraction from the 195-entry ML dataset if obtainable.
- Figure-digitized DSC points for each sample in studies where values are plotted but not tabulated.
- Density/porosity values by sample where only ranges are described in text.
- Exact tensile, compressive and cyclic superelastic curves from supplementary data.
- EBSD/TKD maps or orientation exports for variant statistics.
""")
