import streamlit as st
import pandas as pd

from src.constants import APP_TITLE, APP_SUBTITLE, NITI_DEFAULTS
from src.data_utils import load_exact_process_rows, load_paper_facts


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    "\n".join(
        [
            "This platform is designed as a benchmark-style research app for LPBF/PBF-LB/M NiTi.",
            "",
            "It connects LPBF process parameters, defect risk, Ni-loss risk, B2 ↔ B19′ transformation logic, literature benchmark data, ML-assisted prediction, and uploaded characterization results.",
            "",
            "The central question is not only whether NiTi can be printed densely, but whether it can still remain a useful **shape-memory or superelastic functional material** after processing.",
        ]
    )
)

try:
    exact_rows = load_exact_process_rows()
except Exception:
    exact_rows = pd.DataFrame()

try:
    facts = load_paper_facts()
except Exception:
    facts = pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Exact process rows", len(exact_rows))
c2.metric("Paper-level facts", len(facts))
c3.metric("B2 lattice a", f"{NITI_DEFAULTS.get('B2_a_A', 'NA')} Å")
c4.metric("B19′ β", f"{NITI_DEFAULTS.get('B19p_beta_deg', 'NA')}°")

st.divider()

st.subheader("What this app does")

st.markdown(
    "\n".join(
        [
            "### 1. Process–Function Map",
            "Input laser power, scan speed, hatch spacing, layer thickness, Ni content, oxygen level, remelting and heat treatment.",
            "The app estimates defect risk, Ni-loss risk, transformation window and functional scenario.",
            "",
            "### 2. Literature Benchmark Data",
            "Stores exact extracted rows separately from paper-level facts and demo rows.",
            "",
            "### 3. Machine-Learning Workbench",
            "Trains regression only when enough real numerical rows exist.",
            "When real row-level data are sparse, it uses functional classification instead of fake regression.",
            "",
            "### 4. Prediction Studio",
            "Lets a user input a new process and get a structured screening report.",
            "",
            "### 5. Characterization Upload Lab",
            "Supports SEM/optical images, XRD CSV files and EBSD/TKD-style orientation CSV files.",
            "",
            "### 6. Crystallography and Compatibility",
            "Explores B2/B19′ lattice metrics and simplified transformation descriptors.",
        ]
    )
)

st.warning(
    "This is a research screening and benchmark tool. It is not a substitute for DSC, XRD, EBSD/TKD, density/porosity measurement, cyclic mechanical testing, or process qualification."
)

with st.expander("Current crystallographic defaults"):
    st.json(NITI_DEFAULTS)

with st.expander("Data philosophy"):
    st.markdown(
        "\n".join(
            [
                "The app separates three data layers:",
                "",
                "- exact literature rows",
                "- paper-level numerical facts",
                "- demo/synthetic rows",
                "",
                "Only exact literature rows should be used for defensible scientific claims.",
                "Demo rows are included only to test the ML interface.",
            ]
        )
    )

if len(exact_rows) > 0:
    st.subheader("Preview of exact process rows")
    st.dataframe(exact_rows.head(10), use_container_width=True)

if len(facts) > 0:
    st.subheader("Preview of paper-level numerical facts")
    st.dataframe(facts.head(10), use_container_width=True)
