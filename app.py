import streamlit as st
import pandas as pd
from src.constants import APP_TITLE, APP_SUBTITLE, NITI_DEFAULTS
from src.data_utils import load_exact_process_rows, load_paper_facts

st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown("""
This platform is designed as a defensible, extensible benchmark for LPBF/PBF-LB/M NiTi.  
It links processing, composition, crystallography, transformation temperatures, hysteresis and functional suitability.
""")

facts = load_paper_facts()
rows = load_exact_process_rows()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Exact process rows", len(rows))
c2.metric("Paper-level facts", len(facts))
c3.metric("B2 a", f"{NITI_DEFAULTS['B2_a_A']} Å")
c4.metric("B19′ β", f"{NITI_DEFAULTS['B19p_beta_deg']}°")

st.divider()

st.subheader("What the app can do")
st.markdown("""
- Build an LPBF process-function map from user inputs.
- Compare a user condition against real benchmark cases.
- Train ML models only when enough non-demo data exist.
- Analyze user-uploaded SEM/optical images, XRD peak files and EBSD/TKD-style orientation CSVs.
- Explore B2→B19′ metric distortion, simplified martensite variants and compatibility proxies.
""")

st.warning(
    "The app distinguishes exact literature facts from demo rows. Do not report ML metrics from synthetic/demo rows as scientific evidence."
)

with st.expander("Current exact crystallographic defaults"):
    st.json(NITI_DEFAULTS)

with st.expander("Data layers"):
    st.markdown("""
1. **Exact literature process rows**: process parameters and observations extracted from accessible tables/text.  
2. **Paper-level numerical facts**: exact ranges, constants and targets from literature.  
3. **Demo training seed**: synthetic rows only for interface testing.
""")
