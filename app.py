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
    """
This platform is designed as a benchmark-style research app for LPBF/PBF-LB/M NiTi.

It connects:

```text
LPBF parameters
→ defect / Ni-loss risk
→ B2 ↔ B19′ transformation window
→ crystallographic and functional screening
→ literature benchmark comparison
→ ML-assisted prediction when enough real data exist
→ SEM / XRD / EBSD upload analysis
