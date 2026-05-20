import streamlit as st
from pathlib import Path

st.title("Model Card and Limits")
st.caption("Academic guardrails for using the app.")

st.markdown("""
## Intended use

Early-stage research support, literature benchmarking, process-window reasoning and experiment planning for LPBF/PBF-LB/M NiTi.

## Not intended use

- Medical-device qualification.
- Final process certification.
- Replacement of DSC, XRD, EBSD/TKD, tensile or fatigue testing.
- Claims based on synthetic/demo data.

## Best current model strategy

1. Start with exact literature rows only.
2. Use Gaussian Process regression for small data with uncertainty.
3. Use Random Forest / Extra Trees for nonlinear process maps once the database is large.
4. Report external validation, not only cross-validation.
5. Keep separate models for:
   - transformation temperatures;
   - density/porosity;
   - mechanical properties;
   - functional class.
6. Add crystallographic descriptors when EBSD/TKD data become available.

## Why energy density alone is not enough

Recent large-sample work shows transformation temperatures depend nonlinearly on both laser power and scanning speed, and energy density alone is an inadequate predictor. Therefore the model stores P, v, h and t separately as well as VED.
""")

st.markdown(Path("docs/LITERATURE_DATA_AUDIT.md").read_text(encoding="utf-8"))
