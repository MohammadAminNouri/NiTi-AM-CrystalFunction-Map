import streamlit as st
from pathlib import Path

st.title("Data Extraction Protocol")
st.caption("How to convert papers into defensible benchmark rows.")

st.markdown(Path("docs/DATA_EXTRACTION_PROTOCOL.md").read_text(encoding="utf-8"))

st.download_button(
    "Download empty extraction template",
    data=Path("data/literature_extraction_template.csv").read_text(encoding="utf-8"),
    file_name="literature_extraction_template.csv",
    mime="text/csv"
)
