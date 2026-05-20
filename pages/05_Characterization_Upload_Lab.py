import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from src.image_analysis import load_image, porosity_screen, image_texture_descriptors
from src.xrd_tools import read_xrd_csv, find_peaks_simple, phase_hints
from src.ebsd_tools import read_ebsd_csv, ebsd_summary, grid_for_plot

st.title("Characterization Upload Lab")
st.caption("Upload SEM/optical images, XRD patterns, or EBSD/TKD-style orientation CSV files.")

st.markdown("""
This page is not meant to replace full metallography software. It gives a first-pass, reproducible report that connects uploaded evidence to the process–function question:

> Are the observed defects, phases and orientations consistent with a usable NiTi functional state?
""")

tab_img, tab_xrd, tab_ebsd = st.tabs(["SEM / optical microstructure", "XRD pattern", "EBSD / TKD orientation CSV"])

with tab_img:
    st.subheader("Microstructure feature screening")
    st.markdown("""
Use this for polished SEM/optical images where dark features may represent pores, cracks or pull-outs.  
For final reporting, calibrate pixel size and manually verify segmentation because scratches, etching contrast and polishing artifacts can be misread as pores.
""")
    demo = st.checkbox("Use demo microstructure image", value=True)
    file = None if demo else st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    img = Image.open("assets/demo_microstructure.png") if demo else (load_image(file) if file else None)
    if img is not None:
        c1, c2 = st.columns(2)
        c1.image(img, caption="Input", use_container_width=True)
        invert = st.checkbox("Dark features are pores/cracks", value=True)
        th = st.slider("Manual threshold; 0 = Otsu", 0, 255, 0)
        pixel = st.number_input("Pixel size (µm/pixel; optional)", value=0.0)
        result = porosity_screen(img, invert=invert, manual_threshold=None if th == 0 else th, pixel_size_um=pixel if pixel > 0 else None)
        c2.image(result["mask"], caption="Segmentation mask", use_container_width=True)
        st.subheader("Image-derived descriptors")
        st.json({k: v for k, v in result.items() if k != "mask"})
        st.json(image_texture_descriptors(img))

        area = result["feature_area_fraction_pct"]
        if area < 0.5:
            st.success("Screening interpretation: very low segmented feature fraction. Still verify that contrast truly represents pores/cracks.")
        elif area < 2.0:
            st.info("Screening interpretation: moderate segmented feature fraction. This may be acceptable or unacceptable depending on whether features are pores, cracks or harmless contrast.")
        else:
            st.warning("Screening interpretation: high segmented feature fraction. Check for lack-of-fusion, keyhole pores, cracking or polishing artifacts.")

with tab_xrd:
    st.subheader("XRD screening")
    st.markdown("""
Use a two-column CSV with `two_theta_deg` and `intensity`.  
The tool detects peaks and gives **screening hints** only. B2/B19′ peak overlap is common, so final phase claims need calibration and refinement.
""")
    demo = st.checkbox("Use demo XRD CSV", value=True)
    file = None if demo else st.file_uploader("Upload XRD CSV", type=["csv"])
    if demo or file:
        df = read_xrd_csv("data/demo_xrd.csv" if demo else file)
        fig = px.line(df, x="two_theta_deg", y="intensity", title="XRD pattern")
        peaks = find_peaks_simple(df)
        if len(peaks):
            fig.add_scatter(x=peaks["two_theta_deg"], y=peaks["intensity"], mode="markers", name="peaks")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Detected peaks")
        st.dataframe(peaks, use_container_width=True)
        st.subheader("Phase hints")
        st.dataframe(phase_hints(peaks), use_container_width=True)
        st.markdown("""
**Interpretation path**

- Dominant B2-like peaks near service temperature support superelastic design, but DSC is still required.
- Strong B19′/martensite indicators near room temperature may explain weak superelasticity.
- Secondary-phase peaks can indicate Ti-rich/Ni-rich precipitation, oxidation or local composition shifts.
""")

with tab_ebsd:
    st.subheader("EBSD/TKD map inspection")
    st.markdown("""
Upload a CSV export containing at least `x`, `y`, Euler angles and optionally phase, CI/IQ and grain ID.  
This starter version maps phase/orientation fields and reports spread statistics. A full future version should add exact B2 parent reconstruction and B19′ variant indexing.
""")
    demo = st.checkbox("Use demo EBSD CSV", value=True)
    file = None if demo else st.file_uploader("Upload EBSD/TKD CSV", type=["csv"])
    if demo or file:
        df = read_ebsd_csv("data/demo_ebsd.csv" if demo else file)
        st.dataframe(df.head(), use_container_width=True)
        st.subheader("Map summary")
        summary = ebsd_summary(df)
        st.json(summary)
        columns = [c for c in ["phase", "ci", "iq", "phi1", "Phi", "phi2", "grain_id"] if c in df.columns]
        value = st.selectbox("Map value", columns)
        pivot = grid_for_plot(df, value)
        if pivot is not None:
            st.plotly_chart(px.imshow(pivot, aspect="equal", title=f"Map: {value}"), use_container_width=True)

        if "phase_counts" in summary:
            st.info("Phase maps can be compared with XRD/DSC to decide whether the printed state is mainly B2/austenite, B19′/martensite, or mixed.")
        st.markdown("""
**Future high-level extension**

- compute misorientation histograms;
- reconstruct prior B2 parent grains;
- label B19′ variants;
- quantify variant-pair statistics;
- link variant selection to build direction, residual stress or post-processing.
""")
