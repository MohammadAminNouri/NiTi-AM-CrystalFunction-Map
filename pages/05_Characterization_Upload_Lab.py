import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from src.image_analysis import load_image, porosity_screen, image_texture_descriptors
from src.xrd_tools import read_xrd_csv, find_peaks_simple, phase_hints
from src.ebsd_tools import read_ebsd_csv, ebsd_summary, grid_for_plot

st.title("Characterization Upload Lab")
st.caption("Upload SEM/optical images, XRD patterns, or EBSD/TKD-style orientation CSV files.")

tab_img, tab_xrd, tab_ebsd = st.tabs(["SEM / optical microstructure", "XRD pattern", "EBSD / TKD orientation CSV"])

with tab_img:
    st.subheader("Microstructure feature screening")
    demo = st.checkbox("Use demo microstructure image", value=True)
    file = None if demo else st.file_uploader("Upload image", type=["png","jpg","jpeg","tif","tiff"])
    img = Image.open("assets/demo_microstructure.png") if demo else (load_image(file) if file else None)
    if img is not None:
        c1,c2 = st.columns(2)
        c1.image(img, caption="Input", use_container_width=True)
        invert = st.checkbox("Dark features are pores/cracks", value=True)
        th = st.slider("Manual threshold; 0 = Otsu", 0, 255, 0)
        pixel = st.number_input("Pixel size (µm/pixel; optional)", value=0.0)
        result = porosity_screen(img, invert=invert, manual_threshold=None if th==0 else th, pixel_size_um=pixel if pixel>0 else None)
        c2.image(result["mask"], caption="Segmentation mask", use_container_width=True)
        st.json({k:v for k,v in result.items() if k != "mask"})
        st.json(image_texture_descriptors(img))
        st.warning("Segmentation must be validated manually before publication.")

with tab_xrd:
    st.subheader("XRD screening")
    demo = st.checkbox("Use demo XRD CSV", value=True)
    file = None if demo else st.file_uploader("Upload XRD CSV", type=["csv"])
    if demo or file:
        df = read_xrd_csv("data/demo_xrd.csv" if demo else file)
        fig = px.line(df, x="two_theta_deg", y="intensity")
        peaks = find_peaks_simple(df)
        if len(peaks):
            fig.add_scatter(x=peaks["two_theta_deg"], y=peaks["intensity"], mode="markers", name="peaks")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(peaks, use_container_width=True)
        st.dataframe(phase_hints(peaks), use_container_width=True)
        st.warning("This is screening only; use calibrated phase refinement for final phase claims.")

with tab_ebsd:
    st.subheader("EBSD/TKD map inspection")
    demo = st.checkbox("Use demo EBSD CSV", value=True)
    file = None if demo else st.file_uploader("Upload EBSD/TKD CSV", type=["csv"])
    if demo or file:
        df = read_ebsd_csv("data/demo_ebsd.csv" if demo else file)
        st.dataframe(df.head(), use_container_width=True)
        st.json(ebsd_summary(df))
        columns = [c for c in ["phase","ci","iq","phi1","Phi","phi2","grain_id"] if c in df.columns]
        value = st.selectbox("Map value", columns)
        pivot = grid_for_plot(df, value)
        if pivot is not None:
            st.plotly_chart(px.imshow(pivot, aspect="equal", title=f"Map: {value}"), use_container_width=True)
        st.info("Future extension: exact parent-B2 reconstruction and B19′ variant indexing.")
