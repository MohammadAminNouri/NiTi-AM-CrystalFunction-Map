import pandas as pd
import numpy as np

def read_ebsd_csv(file_or_path):
    return normalize_columns(pd.read_csv(file_or_path))

def normalize_columns(df):
    colmap = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["x", "x_um", "x(um)"]:
            colmap[c] = "x"
        elif lc in ["y", "y_um", "y(um)"]:
            colmap[c] = "y"
        elif lc in ["phi1", "euler1", "euler_1"]:
            colmap[c] = "phi1"
        elif lc in ["phi", "euler2", "euler2_mid"]:
            colmap[c] = "Phi"
        elif lc in ["phi2", "euler3", "euler_3"]:
            colmap[c] = "phi2"
        elif lc in ["phase", "phase_id", "phaseid"]:
            colmap[c] = "phase"
        elif lc in ["ci", "confidence", "confidence_index"]:
            colmap[c] = "ci"
        elif lc in ["iq", "image_quality"]:
            colmap[c] = "iq"
        elif lc in ["grain", "grain_id", "grainid"]:
            colmap[c] = "grain_id"
    return df.rename(columns=colmap)

def ebsd_summary(df):
    out = {"rows": int(len(df))}
    for c in ["x","y","phi1","Phi","phi2","ci","iq"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            out[f"{c}_min"] = float(vals.min())
            out[f"{c}_max"] = float(vals.max())
            out[f"{c}_mean"] = float(vals.mean())
    if "phase" in df.columns:
        out["phase_counts"] = df["phase"].astype(str).value_counts().to_dict()
    if "grain_id" in df.columns:
        out["grain_count"] = int(df["grain_id"].nunique())
    if {"phi1","Phi","phi2"}.issubset(df.columns):
        e = df[["phi1","Phi","phi2"]].apply(pd.to_numeric, errors="coerce")
        out["orientation_spread_proxy_deg"] = float(e.std().mean())
    return out

def grid_for_plot(df, value_col):
    if not {"x","y",value_col}.issubset(df.columns):
        return None
    d = df[["x","y",value_col]].copy()
    for c in ["x","y",value_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    if len(d) == 0:
        return None
    return d.pivot_table(index="y", columns="x", values=value_col, aggfunc="mean")
