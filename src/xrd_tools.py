import numpy as np
import pandas as pd

def read_xrd_csv(file_or_path):
    df = pd.read_csv(file_or_path)
    cols = {c.lower().strip(): c for c in df.columns}
    two_col = cols.get("two_theta") or cols.get("2theta") or cols.get("two_theta_deg") or df.columns[0]
    int_col = cols.get("intensity") or cols.get("counts") or df.columns[1]
    out = df[[two_col, int_col]].copy()
    out.columns = ["two_theta_deg", "intensity"]
    return out.dropna()

def bragg_d_spacing(two_theta_deg, wavelength_A=1.5406):
    theta = np.radians(two_theta_deg/2)
    return float(wavelength_A/(2*np.sin(theta)+1e-12))

def find_peaks_simple(df, min_rel_height=0.12):
    x = df["two_theta_deg"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    if len(y) < 5:
        return pd.DataFrame(columns=["two_theta_deg", "intensity", "d_A"])
    y_norm = (y-y.min())/(y.max()-y.min()+1e-12)
    peaks = []
    for i in range(1, len(y)-1):
        if y_norm[i] > min_rel_height and y_norm[i] > y_norm[i-1] and y_norm[i] > y_norm[i+1]:
            peaks.append({"two_theta_deg": x[i], "intensity": y[i], "d_A": bragg_d_spacing(x[i])})
    return pd.DataFrame(peaks).sort_values("intensity", ascending=False).head(20)

def phase_hints(peaks):
    hints = []
    for val in peaks["two_theta_deg"].to_numpy(float) if len(peaks) else []:
        if 41 <= val <= 43.5:
            hint = "B2/B19′ overlap candidate"
        elif 38 <= val <= 40.5:
            hint = "B19′ or secondary-phase region candidate"
        elif 44 <= val <= 46.5:
            hint = "Ni-rich/Ti-rich precipitate screening region"
        elif 61 <= val <= 64:
            hint = "higher-angle NiTi-related peak candidate"
        else:
            hint = "unassigned screening peak"
        hints.append((val, hint))
    return pd.DataFrame(hints, columns=["two_theta_deg", "screening_hint"])
