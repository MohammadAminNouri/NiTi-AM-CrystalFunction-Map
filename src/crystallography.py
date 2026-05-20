import numpy as np
from math import cos, radians

def monoclinic_metric(a, b, c, beta_deg):
    beta = radians(beta_deg)
    return np.array([[a*a, 0, a*c*cos(beta)], [0, b*b, 0], [a*c*cos(beta), 0, c*c]], dtype=float)

def lattice_misfit_summary(B2_a, B19_a, B19_b, B19_c, beta):
    Gm = monoclinic_metric(B19_a, B19_b, B19_c, beta)
    eig = np.linalg.eigvalsh(Gm)
    eq_lengths = np.sqrt(eig)
    ref = B2_a * np.ones(3)
    strain = (eq_lengths - ref) / ref
    volume_B2 = B2_a**3
    volume_B19 = B19_a * B19_b * B19_c * np.sin(np.radians(beta))
    return {
        "B2_volume_A3": float(volume_B2),
        "B19p_volume_A3": float(volume_B19),
        "volume_change_pct": float(100 * (volume_B19 - volume_B2) / volume_B2),
        "principal_metric_lengths_A": [float(x) for x in eq_lengths],
        "principal_metric_strain": [float(x) for x in strain],
        "strain_anisotropy": float(np.max(strain) - np.min(strain)),
        "lambda2_proxy": float(sorted(eq_lengths / B2_a)[1])
    }

def rotation_matrix(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    a = radians(angle_deg)
    C, S = np.cos(a), np.sin(a)
    return np.array([
        [C + x*x*(1-C), x*y*(1-C)-z*S, x*z*(1-C)+y*S],
        [y*x*(1-C)+z*S, C+y*y*(1-C), y*z*(1-C)-x*S],
        [z*x*(1-C)-y*S, z*y*(1-C)+x*S, C+z*z*(1-C)]
    ])

def generate_simplified_b19_variants(n=12):
    axes = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[1,0,-1],[0,1,-1],[1,1,1],[1,-1,1],[-1,1,1]]
    angles = [0,12,-12,24,-24,36,-36,48,-48,60,-60,72]
    return [{"variant": f"V{i+1:02d}", "axis": axes[i], "angle_deg": angles[i], "matrix": rotation_matrix(axes[i], angles[i])} for i in range(n)]

def misorientation_angle(R1, R2):
    M = R1 @ R2.T
    val = np.clip((np.trace(M) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(val)))

def variant_pair_table(variants):
    rows = []
    for i, vi in enumerate(variants):
        for j, vj in enumerate(variants):
            if j <= i:
                continue
            angle = misorientation_angle(vi["matrix"], vj["matrix"])
            # Compatibility proxy: low-to-moderate misorientation with twin-like peaks is favorable.
            comp = np.exp(-min(abs(angle-60), abs(angle-90), abs(angle-45))/40)
            rows.append({"variant_i": vi["variant"], "variant_j": vj["variant"], "misorientation_deg": round(angle,3), "compatibility_proxy": round(float(comp),3)})
    return rows

def stress_variant_scores(variants, stress_axis=(0,0,1)):
    axis = np.asarray(stress_axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rows = []
    for v in variants:
        transformed = v["matrix"] @ np.array([1,0,0], dtype=float)
        transformed = transformed / (np.linalg.norm(transformed) + 1e-12)
        score = abs(np.dot(transformed, axis))
        rows.append({"variant": v["variant"], "stress_alignment_score": round(float(score),3), "variant_selection": "favored" if score>0.75 else "moderate" if score>0.45 else "unfavored"})
    return rows
