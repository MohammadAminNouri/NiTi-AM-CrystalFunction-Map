from PIL import Image
import numpy as np

def load_image(file):
    return Image.open(file).convert("L")

def otsu_threshold(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0,255))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0
    w_b = 0
    max_var = 0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f)**2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold

def connected_components(binary):
    binary = binary.astype(bool)
    H, W = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    areas = []
    for y in range(H):
        for x in range(W):
            if binary[y, x] and not seen[y, x]:
                stack = [(y, x)]
                seen[y, x] = True
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx] = True
                            stack.append((ny,nx))
                areas.append(area)
    return areas

def porosity_screen(img, invert=True, manual_threshold=None, pixel_size_um=None):
    gray = np.array(img.convert("L"))
    th = int(manual_threshold) if manual_threshold is not None else otsu_threshold(gray)
    pores = gray < th if invert else gray > th
    areas_px = connected_components(pores)
    area_fraction = pores.mean()
    out = {
        "threshold": th,
        "feature_area_fraction_pct": float(100*area_fraction),
        "feature_count": int(len(areas_px)),
        "feature_area_px_mean": float(np.mean(areas_px)) if areas_px else 0.0,
        "feature_area_px_p95": float(np.percentile(areas_px, 95)) if areas_px else 0.0,
        "mask": pores.astype(np.uint8) * 255,
    }
    if pixel_size_um and pixel_size_um > 0 and areas_px:
        scale = pixel_size_um**2
        out["feature_area_um2_mean"] = float(np.mean(areas_px) * scale)
        out["feature_area_um2_p95"] = float(np.percentile(areas_px, 95) * scale)
    return out

def image_texture_descriptors(img):
    gray = np.asarray(img.convert("L"), dtype=float)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx*gx + gy*gy)
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "mean_gradient": float(grad.mean()),
        "gradient_p95": float(np.percentile(grad, 95)),
    }
