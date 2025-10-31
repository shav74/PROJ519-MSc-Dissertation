import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# A1 — Toy comparative RCS simulator (with Size Sweep + Aspect Jitter)
# Shapes: sphere, cylinder, plate, faceted shell
# Outputs:
#   - Heatmaps per shape (RCS vs freq & aspect)
#   - Line plots at 0° and 45° (exact angle)
#   - Jittered line plots at 0°±5° and 45°±5°
#   - Size-sweep comparisons (0.8x / 1.0x / 1.2x) at 0°
#   - Snapshot CSVs (5 GHz @ 0° exact and jittered) + size-sweep deltas
# NOTE: Relative trends only (NOT certified absolute RCS).
# ---------------------------

def db_to_linear(db):
    return 10.0 ** (db / 10.0)

def linear_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))

# --- Toy models (trend-focused) ---

def rcs_sphere(radius_m, lam_array):
    # Approx constant vs angle (optical-ish region): ~ pi r^2
    return np.pi * radius_m**2 * np.ones_like(lam_array)

def rcs_cylinder(radius_m, length_m, lam_array, aspect_deg):
    # Broadside strongest, weaker as rotated; weak frequency dependence
    aspect = np.deg2rad(aspect_deg)
    base = (2 * np.pi * (length_m**2) / lam_array)  # simple trend
    ang = np.abs(np.cos(aspect)) ** 1.5             # angular dependence
    return base * ang

def rcs_plate(area_m2, lam_array, aspect_deg):
    # Specular-like response near boresight; decays with tilt
    aspect = np.deg2rad(aspect_deg)
    cos_term = np.cos(aspect)
    spec = (4 * np.pi * (area_m2**2) / (lam_array**2)) * (np.maximum(cos_term, 0.0) ** 2)
    return spec

def rcs_facet_bundle(area_m2, lam_array, aspect_deg):
    # Sum of 4 tilted plates (±30°, ±60°) as a simple faceted shell proxy
    offsets = np.array([-30.0, 30.0, -60.0, 60.0])
    total = 0.0
    for off in offsets:
        total += rcs_plate(area_m2 / 4.0, lam_array, aspect_deg - off)
    return total

def simulate_rcs(shapes_df, materials_df,
                 f_ghz=np.linspace(2, 10, 81),
                 aspect_deg=np.linspace(-90, 90, 181),
                 material_id="baseline_plastic"):
    """Return frequency, aspects, and dict of grids per shape (aspect x freq)."""
    c = 3e8
    lam = c / (f_ghz * 1e9)

    if material_id not in set(materials_df["material_id"]):
        raise ValueError(f"Unknown material_id: {material_id}")
    mat_loss_db = float(materials_df.loc[materials_df["material_id"] == material_id, "rcs_loss_db"].values[0])
    mat_lin = db_to_linear(mat_loss_db)

    results = {}
    for _, row in shapes_df.iterrows():
        sid = str(row["shape_id"])
        typ = str(row["type"]).lower()

        if typ == "sphere":
            sigma = rcs_sphere(row["radius_m"], lam)
            grid = np.tile(sigma, (len(aspect_deg), 1))

        elif typ == "cylinder":
            grid = np.zeros((len(aspect_deg), len(f_ghz)))
            for i, ad in enumerate(aspect_deg):
                grid[i, :] = rcs_cylinder(row["radius_m"], row["length_m"], lam, ad)

        elif typ == "plate":
            grid = np.zeros((len(aspect_deg), len(f_ghz)))
            for i, ad in enumerate(aspect_deg):
                grid[i, :] = rcs_plate(row["area_m2"], lam, ad)

        elif typ == "faceted":
            grid = np.zeros((len(aspect_deg), len(f_ghz)))
            for i, ad in enumerate(aspect_deg):
                grid[i, :] = rcs_facet_bundle(row["area_m2"], lam, ad)

        else:
            print(f"[warn] Unknown shape type '{typ}' for {sid} — skipping.")
            continue

        results[sid] = grid * mat_lin

    return f_ghz, aspect_deg, results

# --- Helpers: plotting ---

def plot_heatmap(f_ghz, aspect_deg, grid, title, outpath):
    plt.figure(figsize=(6, 4))
    plt.imshow(linear_to_db(grid),
               origin="lower", aspect="auto",
               extent=[f_ghz[0], f_ghz[-1], aspect_deg[0], aspect_deg[-1]])
    plt.colorbar(label="Relative RCS (dB)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Aspect (deg)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_line_compare(f_ghz, results_at_aspect, labels, outpath, aspect_label, title_prefix="RCS vs Frequency"):
    plt.figure(figsize=(6, 4))
    for arr, lab in zip(results_at_aspect, labels):
        plt.plot(f_ghz, linear_to_db(arr), label=lab)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Relative RCS (dB)")
    plt.title(f"{title_prefix} @ aspect {aspect_label}°")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# --- NEW: Aspect jitter averaging ---

def jittered_curve(grid, aspects_deg, center_deg, jitter_deg=5):
    """Average RCS over [center-jitter, center+jitter] across aspect rows."""
    mask = (aspects_deg >= center_deg - jitter_deg) & (aspects_deg <= center_deg + jitter_deg)
    if not np.any(mask):
        # Fallback to nearest aspect
        idx = int(np.argmin(np.abs(aspects_deg - center_deg)))
        return grid[idx, :]
    return np.mean(grid[mask, :], axis=0)

# --- NEW: Size sweep utilities ---

def scale_shape_row(row, scale):
    """Return a new dict with scaled dimensions.
       Linear scale s → area scales ~ s^2."""
    new = row.copy()
    new["shape_id"] = f"{row['shape_id']}_x{scale:.1f}".replace('.', 'p')
    typ = str(row["type"]).lower()
    if typ == "sphere":
        new["radius_m"] = float(row["radius_m"]) * scale
    elif typ == "cylinder":
        new["radius_m"] = float(row["radius_m"]) * scale
        new["length_m"] = float(row["length_m"]) * scale
    elif typ in ["plate", "faceted"]:
        new["area_m2"] = float(row["area_m2"]) * (scale ** 2)
    return new

def run_size_sweep_for_shape(row, materials_df, material_id, f_ghz, aspects,
                             scales=(0.8, 1.0, 1.2)):
    """Simulate one base shape at multiple scales and return dict scale->grid."""
    # Build a tiny shapes_df from scaled variants
    rows = [scale_shape_row(row, s) for s in scales]
    sdf = pd.DataFrame(rows)
    f, a, res = simulate_rcs(sdf, materials_df, f_ghz=f_ghz, aspect_deg=aspects, material_id=material_id)
    # Reorder by scales
    grids = {}
    for s, r in zip(scales, rows):
        grids[s] = res[r["shape_id"]]
    return f, a, grids

def save_size_sweep_plots(shape_id, f_ghz, aspects, grids_by_scale, outdir, compare_aspect=0):
    """Plot RCS vs freq at compare_aspect for 0.8x/1.0x/1.2x."""
    idx = int(np.argmin(np.abs(aspects - compare_aspect)))
    arrs, labels = [], []
    for s in sorted(grids_by_scale.keys()):
        arrs.append(grids_by_scale[s][idx, :])
        labels.append(f"{shape_id} @ {s:.1f}x")
    plot_line_compare(f_ghz, arrs, labels,
                      os.path.join(outdir, f"size_sweep_{shape_id}_{compare_aspect}deg.png"),
                      compare_aspect,
                      title_prefix="Size Sweep")

def size_sweep_summary(shape_id, f_ghz, aspects, grids_by_scale, f_snap=5.0, aspect_snap=0.0):
    """Return a DataFrame summarising RCS at f_snap, aspect_snap for each scale and Δ vs 1.0x."""
    # Find indices
    fi = int(np.argmin(np.abs(f_ghz - f_snap)))
    ai = int(np.argmin(np.abs(aspects - aspect_snap)))
    # Extract values
    rows = []
    baseline = None
    for s in sorted(grids_by_scale.keys()):
        val_lin = grids_by_scale[s][ai, fi]
        val_db = float(linear_to_db(val_lin))
        rows.append({"shape_id": shape_id, "scale": s, "RCS_rel_dB": val_db})
        if np.isclose(s, 1.0):
            baseline = val_db
    # Compute deltas vs 1.0x
    for r in rows:
        r["delta_vs_1p0x_dB"] = r["RCS_rel_dB"] - baseline
    return pd.DataFrame(rows)

# --- Main workflow ---

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    shapes_path = os.path.join(base_dir, "shapes.csv")
    mats_path = os.path.join(base_dir, "material_presets.csv")

    shapes = pd.read_csv(shapes_path)
    mats = pd.read_csv(mats_path)

    f_ghz = np.linspace(2, 10, 81)
    aspects = np.linspace(-90, 90, 181)

    # Run two scenarios: baseline and a light absorber
    material_cases = ["baseline_plastic", "rf_absorb_foam_light"]

    for material in material_cases:
        # ---------- Core runs ----------
        f, a, res = simulate_rcs(shapes, mats, f_ghz=f_ghz, aspect_deg=aspects, material_id=material)
        outdir = os.path.join(base_dir, f"outputs_{material}")
        os.makedirs(outdir, exist_ok=True)

        # Heatmaps per shape
        for sid, grid in res.items():
            plot_heatmap(f, a, grid, f"{sid} — {material}", os.path.join(outdir, f"heatmap_{sid}.png"))

        # Exact-angle comparisons at 0° and 45°
        for ang in [0, 45]:
            arrs, labels = [], []
            idx = int(np.argmin(np.abs(a - ang)))
            for sid, grid in res.items():
                arrs.append(grid[idx, :])
                labels.append(sid)
            plot_line_compare(f, arrs, labels, os.path.join(outdir, f"compare_{ang}deg.png"), ang)

        # ---------- NEW: Aspect jitter comparisons ----------
        for ang in [0, 45]:
            arrs_j, labels_j = [], []
            for sid, grid in res.items():
                arrs_j.append(jittered_curve(grid, a, center_deg=ang, jitter_deg=5))
                labels_j.append(sid)
            plot_line_compare(f, arrs_j, labels_j, os.path.join(outdir, f"compare_jitter_{ang}deg_pm5.png"),
                              ang, title_prefix="RCS vs Frequency (Jitter ±5°)")

        # Snapshot summary at 5 GHz & 0° (exact)
        f0 = 5.0
        f_s, a_s, res_s = simulate_rcs(shapes, mats, f_ghz=np.array([f0]), aspect_deg=np.array([0.0]), material_id=material)
        rows_exact = []
        for sid, grid in res_s.items():
            rows_exact.append({"material": material, "shape_id": sid, "RCS_rel_dB": float(linear_to_db(grid[0, 0]))})
        pd.DataFrame(rows_exact).to_csv(os.path.join(base_dir, f"summary_{material}_5GHz_0deg_exact.csv"), index=False)

        # Snapshot summary at 5 GHz & 0° (jittered ±5°)
        rows_jit = []
        for sid, grid in res.items():
            curve = jittered_curve(grid, a, center_deg=0.0, jitter_deg=5)   # full freq curve
            fi = int(np.argmin(np.abs(f - f0)))
            rows_jit.append({"material": material, "shape_id": sid, "RCS_rel_dB_jittered": float(linear_to_db(curve[fi]))})
        pd.DataFrame(rows_jit).to_csv(os.path.join(base_dir, f"summary_{material}_5GHz_0deg_jitter_pm5.csv"), index=False)

        # ---------- NEW: Size sweep for each base shape ----------
        outdir_sw = os.path.join(base_dir, f"outputs_{material}_sizesweep")
        os.makedirs(outdir_sw, exist_ok=True)
        sweep_summaries = []

        for _, row in shapes.iterrows():
            sid = str(row["shape_id"])
            f_sw, a_sw, grids = run_size_sweep_for_shape(row, mats, material, f_ghz, aspects, scales=(0.8, 1.0, 1.2))
            save_size_sweep_plots(sid, f_sw, a_sw, grids, outdir_sw, compare_aspect=0)
            df_sum = size_sweep_summary(sid, f_sw, a_sw, grids, f_snap=5.0, aspect_snap=0.0)
            df_sum["material"] = material
            sweep_summaries.append(df_sum)

        if sweep_summaries:
            pd.concat(sweep_summaries, ignore_index=True).to_csv(
                os.path.join(base_dir, f"summary_{material}_size_sweep_5GHz_0deg.csv"),
                index=False
            )

if __name__ == "__main__":
    main()
