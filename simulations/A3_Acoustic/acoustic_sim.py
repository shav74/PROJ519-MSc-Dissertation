import os, json, math, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import librosa
    import librosa.display
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

# -----------------------------
# Utility functions
# -----------------------------

def a_weighting_db(freq_hz):
    """Return A-weighting (dB) for array-like freq_hz (20 Hz..20 kHz). IEC 61672 approx."""
    f = np.maximum(np.asarray(freq_hz, dtype=float), 1e-6)
    ra_num = (12194**2) * (f**4)
    ra_den = (f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2)*(f**2 + 737.9**2)) * (f**2 + 12194**2)
    ra = ra_num / np.maximum(ra_den, 1e-30)
    a = 20*np.log10(ra) + 2.00  # 2.00 dB standard offset
    return a

def air_absorption_db_per_m(freq_hz, temp_C=20.0, rh_percent=50.0, scale=1.0):
    """
    Very simplified atmospheric absorption (dB/m).
    Roughly ~0.0001 dB/m @1 kHz, ~0.01 dB/m @10 kHz at 20C/50% RH.
    We use a smooth power law and a user 'scale' knob.
    """
    f = np.maximum(np.asarray(freq_hz, dtype=float), 1.0)
    base = 1e-4 * (f/1000.0)**1.2  # gentle rise with frequency
    return scale * base

def spl_sum_db(levels_db, axis=None):
    """Sum SPL in dB (power sum)."""
    levels_lin = 10**(np.asarray(levels_db)/10.0)
    s = np.sum(levels_lin, axis=axis)
    return 10*np.log10(np.maximum(s, 1e-30))

def db_to_lin(db):
    return 10**(db/10.0)

def lin_to_db(lin):
    return 10*np.log10(np.maximum(lin, 1e-30))

# -----------------------------
# Source / geometry models
# -----------------------------

def thrust_to_rpm(thrust_N, D_m, rho_air=1.2, k_T=1.0e-5):
    """
    Solve RPM for target thrust:
    T ≈ k_T * rho * D^4 * (RPM/60)^2  => RPM ≈ 60*sqrt(T / (k_T*rho*D^4))
    """
    denom = max(k_T * rho_air * (D_m**4), 1e-12)
    rpm = 60.0 * math.sqrt(thrust_N / denom)
    return rpm

def directivity_gain(theta_rad, model='sin', q=2.0):
    """
    Simple directivity: 'sin' peaks at 90° (loudest in plane),
    'cos' peaks along axis (0°).
    """
    if model == 'sin':
        g = np.sin(np.clip(theta_rad, 0.0, np.pi/2))**q
    else:
        g = np.cos(np.clip(theta_rad, 0.0, np.pi/2))**q
    return np.maximum(g, 1e-4)

def gaussian_line(freqs, f0, width_hz):
    """Unit-area Gaussian in linear power domain centered at f0 with std ~ width_hz/2.355."""
    sigma = max(width_hz/2.355, 1.0)
    return np.exp(-0.5*((freqs - f0)/sigma)**2) / (sigma*np.sqrt(2*np.pi))

def source_spectrum_one_rotor(freqs, rpm, D_m, blades, model_cfg, cfg_modifiers):
    """
    Build a plausible one-rotor source spectrum at 1 m, in linear power per Hz (relative).
    - Tonal lines at BPF and harmonics with decays.
    - Broadband floor shaped by RPM and modifiers (shroud/liner/isolation).
    """
    freqs = np.asarray(freqs, dtype=float)
    # Base scalings
    m = model_cfg.get('rpm_exponent', 4.5)     # tonal/broadband dependence on RPM
    n = model_cfg.get('diam_exponent', 1.5)    # dependence on diameter
    base_tone_db = model_cfg.get('tone_base_db', 50.0)
    base_bb_db   = model_cfg.get('bb_base_db', 30.0)
    harm_decay_db = model_cfg.get('harmonic_decay_db', 8.0)
    line_width_hz = model_cfg.get('line_width_hz', 30.0)
    bb_tilt_db_per_khz = model_cfg.get('bb_tilt_db_per_khz', -1.0) # broadband high-freq tilt

    # Modifiers (dB deltas applied later in appropriate bands)
    shroud_delta_db   = cfg_modifiers.get('shroud_bb_delta_db', -0.5)
    liner_delta_db    = cfg_modifiers.get('liner_bb_delta_db', -1.5)
    isolation_delta_db= cfg_modifiers.get('iso_lowband_delta_db', -1.0)

    # Base tonal level
    scale_tone_db = base_tone_db + m*np.log10(max(rpm,1)/1000.0) + n*np.log10(max(D_m,1e-3)/0.25)
    # Base broadband level
    scale_bb_db   = base_bb_db   + 0.8*m*np.log10(max(rpm,1)/1000.0) + 0.5*n*np.log10(max(D_m,1e-3)/0.25)

    # Tonal lines (BPF and a few harmonics)
    bpf = blades * rpm / 60.0
    tonal_lin = np.zeros_like(freqs, dtype=float)
    for h in range(1, 4):  # 1x..3x BPF
        amp_db = scale_tone_db - (h-1)*harm_decay_db
        tonal_lin += db_to_lin(amp_db) * gaussian_line(freqs, h*bpf, line_width_hz)

    # Broadband floor: flat-ish with slight downward tilt at HF
    bb_db = scale_bb_db + (bb_tilt_db_per_khz * (freqs/1000.0))
    bb_lin = db_to_lin(bb_db)

    # Apply modifiers (broadband): shroud/liner affect mid-high bands; isolation affects low-mid band
    # We'll implement simple piecewise gains.
    bb_gain = np.ones_like(freqs)
    if cfg_modifiers.get('has_shroud', False):
        bb_gain *= db_to_lin(shroud_delta_db)
    if cfg_modifiers.get('has_liner', False):
        # emphasize near BPF region (~0.7..3x BPF)
        band = (freqs >= 0.7*bpf) & (freqs <= 3.0*bpf)
        bb_gain[band] *= db_to_lin(liner_delta_db)
    if cfg_modifiers.get('has_isolation', False):
        # reduce low-mid band
        band = (freqs >= 50) & (freqs <= 1000)
        bb_gain[band] *= db_to_lin(isolation_delta_db)

    bb_lin *= bb_gain

    # Combine tonal + broadband (linear power/Hz)
    return tonal_lin + bb_lin, bpf

def rotor_to_mic_theta(drone_alt_m, horiz_dist_m):
    """
    Angle between rotor axis (vertical) and vector to mic.
    If mic is on ground and rotor axis is vertical, theta ~ arctan(horiz / alt).
    """
    return np.arctan2(max(horiz_dist_m, 0.0), max(drone_alt_m, 1e-6))

def propagate_and_weight(freqs, src_lin_power, distance_m, cfg_env, directivity=1.0):
    """
    Apply geometric spreading, air absorption, A-weighting; return A-weighted spectrum (linear).
    src_lin_power is linear spectral power at 1 m (arbitrary ref).
    """
    # Geometric spreading from 1 m
    spread_db = -20.0*np.log10(max(distance_m, 1e-3))
    # Air absorption (dB)
    alpha_db_per_m = air_absorption_db_per_m(freqs,
                                             cfg_env['temp_C'],
                                             cfg_env['rh_percent'],
                                             cfg_env['absorption_scale'])
    air_db = -alpha_db_per_m * distance_m
    # Apply propagation and directivity
    spec_db = lin_to_db(src_lin_power) + spread_db + air_db + lin_to_db(np.array(directivity))
    # A-weighting
    A_db = a_weighting_db(freqs)
    specA_db = spec_db + A_db
    return db_to_lin(specA_db)

def laeq_from_spectrum(freqs, specA_lin):
    """
    Approximate LAeq from A-weighted spectrum (linear power/Hz).
    Integrate over frequency.
    """
    # numeric integration over Hz
    df = np.diff(freqs).mean()
    total = np.sum(specA_lin) * max(df, 1.0)
    return lin_to_db(total)

def sum_four_rotors(specs_lin):
    """Sum four rotor spectra (linear) incoherently."""
    # specs_lin: list of arrays
    return np.sum(np.stack(specs_lin, axis=0), axis=0)

# -----------------------------
# IO helpers
# -----------------------------

def load_all(base_dir):
    with open(os.path.join(base_dir, "config.yml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    props = pd.read_csv(os.path.join(base_dir, "props.csv"))
    configs = pd.read_csv(os.path.join(base_dir, "configs.csv"))
    rpm_sets = pd.read_csv(os.path.join(base_dir, "rpm_sets.csv"))
    calib_path = os.path.join(base_dir, "calibration.json")
    cal = None
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
        except Exception:
            cal = None
    return cfg, props, configs, rpm_sets, cal

# -----------------------------
# Main simulation routines
# -----------------------------

def simulate_config_LAeq_curve(cfg, prop_row, conf_row, rpm_target, out_dir):
    """
    Produce LAeq vs distance curve, spectrogram at a reference distance, footprint map,
    and return summary dict for this (config_id, mode).
    """
    freqs = np.linspace(cfg['eval']['fmin_hz'], cfg['eval']['fmax_hz'], cfg['eval']['n_freq'])
    distances = np.array(cfg['eval']['distances_m'], dtype=float)

    # One-point calibration gain (dB) to align model with a real datum, if provided
    cal_gain_db = 0.0
    if cfg.get('calibration', {}).get('enabled', False) and cfg.get('calibration', {}).get('cal_point', None):
        cal = cfg['calibration']['cal_point']
        if cal['rpm'] > 0 and cal['distance_m'] > 0:
            # Build a quick source at cal RPM for this prop and compute predicted LAeq at cal distance.
            src_lin, bpf = source_spectrum_one_rotor(freqs, cal['rpm'], prop_row['diameter_m'],
                                                     int(prop_row['blades']), cfg['source_model'],
                                                     modifiers_for_conf(conf_row))
            theta = rotor_to_mic_theta(cfg['geometry']['drone_altitude_m'], cal['distance_m'])
            direct = directivity_gain(theta, cfg['directivity']['model'], cfg['directivity']['q'])
            specA_lin = propagate_and_weight(freqs, src_lin, cal['distance_m'], cfg['environment'], direct)
            la_pred_one = laeq_from_spectrum(freqs, specA_lin)
            la_pred_quad = laeq_from_spectrum(freqs, sum_four_rotors([specA_lin]*4))
            cal_gain_db = cal['laeq_db'] - la_pred_quad  # shift so model matches datum

    # Determine RPM for hover thrust, unless rpm_target provided as override
    rpm = float(rpm_target) if rpm_target > 0 else thrust_to_rpm(cfg['eval']['hover_thrust_N'],
                                                                 prop_row['diameter_m'],
                                                                 cfg['environment']['rho_air'],
                                                                 cfg['source_model']['k_T'])
    # Build one-rotor source @1m
    src_lin, bpf = source_spectrum_one_rotor(freqs, rpm, prop_row['diameter_m'],
                                             int(prop_row['blades']), cfg['source_model'],
                                             modifiers_for_conf(conf_row))
    # Distance curve
    la_list = []
    for d in distances:
        theta = rotor_to_mic_theta(cfg['geometry']['drone_altitude_m'], d)
        direct = directivity_gain(theta, cfg['directivity']['model'], cfg['directivity']['q'])
        specA_lin_one = propagate_and_weight(freqs, src_lin, d, cfg['environment'], direct)
        specA_lin_quad = sum_four_rotors([specA_lin_one]*4)
        la = laeq_from_spectrum(freqs, specA_lin_quad) + cal_gain_db
        la_list.append(la)

    # Save curve plot + CSV
    os.makedirs(os.path.join(out_dir, "curves"), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(distances, la_list)
    plt.xlabel("Distance (m)")
    plt.ylabel("LAeq (dB)")
    plt.title(f"{conf_row['config_id']} — LAeq vs Distance (RPM≈{int(rpm)})")
    plt.tight_layout()
    curve_png = os.path.join(out_dir, "curves", f"curve_{conf_row['config_id']}.png")
    plt.savefig(curve_png, dpi=200)
    plt.close()

    curve_csv = os.path.join(out_dir, "curves", f"curve_{conf_row['config_id']}.csv")
    pd.DataFrame({"distance_m": distances, "LAeq_dB": la_list}).to_csv(curve_csv, index=False)

    # Spectrogram (at reference distance)
    ref_d = float(cfg['eval']['spectrogram_at_distance_m'])
    theta = rotor_to_mic_theta(cfg['geometry']['drone_altitude_m'], ref_d)
    direct = directivity_gain(theta, cfg['directivity']['model'], cfg['directivity']['q'])
    specA_lin_one = propagate_and_weight(freqs, src_lin, ref_d, cfg['environment'], direct)
    specA_lin_quad = sum_four_rotors([specA_lin_one]*4)
    la_ref = laeq_from_spectrum(freqs, specA_lin_quad) + cal_gain_db

    os.makedirs(os.path.join(out_dir, "spectrograms"), exist_ok=True)
    if HAVE_LIBROSA and cfg['eval']['make_spectrograms']:
        # Synthesize 10 s signal with tones + shaped noise to match the spectrum roughly
        sr = int(cfg['eval']['sample_rate'])
        dur = float(cfg['eval']['spectrogram_seconds'])
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        # Tones: BPF and a couple harmonics
        sig = np.zeros_like(t)
        amps = [1.0, 0.5, 0.3]
        for h,a in zip([1,2,3], amps):
            sig += a*np.sin(2*np.pi*(h*bpf)*t)
        # Broadband noise
        rng = np.random.default_rng(1234)
        noise = rng.standard_normal(len(t))
        # Simple lowpass to ~8 kHz to mimic HF roll-off
        from scipy.signal import butter, lfilter
        b,a = butter(4, 8000/(sr/2), btype='low')
        noise = lfilter(b,a,noise)
        sig = sig + 0.4*noise
        # Normalize roughly to LAeq (not exact)
        sig = sig / (np.std(sig)+1e-9)

        S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(6,4))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.title(f"{conf_row['config_id']} — Mel Spectrogram (ref {ref_d} m)")
        plt.tight_layout()
        spec_png = os.path.join(out_dir, "spectrograms", f"melspec_{conf_row['config_id']}.png")
        plt.savefig(spec_png, dpi=200)
        plt.close()

        npy_path = os.path.join(out_dir, "spectrograms", f"melspec_{conf_row['config_id']}.npy")
        np.save(npy_path, S_db)

    # Footprint (2D LAeq map on ground)
    os.makedirs(os.path.join(out_dir, "footprints"), exist_ok=True)
    x_min, x_max = cfg['eval']['footprint']['x_min_m'], cfg['eval']['footprint']['x_max_m']
    y_min, y_max = cfg['eval']['footprint']['y_min_m'], cfg['eval']['footprint']['y_max_m']
    step = cfg['eval']['footprint']['step_m']
    xs = np.arange(x_min, x_max+step, step)
    ys = np.arange(y_min, y_max+step, step)
    grid = np.zeros((len(ys), len(xs)))
    # mic at ground (height ~ mic_height_m), drone at (0,0,alt)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            d = np.sqrt(x**2 + y**2)  # horizontal distance
            theta = rotor_to_mic_theta(cfg['geometry']['drone_altitude_m'], d)
            direct = directivity_gain(theta, cfg['directivity']['model'], cfg['directivity']['q'])
            specA_lin_one = propagate_and_weight(freqs, src_lin, np.sqrt(d**2 + (cfg['geometry']['drone_altitude_m']-cfg['geometry']['mic_height_m'])**2),
                                                 cfg['environment'], direct)
            specA_lin_quad = sum_four_rotors([specA_lin_one]*4)
            grid[iy, ix] = laeq_from_spectrum(freqs, specA_lin_quad) + cal_gain_db

    plt.figure(figsize=(6,5))
    im = plt.imshow(grid, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='equal')
    plt.colorbar(label="LAeq (dB)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"{conf_row['config_id']} — Ground SPL Footprint (drone at 0,0)")
    plt.tight_layout()
    foot_png = os.path.join(out_dir, "footprints", f"footprint_{conf_row['config_id']}.png")
    plt.savefig(foot_png, dpi=200)
    plt.close()

    # Summary row (use 10/20/50/100 m if present)
    snap_dists = [10, 20, 50, 100]
    snap = {}
    for sd in snap_dists:
        if sd in distances:
            idx = np.where(distances == sd)[0][0]
            snap[f"LAeq_{sd}m_dB"] = la_list[idx]
    row = {
        "config_id": conf_row['config_id'],
        "prop": conf_row['prop'],
        "shroud": conf_row['shroud'],
        "liner": conf_row['liner'],
        "isolation": conf_row['isolation'],
        "added_mass_g": float(conf_row.get('added_mass_g', 0.0)),
        "rpm_hover": float(rpm),
        "bpf_hz": float(bpf),
        **snap
    }
    return row

def modifiers_for_conf(conf_row):
    return {
        "has_shroud": str(conf_row['shroud']).lower() != 'none',
        "has_liner": str(conf_row['liner']).lower() == 'on',
        "has_isolation": str(conf_row['isolation']).lower() == 'on',
        # dB deltas (defaults, can be adjusted in source_model too)
        "shroud_bb_delta_db": -0.5,
        "liner_bb_delta_db": -1.5,
        "iso_lowband_delta_db": -1.0,
    }

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    cfg, props, confs, rpm_sets, cal = load_all(base)

    # If calibration.json exists but config.yml says disabled, copy that point into cfg to use it
    if cal and not cfg.get('calibration', {}).get('enabled', False):
        cfg['calibration'] = {"enabled": True, "cal_point": cal}

    os.makedirs(os.path.join(base, "outputs", "curves"), exist_ok=True)
    os.makedirs(os.path.join(base, "outputs", "spectrograms"), exist_ok=True)
    os.makedirs(os.path.join(base, "outputs", "footprints"), exist_ok=True)
    os.makedirs(os.path.join(base, "outputs", "summary_tables"), exist_ok=True)

    # Choose the first rpm_set row (e.g., 'hover') by default
    rpm_set = rpm_sets.iloc[0]
    thrust_N = float(rpm_set.get('thrust_N', 10.0))
    rpm_override = float(rpm_set.get('rpm_override', 0.0))  # leave 0 to use thrust→RPM

    # Merge cfg values for convenience
    cfg['eval']['hover_thrust_N'] = thrust_N

    summary_rows = []
    for _, conf in confs.iterrows():
        prop_name = conf['prop']
        prop_row = props.loc[props['name'] == prop_name]
        if prop_row.empty:
            print(f"[warn] Unknown prop '{prop_name}' for config '{conf['config_id']}', skipping.")
            continue
        prop_row = prop_row.iloc[0]
        row = simulate_config_LAeq_curve(cfg, prop_row, conf, rpm_override, os.path.join(base, "outputs"))
        summary_rows.append(row)

    # Save summary table
    df = pd.DataFrame(summary_rows)
    # Compute benefit-per-gram vs the FIRST config as baseline if LAeq_50m exists
    if 'LAeq_50m_dB' in df.columns:
        base_val = float(df['LAeq_50m_dB'].iloc[0])
        base_mass = float(df['added_mass_g'].iloc[0])
        deltas = []
        for i, r in df.iterrows():
            dB_reduction = base_val - float(r['LAeq_50m_dB'])
            extra_mass_g = max(float(r['added_mass_g']) - base_mass, 0.0)
            bpg = dB_reduction/extra_mass_g if extra_mass_g > 0 else np.nan
            deltas.append((dB_reduction, bpg))
        df['delta_LAeq_50m_dB_vs_first'] = [d[0] for d in deltas]
        df['benefit_per_gram_dB_per_g'] = [d[1] for d in deltas]

    out_csv = os.path.join(base, "outputs", "summary_tables", "comparison.csv")
    df.to_csv(out_csv, index=False)
    print("Done. See outputs/ for curves, spectrograms (if librosa+scipy installed), footprints, and summary_tables.")

if __name__ == "__main__":
    main()

