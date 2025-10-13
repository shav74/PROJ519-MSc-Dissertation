import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml, os

SIGMA = 5.670374419e-8  # Stefan–Boltzmann (W/m^2/K^4)

def load_inputs(base):
    import io
    # Be explicit about encodings on Windows
    nodes = pd.read_csv(os.path.join(base, "params_nodes.csv"), encoding="utf-8-sig")
    links = pd.read_csv(os.path.join(base, "params_links.csv"), encoding="utf-8-sig")
    prof  = pd.read_csv(os.path.join(base, "mission_profiles.csv"), encoding="utf-8-sig")

    cfg_path = os.path.join(base, "config.yml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_text = f.read()
    except UnicodeDecodeError:
        # Fallback if file was saved with Windows-1252 or has BOM oddities
        with open(cfg_path, "r", encoding="cp1252", errors="replace") as f:
            cfg_text = f.read()
        print("[warn] config.yml contained non-UTF-8 characters; loaded with cp1252 and replacement.")

    # Normalize any smart quotes to straight quotes to keep YAML happy
    cfg_text = (cfg_text
                .replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))

    import yaml
    cfg = yaml.safe_load(io.StringIO(cfg_text))
    return nodes, links, prof, cfg


def make_profile(time_s, prof_df):
    # Piecewise-constant powers for nodes listed in prof_df columns (except time)
    t = prof_df["time_s"].values
    def interp_col(col):
        return np.interp(time_s, t, prof_df[col].values, left=prof_df[col].values[0], right=prof_df[col].values[-1])
    return {col: interp_col(col) for col in prof_df.columns if col != "time_s"}

def simulate(nodes_df, links_df, prof_df, cfg):
    # Setup
    t_end   = cfg["sim"]["t_end_s"]
    dt      = cfg["sim"]["dt_s"]
    Tamb    = cfg["environment"]["ambient_C"] + 273.15
    h       = cfg["environment"]["h_conv_W_m2K"]
    ir_thr  = cfg["metrics"]["ir_threshold_C_above_amb"]
    nsteps  = int(t_end/dt) + 1
    time    = np.linspace(0, t_end, nsteps)

    # Node state arrays
    names = list(nodes_df["name"])
    idx   = {n:i for i,n in enumerate(names)}
    n     = len(names)
    T     = np.ones((nsteps, n)) * (Tamb)  # start at ambient
    C     = nodes_df["mass_kg"].values * nodes_df["cp_J_kgK"].values
    A     = nodes_df["area_m2"].values
    eps   = nodes_df["emissivity"].values
    # Optional multipliers for "mitigations"
    A_eff = A * nodes_df["area_scale"].values
    h_eff = h * nodes_df["h_scale"].values
    eps_eff = eps * nodes_df["eps_scale"].values

    # Build conduction adjacency (symmetric k_cond_eff W/K)
    K = np.zeros((n, n))
    for _, r in links_df.iterrows():
        i, j = idx[r["n1"]], idx[r["n2"]]
        k = r["k_cond_W_K"] * r["scale"]
        K[i, j] += k
        K[j, i] += k

    # Mission power inputs per node
    prof = make_profile(time, prof_df)  # dict name-> array of power W (positive adds heat)
    P_in = np.zeros((nsteps, n))
    for name in names:
        if name in prof:
            P_in[:, idx[name]] = prof[name]
    # Shell solar/extra load placeholder (optional)
    if "shell_extra_W" in prof:
        P_in[:, idx["shell"]] += prof["shell_extra_W"]

    # Integrate (explicit Euler, small dt)
    for k in range(1, nsteps):
        Tk = T[k-1, :]
        # Convection
        Q_conv = h_eff * A_eff * (Tk - Tamb)
        # Radiation
        Q_rad  = eps_eff * SIGMA * A_eff * (Tk**4 - Tamb**4)
        # Conduction between nodes
        Q_cond = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and K[i, j] > 0:
                    Q_cond[i] += K[i, j]*(Tk[i]-Tk[j])

        # Net power out (loss) positive when T > Tamb
        Q_loss = Q_conv + Q_rad + Q_cond
        dTdt   = (P_in[k, :] - Q_loss)/np.maximum(C, 1e-9)
        T[k, :] = Tk + dTdt*dt

    # Metrics: time-to-IR-threshold for shell (surface)
    shell_i = idx.get("shell", None)
    tt = None
    if shell_i is not None:
        over = (T[:, shell_i] - Tamb) >= (ir_thr)
        tt = time[over][0] if np.any(over) else np.nan

    return time, T, names, tt, Tamb

def plot_series(time, T, names, Tamb, outpath, title):
    plt.figure(figsize=(7,4))
    for i, n in enumerate(names):
        plt.plot(time/60.0, T[:, i]-273.15, label=n)  # °C
    plt.axhline(Tamb-273.15, linestyle="--")
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run_case(base_dir, case_tag):
    nodes, links, prof, cfg = load_inputs(base_dir)
    # Apply case toggles from config.yml
    if case_tag == "baseline":
        pass
    elif case_tag == "insulated":
        nodes["h_scale"]   *= cfg["cases"]["insulated"]["h_scale"]
        nodes["eps_scale"] *= cfg["cases"]["insulated"]["eps_scale"]
    elif case_tag == "heat_spreader":
        # increase conduction from motors/ESC to shell
        links.loc[links["type"]=="motor_to_shell", "scale"] *= cfg["cases"]["heat_spreader"]["k_scale"]
        links.loc[links["type"]=="esc_to_shell",   "scale"] *= cfg["cases"]["heat_spreader"]["k_scale"]
    elif case_tag == "insulated_plus_spreader":
        nodes["h_scale"]   *= cfg["cases"]["insulated"]["h_scale"]
        nodes["eps_scale"] *= cfg["cases"]["insulated"]["eps_scale"]
        links.loc[links["type"].isin(["motor_to_shell","esc_to_shell"]), "scale"] *= cfg["cases"]["heat_spreader"]["k_scale"]
    else:
        raise ValueError("Unknown case_tag")

    time, T, names, tthr, Tamb = simulate(nodes, links, prof, cfg)
    os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
    plot_series(time, T, names, Tamb, os.path.join(base_dir, "outputs", f"thermal_{case_tag}.png"),
                f"Thermal transient — {case_tag}")
    # Snapshot CSV @ end
    df = pd.DataFrame(T-273.15, columns=names)
    df.insert(0, "time_s", time)
    df.to_csv(os.path.join(base_dir, "outputs", f"thermal_{case_tag}.csv"), index=False)
    return tthr

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    cases = ["baseline", "insulated", "heat_spreader", "insulated_plus_spreader"]
    results = []
    for c in cases:
        tthr = run_case(base, c)
        results.append({"case": c, "time_to_IR_threshold_s": tthr})
    pd.DataFrame(results).to_csv(os.path.join(base, "outputs", "summary_time_to_threshold.csv"), index=False)
    print("Done. See 'outputs/' for plots and CSVs.")

if __name__ == "__main__":
    main()
