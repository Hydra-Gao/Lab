import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR

PLOT_MODE = "all"  # all / significant
SIGNIFICANCE_COLUMNS = [
    "is_motion_baseline_responsive",
    "is_motion_baseline_suppressed",
    "is_direction_tuned_motion_baseline",
]
SCREEN_ORDER = ["left", "front", "right"]
TIME_RANGE = (-2.0, 4.0)
BIN_WIDTH = 0.1
DIRECTION_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]


def sem(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def format_value(x, ndigits=4):
    if pd.isna(x):
        return "nan"
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{x:.{ndigits}f}"


def direction_label(direction):
    try:
        d = float(direction)
        if d.is_integer():
            return str(int(d))
        return str(d)
    except Exception:
        return str(direction)


def sorted_unique_nonnull(values):
    vals = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def close_polar(theta, r):
    theta = np.asarray(theta)
    r = np.asarray(r)
    if len(theta) == 0:
        return theta, r
    return np.r_[theta, theta[0]], np.r_[r, r[0]]


def ensure_columns(df, table_name, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def get_screen_col(df):
    for col in ["active_screen_role", "screen_role"]:
        if col in df.columns:
            return col
    raise ValueError("No screen-role column found.")


def normalize_screen_speed_columns(df):
    if df is None:
        return None
    df = df.copy()
    for col in ["active_screen_role", "screen_role"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "speed" in df.columns:
        df["speed"] = df["speed"].astype(str).str.strip()
    return df


def get_units_to_plot(units, sig):
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()
    if sig is None:
        raise FileNotFoundError("unit_significance_summary.csv not found, but PLOT_MODE='significant'.")
    mask = np.zeros(len(sig), dtype=bool)
    for col in SIGNIFICANCE_COLUMNS:
        if col in sig.columns:
            mask |= sig[col].fillna(False).astype(bool)
    return sorted_unique_nonnull(sig.loc[mask, "unit_id"])


def filter_unit_screen(df, unit_id, screen_role=None):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[df["unit_id"] == unit_id].copy()
    scol = get_screen_col(out)
    if screen_role is not None:
        out = out[out[scol] == screen_role].copy()
    return out


def plot_screen_direction_response_curve(ax, trial_unit_screen, screen_role):
    df = trial_unit_screen[["direction", "trial_id", "moving_minus_baseline"]].dropna(subset=["direction", "moving_minus_baseline"]).copy()
    if df.empty:
        ax.set_title(f"{screen_role}: direction response")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    directions = [d for d in DIRECTION_ORDER if d in set(pd.to_numeric(df["direction"], errors="coerce"))]
    summary = (
        df.groupby("direction", dropna=False)["moving_minus_baseline"]
        .agg(mean="mean", sem=sem)
        .reindex(directions)
        .reset_index()
    )
    x = np.arange(len(directions))
    rng = np.random.default_rng(42)
    for i, direction in enumerate(directions):
        y_raw = df.loc[df["direction"] == direction, "moving_minus_baseline"].dropna().to_numpy(dtype=float)
        if len(y_raw) == 0:
            continue
        x_raw = i + rng.uniform(-0.08, 0.08, size=len(y_raw))
        ax.scatter(x_raw, y_raw, s=16, alpha=0.45, linewidths=0)
    ax.errorbar(x, summary["mean"].values, yerr=summary["sem"].values, fmt='-o', capsize=4, linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([direction_label(d) for d in directions])
    ax.set_xlabel("Direction")
    ax.set_ylabel("Moving - baseline FR")
    ax.set_title(f"{screen_role}: direction response")


def plot_screen_static_moving_polar(ax, condition_unit_screen, screen_role):
    baseline_col = "pooled_baseline_fr" if "pooled_baseline_fr" in condition_unit_screen.columns else ("baseline_fr" if "baseline_fr" in condition_unit_screen.columns else "static_fr")
    df = condition_unit_screen[["direction", baseline_col, "moving_fr"]].dropna(subset=["direction"]).copy()
    if df.empty:
        ax.set_title(f"{screen_role}: baseline vs moving")
        return
    df = (
        df.groupby("direction", as_index=False)
        .agg(baseline=(baseline_col, "mean"), moving_fr=("moving_fr", "mean"))
        .sort_values("direction")
    )
    theta = np.deg2rad(df["direction"].astype(float).values)
    for col, label in [("baseline", "baseline FR"), ("moving_fr", "moving FR")]:
        r = df[col].clip(lower=0).values
        theta_c, r_c = close_polar(theta, r)
        ax.plot(theta_c, r_c, marker='o', linewidth=1.6, label=label)
        ax.fill(theta_c, r_c, alpha=0.08)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(DIRECTION_ORDER))
    ax.set_xticklabels([str(int(d)) for d in DIRECTION_ORDER])
    ax.set_title(f"{screen_role}: baseline vs moving")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)


def add_screen_stats_text(ax, unit_id, screen_role, sig_unit_screen):
    ax.axis("off")
    lines = [f"Unit {unit_id}", f"Screen: {screen_role}", ""]
    if sig_unit_screen is None or sig_unit_screen.empty:
        lines += ["No significance data"]
    else:
        row = sig_unit_screen.iloc[0]
        lines += [
            f"p_motion_baseline_two_sided = {format_value(row.get('p_motion_baseline_two_sided', np.nan))}",
            f"p_direction_tuning_motion_baseline = {format_value(row.get('p_direction_tuning_motion_baseline', np.nan))}",
            f"q_motion_baseline = {format_value(row.get('q_motion_baseline', np.nan))}",
            f"q_direction_tuning_motion_baseline = {format_value(row.get('q_direction_tuning_motion_baseline', np.nan))}",
        ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8)


def plot_cross_screen_summary_page(pdf, unit_id, trial_unit, condition_unit, sig_unit):
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 0.8], hspace=0.45, wspace=0.35)
    for col, screen_role in enumerate(SCREEN_ORDER):
        trial_screen = filter_unit_screen(trial_unit, unit_id, screen_role)
        condition_screen = filter_unit_screen(condition_unit, unit_id, screen_role)
        sig_screen = filter_unit_screen(sig_unit, unit_id, screen_role) if sig_unit is not None else pd.DataFrame()
        ax_curve = fig.add_subplot(gs[0, col])
        plot_screen_direction_response_curve(ax_curve, trial_screen, screen_role)
        ax_polar = fig.add_subplot(gs[1, col], projection='polar')
        plot_screen_static_moving_polar(ax_polar, condition_screen, screen_role)
        ax_text = fig.add_subplot(gs[2, col])
        add_screen_stats_text(ax_text, unit_id, screen_role, sig_screen)
    fig.suptitle(f"Unit {unit_id} | 8-direction cross-screen summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def plot_raster(ax, labeled_ss, directions):
    y = 0
    yticks = []
    ylabels = []
    for direction in directions:
        df_dir = labeled_ss[labeled_ss["direction"] == direction]
        trial_ids = sorted_unique_nonnull(df_dir["trial_id"])
        if len(trial_ids) == 0:
            continue
        start_y = y
        for trial_id in trial_ids:
            df_trial = df_dir[df_dir["trial_id"] == trial_id]
            spike_times = df_trial["time_from_moving_onset"].to_numpy(dtype=float)
            ax.vlines(spike_times, y - 0.4, y + 0.4, linewidth=0.5)
            y += 1
        end_y = y - 1
        yticks.append((start_y + end_y) / 2)
        ylabels.append(direction_label(direction))
        ax.axhline(y - 0.5, linewidth=0.5, alpha=0.4)
        y += 1
    ax.axvspan(TIME_RANGE[0], 0, alpha=0.08)
    ax.axvspan(0, TIME_RANGE[1], alpha=0.08)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Direction / trials")
    if y == 0:
        ax.text(0.5, 0.5, "No spikes / trials", ha="center", va="center", transform=ax.transAxes)


def plot_psth_one_direction(ax, labeled_ss, direction):
    df_dir = labeled_ss[labeled_ss["direction"] == direction]
    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2
    n_trials = df_dir["trial_id"].nunique()
    if n_trials > 0:
        counts, _ = np.histogram(df_dir["time_from_moving_onset"].to_numpy(dtype=float), bins=bins)
        fr = counts / n_trials / BIN_WIDTH
        ax.plot(centers, fr, linewidth=1.2)
    ax.axvspan(TIME_RANGE[0], 0, alpha=0.08)
    ax.axvspan(0, TIME_RANGE[1], alpha=0.08)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_title(f"Dir {direction_label(direction)}", fontsize=9)


def plot_screen_page(pdf, unit_id, screen_role, labeled_unit):
    labeled_ss = filter_unit_screen(labeled_unit, unit_id, screen_role)
    if labeled_ss.empty:
        return
    directions = [d for d in DIRECTION_ORDER if d in set(pd.to_numeric(labeled_ss["direction"], errors="coerce"))]
    fig = plt.figure(figsize=(16, 10))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)
    ax_raster = fig.add_subplot(outer[0, 0])
    plot_raster(ax_raster, labeled_ss, directions)
    right = outer[0, 1].subgridspec(4, 2, hspace=0.55, wspace=0.35)
    for idx, direction in enumerate(directions[:8]):
        ax = fig.add_subplot(right[idx // 2, idx % 2])
        plot_psth_one_direction(ax, labeled_ss, direction)
        if idx // 2 == 3:
            ax.set_xlabel("Time (s)")
        if idx % 2 == 0:
            ax.set_ylabel("FR")
    fig.suptitle(f"Unit {unit_id} | {screen_role} screen", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def plot_one_unit(unit_id, labeled, trial_summary, condition_summary, sig, out_dir):
    labeled_unit = labeled[labeled["unit_id"] == unit_id].copy()
    trial_unit = trial_summary[trial_summary["unit_id"] == unit_id].copy()
    condition_unit = condition_summary[condition_summary["unit_id"] == unit_id].copy()
    sig_unit = sig[sig["unit_id"] == unit_id].copy() if sig is not None and not sig.empty else pd.DataFrame()
    if labeled_unit.empty or trial_unit.empty or condition_unit.empty:
        return
    out_path = out_dir / f"unit_{unit_id}_8direction_summary.pdf"
    with PdfPages(out_path) as pdf:
        plot_cross_screen_summary_page(pdf, unit_id, trial_unit, condition_unit, sig_unit)
        for screen_role in SCREEN_ORDER:
            plot_screen_page(pdf, unit_id, screen_role, labeled_unit)
    print(f"Saved: {out_path}")


def main():
    print("===== Plot NP 8-direction unit summaries =====")
    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units_8directions_np"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv")
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / "curated_units.csv")
    trial_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv")
    condition_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv")
    sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() else None

    ensure_columns(labeled, "labeled_spikes.csv", ["unit_id", "trial_id", "direction", "time_from_moving_onset"])
    ensure_columns(trial_summary, "unit_trial_summary.csv", ["unit_id", "trial_id", "direction", "moving_minus_baseline"])
    ensure_columns(condition_summary, "unit_condition_summary.csv", ["unit_id", "direction", "moving_fr"])
    labeled = normalize_screen_speed_columns(labeled)
    trial_summary = normalize_screen_speed_columns(trial_summary)
    condition_summary = normalize_screen_speed_columns(condition_summary)
    sig = normalize_screen_speed_columns(sig)

    # keep only 8-direction rows
    if "trial_kind" in labeled.columns:
        labeled = labeled[labeled["trial_kind"] == "single_screen_8direction"].copy()
    if "trial_kind" in trial_summary.columns:
        trial_summary = trial_summary[trial_summary["trial_kind"] == "single_screen_8direction"].copy()
    if "trial_kind" in condition_summary.columns:
        condition_summary = condition_summary[condition_summary["trial_kind"] == "single_screen_8direction"].copy()

    units_to_plot = get_units_to_plot(units, sig)
    print(f"Units to plot: {len(units_to_plot)}")
    for unit_id in units_to_plot:
        plot_one_unit(unit_id, labeled, trial_summary, condition_summary, sig, plot_dir)
    print(plot_dir)


if __name__ == "__main__":
    main()
