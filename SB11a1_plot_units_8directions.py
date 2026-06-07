# Unit-level plots for combined single-screen 8-direction / 2-speed stimulus
# recorded across multiple screen locations.
#
# Expected analysis structure:
#   unit_trial_summary.csv
#       unit × trial, with screen_role and speed
#
#   unit_condition_summary.csv
#       unit × screen_role × speed × direction
#
#   unit_tuning_summary.csv
#       unit × screen_role × speed
#
#   unit_significance_summary.csv
#       unit × screen_role × speed
#
#   unit_direction_significance.csv
#       unit × screen_role × speed × direction
#
# PDF structure for each unit:
#   Page 1:
#       Cross-screen comparison summary
#       Row 1: direction response curve for front / left / right
#       Row 2: polar plot for front / left / right
#              static FR and moving FR together
#       Row 3: screen-specific statistics text
#
#   Page 2+:
#       Front screen, speed 2.0
#       Front screen, speed 3.2 / 32.0
#       Left screen, speed 2.0
#       Left screen, speed 3.2 / 32.0
#       Right screen, speed 2.0
#       Right screen, speed 3.2 / 32.0
#
# Each screen × speed page contains:
#       raster by direction
#       PSTH by direction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


# =====================
# Plot settings
# =====================

PLOT_MODE = "all"  # "all" or "significant"

SIGNIFICANCE_COLUMNS = [
    "is_motion_baseline_responsive",
    "is_motion_baseline_suppressed",
    "is_direction_tuned_motion_baseline",
]

SCREEN_ORDER = ["front", "left", "right"]

TIME_RANGE = (-3.0, 5.0)
BIN_WIDTH = 0.1

# Whether Page 1 cross-screen summary averages across speeds.
# Detailed pages are always screen × speed specific.
CROSS_SCREEN_SUMMARY_POOL_SPEEDS = True


# =====================
# Basic helpers
# =====================

def sem(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def format_value(x, ndigits=3):
    if pd.isna(x):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{x:.{ndigits}f}"
    return str(x)


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


def format_polar_axis(ax, title):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0", "45", "90", "135", "180", "225", "270", "315"])
    ax.set_title(title)


def normalize_screen_speed_columns(df):
    if df is None:
        return None

    if "screen_role" in df.columns:
        df["screen_role"] = df["screen_role"].astype(str).str.strip()
    if "speed" in df.columns:
        df["speed"] = df["speed"].astype(str).str.strip()

    return df


def ensure_columns(df, table_name, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def get_screen_order_available(*dfs):
    screens = []

    for screen in SCREEN_ORDER:
        for df in dfs:
            if df is not None and "screen_role" in df.columns:
                if screen in set(df["screen_role"].dropna().astype(str)):
                    screens.append(screen)
                    break

    # Include any nonstandard screen_role values after the preferred order.
    extras = []
    for df in dfs:
        if df is not None and "screen_role" in df.columns:
            vals = sorted_unique_nonnull(df["screen_role"])
            for v in vals:
                if v not in screens and v not in extras:
                    extras.append(v)

    return screens + extras


def get_speed_order_available(condition_unit_screen):
    speeds = sorted_unique_nonnull(condition_unit_screen["speed"])
    return speeds


def filter_unit_screen_speed(df, unit_id, screen_role=None, speed=None):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df[df["unit_id"] == unit_id].copy()

    if screen_role is not None and "screen_role" in out.columns:
        out = out[out["screen_role"] == screen_role].copy()

    if speed is not None and "speed" in out.columns:
        out = out[out["speed"] == speed].copy()

    return out


def get_units_to_plot(units, sig):
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()

    if PLOT_MODE == "significant":
        if sig is None:
            raise FileNotFoundError(
                "unit_significance_summary.csv not found, "
                "but PLOT_MODE='significant'."
            )

        mask = np.zeros(len(sig), dtype=bool)

        for col in SIGNIFICANCE_COLUMNS:
            if col in sig.columns:
                mask |= sig[col].fillna(False).astype(bool)

        return sorted_unique_nonnull(sig.loc[mask, "unit_id"])

    raise ValueError("PLOT_MODE must be 'all' or 'significant'.")


# =====================
# Page 1: cross-screen summary
# =====================

def plot_screen_direction_response_curve(ax, trial_unit_screen, screen_role):
    """
    Direction response curve for one screen.

    Uses trial-level moving_minus_baseline.
    If multiple speeds exist, this summary pools speeds for the cross-screen page.
    Detailed screen × speed pages are shown later.
    """
    df = trial_unit_screen[
        ["direction", "trial_id", "moving_minus_baseline"]
    ].dropna(subset=["direction", "moving_minus_baseline"]).copy()

    if df.empty:
        ax.set_title(f"{screen_role}: direction response")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    directions = sorted_unique_nonnull(df["direction"])

    summary = (
        df.groupby("direction", dropna=False)["moving_minus_baseline"]
        .agg(mean="mean", sem=sem)
        .reindex(directions)
        .reset_index()
    )

    x = np.arange(len(directions))

    ax.errorbar(
        x,
        summary["mean"].values,
        yerr=summary["sem"].values,
        fmt="-o",
        capsize=4,
        linewidth=1.5,
    )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([direction_label(d) for d in directions])
    ax.set_xlabel("Direction")
    ax.set_ylabel("Moving - static/baseline FR")
    ax.set_title(f"{screen_role}: direction response")


def plot_screen_static_moving_polar(ax, condition_unit_screen, screen_role):
    """
    One polar plot per screen.

    Plots static FR and moving FR together.
    If multiple speeds exist, this cross-screen summary averages across speeds.
    """
    df = condition_unit_screen[
        ["direction", "static_fr", "moving_fr"]
    ].dropna(subset=["direction"]).copy()

    if df.empty:
        format_polar_axis(ax, f"{screen_role}: static vs moving")
        return

    df = (
        df.groupby("direction", as_index=False)
        .agg(
            static_fr=("static_fr", "mean"),
            moving_fr=("moving_fr", "mean"),
        )
        .sort_values("direction")
    )

    theta = np.deg2rad(df["direction"].astype(float).values)
    theta_c, _ = close_polar(theta, np.zeros(len(theta)))

    for col, label in [
        ("static_fr", "static FR"),
        ("moving_fr", "moving FR"),
    ]:
        r = df[col].clip(lower=0).values
        _, r_c = close_polar(theta, r)
        ax.plot(theta_c, r_c, marker="o", linewidth=1.6, label=label)
        ax.fill(theta_c, r_c, alpha=0.08)

    format_polar_axis(ax, f"{screen_role}: static vs moving")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


def add_screen_stats_text(ax, unit_id, screen_role, tuning_unit_screen, sig_unit_screen):
    """
    Third row of Page 1.

    Shows one compact statistics block per screen.
    Since tuning/significance is screen × speed, list both speeds under each screen.
    """
    ax.axis("off")

    lines = [
        f"Unit {unit_id}",
        f"Screen: {screen_role}",
    ]

    if tuning_unit_screen is None or tuning_unit_screen.empty:
        lines += ["", "No tuning data"]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8)
        return

    tuning_unit_screen = tuning_unit_screen.sort_values("speed").copy()

    for _, row in tuning_unit_screen.iterrows():
        speed = row.get("speed", np.nan)

        lines += [
            "",
            f"Speed: {speed}",
            f"Class: {row.get('response_class', 'NA')}",
            f"PD method: {row.get('pd_method', 'NA')}",
            f"Mean static FR: {format_value(row.get('mean_static_fr', np.nan), 2)}",
            f"Mean moving FR: {format_value(row.get('mean_moving_fr', np.nan), 2)}",
            f"Mean moving-static/base: {format_value(row.get('mean_moving_minus_baseline', np.nan), 2)}",
            f"Primary PD: {format_value(row.get('preferred_direction', np.nan), 2)}",
            f"Primary DSI: {format_value(row.get('dsi', np.nan), 3)}",
            f"Vector strength: {format_value(row.get('vector_strength', np.nan), 3)}",
        ]

        if sig_unit_screen is not None and not sig_unit_screen.empty:
            sig_row = sig_unit_screen[sig_unit_screen["speed"] == speed]
            if not sig_row.empty:
                s = sig_row.iloc[0]
                lines += [
                    f"p motion: {format_value(s.get('p_motion_baseline_two_sided', np.nan), 4)}",
                    f"q motion: {format_value(s.get('q_motion_baseline', np.nan), 4)}",
                    f"p direction: {format_value(s.get('p_direction_tuning_motion_baseline', np.nan), 4)}",
                    f"q direction: {format_value(s.get('q_direction_tuning_motion_baseline', np.nan), 4)}",
                ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=7.5,
    )


def plot_cross_screen_summary_page(
    pdf,
    unit_id,
    trial_unit,
    condition_unit,
    tuning_unit,
    sig_unit,
):
    screens = get_screen_order_available(trial_unit, condition_unit, tuning_unit, sig_unit)

    if len(screens) == 0:
        return

    # Always allocate 3 columns for front / left / right layout.
    # If a screen is missing, that panel will say No data.
    plot_screens = SCREEN_ORDER

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.1, 1.2],
        hspace=0.45,
        wspace=0.4,
    )

    for col, screen_role in enumerate(plot_screens):
        trial_screen = (
            trial_unit[trial_unit["screen_role"] == screen_role].copy()
            if "screen_role" in trial_unit.columns
            else pd.DataFrame()
        )

        condition_screen = (
            condition_unit[condition_unit["screen_role"] == screen_role].copy()
            if "screen_role" in condition_unit.columns
            else pd.DataFrame()
        )

        tuning_screen = (
            tuning_unit[tuning_unit["screen_role"] == screen_role].copy()
            if "screen_role" in tuning_unit.columns
            else pd.DataFrame()
        )

        sig_screen = (
            sig_unit[sig_unit["screen_role"] == screen_role].copy()
            if sig_unit is not None and not sig_unit.empty and "screen_role" in sig_unit.columns
            else pd.DataFrame()
        )

        ax_curve = fig.add_subplot(gs[0, col])
        plot_screen_direction_response_curve(
            ax_curve,
            trial_screen,
            screen_role,
        )

        ax_polar = fig.add_subplot(gs[1, col], projection="polar")
        plot_screen_static_moving_polar(
            ax_polar,
            condition_screen,
            screen_role,
        )

        ax_text = fig.add_subplot(gs[2, col])
        add_screen_stats_text(
            ax_text,
            unit_id,
            screen_role,
            tuning_screen,
            sig_screen,
        )

    fig.suptitle(
        f"Unit {unit_id} cross-screen comparison summary\n"
        f"Page 1 summary pools speeds for direction/polar panels; later pages are screen × speed specific.",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# =====================
# Screen × speed pages
# =====================

def plot_raster_by_direction(ax, labeled_unit_screen_speed, directions, screen_role, speed):
    y = 0
    yticks = []
    ylabels = []

    for direction in directions:
        df_dir = labeled_unit_screen_speed[
            labeled_unit_screen_speed["direction"] == direction
        ]

        for trial_id, df_trial in df_dir.groupby("trial_id"):
            spike_times = df_trial["time_from_moving_onset"].values
            ax.vlines(
                spike_times,
                y - 0.4,
                y + 0.4,
                linewidth=0.5,
            )
            y += 1

        if len(df_dir) > 0:
            yticks.append(y - 0.5)
            ylabels.append(direction_label(direction))

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Direction / trials")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title(f"Raster by direction | {screen_role}, speed = {speed}")


def plot_psth_by_direction(ax, labeled_unit_screen_speed, directions, screen_role, speed):
    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    for direction in directions:
        df_dir = labeled_unit_screen_speed[
            labeled_unit_screen_speed["direction"] == direction
        ]

        n_trials = df_dir["trial_id"].nunique()

        if n_trials == 0:
            continue

        counts, _ = np.histogram(
            df_dir["time_from_moving_onset"],
            bins=bins,
        )

        fr = counts / n_trials / BIN_WIDTH

        ax.plot(
            centers,
            fr,
            label=direction_label(direction),
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    ax.set_title(f"PSTH by direction | {screen_role}, speed = {speed}")
    ax.legend(title="Direction", fontsize=8, ncol=4)


def add_screen_speed_summary_text(
    ax,
    unit_id,
    screen_role,
    speed,
    tuning_row,
    sig_row,
    dir_sig_speed,
):
    ax.axis("off")

    lines = [
        f"Unit {unit_id}",
        f"Screen: {screen_role}",
        f"Speed: {speed}",
    ]

    if tuning_row is not None and not tuning_row.empty:
        r = tuning_row.iloc[0]
        lines += [
            "",
            f"Class: {r.get('response_class', 'NA')}",
            f"PD method: {r.get('pd_method', 'NA')}",
            f"Positive dirs: {format_value(r.get('n_positive_directions', np.nan), 0)}",
            f"Negative dirs: {format_value(r.get('n_negative_directions', np.nan), 0)}",
            "",
            f"Mean static FR: {format_value(r.get('mean_static_fr', np.nan), 2)}",
            f"Mean moving FR: {format_value(r.get('mean_moving_fr', np.nan), 2)}",
            f"Mean moving-baseline: {format_value(r.get('mean_moving_minus_baseline', np.nan), 2)}",
            "",
            f"Primary PD: {format_value(r.get('preferred_direction', np.nan), 2)}",
            f"Primary DSI: {format_value(r.get('dsi', np.nan), 3)}",
            f"Vector strength: {format_value(r.get('vector_strength', np.nan), 3)}",
            "",
            f"Moving FR PD: {format_value(r.get('moving_fr_preferred_direction', np.nan), 2)}",
            f"Moving FR DSI: {format_value(r.get('moving_fr_dsi', np.nan), 3)}",
            f"Excitation PD: {format_value(r.get('excitation_preferred_direction', np.nan), 2)}",
            f"Excitation DSI: {format_value(r.get('excitation_dsi', np.nan), 3)}",
            f"Suppression PD: {format_value(r.get('suppression_preferred_direction', np.nan), 2)}",
            f"Suppression DSI: {format_value(r.get('suppression_dsi', np.nan), 3)}",
        ]

    if sig_row is not None and not sig_row.empty:
        r = sig_row.iloc[0]
        lines += [
            "",
            f"p motion-baseline 2s: {format_value(r.get('p_motion_baseline_two_sided', np.nan), 4)}",
            f"p responsive: {format_value(r.get('p_motion_baseline_responsive', np.nan), 4)}",
            f"p suppressed: {format_value(r.get('p_motion_baseline_suppressed', np.nan), 4)}",
            f"p direction tuning: {format_value(r.get('p_direction_tuning_motion_baseline', np.nan), 4)}",
            "",
            f"q motion-baseline: {format_value(r.get('q_motion_baseline', np.nan), 4)}",
            f"q direction tuning: {format_value(r.get('q_direction_tuning_motion_baseline', np.nan), 4)}",
        ]

    if dir_sig_speed is not None and not dir_sig_speed.empty:
        n_sig = int(
            dir_sig_speed
            .get("is_direction_response_significant", pd.Series(False))
            .fillna(False)
            .sum()
        )
        n_exc = int(
            dir_sig_speed
            .get("is_direction_excited", pd.Series(False))
            .fillna(False)
            .sum()
        )
        n_sup = int(
            dir_sig_speed
            .get("is_direction_suppressed", pd.Series(False))
            .fillna(False)
            .sum()
        )

        lines += [
            "",
            f"Significant directions: {n_sig}",
            f"Excited directions: {n_exc}",
            f"Suppressed directions: {n_sup}",
        ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8,
    )


def plot_screen_speed_page(
    pdf,
    unit_id,
    screen_role,
    speed,
    labeled_unit,
    trial_unit,
    condition_unit,
    tuning_unit,
    sig_unit,
    dir_sig_unit,
):
    labeled_ss = filter_unit_screen_speed(
        labeled_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    )

    trial_ss = filter_unit_screen_speed(
        trial_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    )

    condition_ss = filter_unit_screen_speed(
        condition_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    )

    tuning_ss = filter_unit_screen_speed(
        tuning_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    )

    sig_ss = filter_unit_screen_speed(
        sig_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    ) if sig_unit is not None else pd.DataFrame()

    dir_sig_ss = filter_unit_screen_speed(
        dir_sig_unit,
        unit_id,
        screen_role=screen_role,
        speed=speed,
    ) if dir_sig_unit is not None else pd.DataFrame()

    if labeled_ss.empty or condition_ss.empty:
        print(f"Skipping unit {unit_id}, {screen_role}, speed {speed}: no labeled/condition data.")
        return

    directions = sorted_unique_nonnull(condition_ss["direction"])

    if len(directions) == 0:
        print(f"Skipping unit {unit_id}, {screen_role}, speed {speed}: no directions.")
        return

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.4, 1.2, 0.9],
        hspace=0.45,
    )

    ax_raster = fig.add_subplot(gs[0, 0])
    plot_raster_by_direction(
        ax_raster,
        labeled_ss,
        directions,
        screen_role,
        speed,
    )

    ax_psth = fig.add_subplot(gs[1, 0])
    plot_psth_by_direction(
        ax_psth,
        labeled_ss,
        directions,
        screen_role,
        speed,
    )

    ax_text = fig.add_subplot(gs[2, 0])
    add_screen_speed_summary_text(
        ax_text,
        unit_id,
        screen_role,
        speed,
        tuning_ss,
        sig_ss,
        dir_sig_ss,
    )

    fig.suptitle(
        f"Unit {unit_id} | {screen_role} screen | speed = {speed}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =====================
# Unit PDF
# =====================

def plot_one_unit(
    unit_id,
    labeled,
    trial_summary,
    condition_summary,
    tuning_summary,
    sig,
    dir_sig,
    out_dir,
):
    labeled_unit = labeled[labeled["unit_id"] == unit_id].copy()
    trial_unit = trial_summary[trial_summary["unit_id"] == unit_id].copy()
    condition_unit = condition_summary[condition_summary["unit_id"] == unit_id].copy()
    tuning_unit = tuning_summary[tuning_summary["unit_id"] == unit_id].copy()

    sig_unit = (
        sig[sig["unit_id"] == unit_id].copy()
        if sig is not None and not sig.empty
        else pd.DataFrame()
    )

    dir_sig_unit = (
        dir_sig[dir_sig["unit_id"] == unit_id].copy()
        if dir_sig is not None and not dir_sig.empty
        else pd.DataFrame()
    )

    if labeled_unit.empty or trial_unit.empty or condition_unit.empty:
        print(f"Skipping unit {unit_id}: missing labeled/trial/condition data.")
        return

    out_path = out_dir / f"unit_{unit_id}_screen_speed_summary.pdf"

    with PdfPages(out_path) as pdf:

        # Page 1: cross-screen summary.
        plot_cross_screen_summary_page(
            pdf=pdf,
            unit_id=unit_id,
            trial_unit=trial_unit,
            condition_unit=condition_unit,
            tuning_unit=tuning_unit,
            sig_unit=sig_unit,
        )

        # Pages 2+: screen × speed.
        for screen_role in SCREEN_ORDER:
            condition_screen = condition_unit[
                condition_unit["screen_role"] == screen_role
            ].copy()

            if condition_screen.empty:
                continue

            speeds = get_speed_order_available(condition_screen)

            for speed in speeds:
                plot_screen_speed_page(
                    pdf=pdf,
                    unit_id=unit_id,
                    screen_role=screen_role,
                    speed=speed,
                    labeled_unit=labeled_unit,
                    trial_unit=trial_unit,
                    condition_unit=condition_unit,
                    tuning_unit=tuning_unit,
                    sig_unit=sig_unit,
                    dir_sig_unit=dir_sig_unit,
                )

    print(f"Saved: {out_path}")


# =====================
# Main
# =====================

def main():
    print("===== Plot screen- and speed-specific single-screen unit summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units_single_screen_screen_speed_split"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv")
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / "curated_units.csv")
    trial_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv")
    condition_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv")
    tuning_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv")

    sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    dir_sig_path = ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv"

    sig = pd.read_csv(sig_path) if sig_path.exists() else None
    dir_sig = pd.read_csv(dir_sig_path) if dir_sig_path.exists() else None

    # Required columns.
    ensure_columns(labeled, "labeled_spikes.csv", ["unit_id", "trial_id", "screen_role", "speed"])
    ensure_columns(trial_summary, "unit_trial_summary.csv", ["unit_id", "trial_id", "screen_role", "speed"])
    ensure_columns(condition_summary, "unit_condition_summary.csv", ["unit_id", "screen_role", "speed", "direction"])
    ensure_columns(tuning_summary, "unit_tuning_summary.csv", ["unit_id", "screen_role", "speed"])

    if sig is not None:
        ensure_columns(sig, "unit_significance_summary.csv", ["unit_id", "screen_role", "speed"])

    if dir_sig is not None:
        ensure_columns(dir_sig, "unit_direction_significance.csv", ["unit_id", "screen_role", "speed", "direction"])

    # Normalize screen/speed columns.
    labeled = normalize_screen_speed_columns(labeled)
    trial_summary = normalize_screen_speed_columns(trial_summary)
    condition_summary = normalize_screen_speed_columns(condition_summary)
    tuning_summary = normalize_screen_speed_columns(tuning_summary)
    sig = normalize_screen_speed_columns(sig)
    dir_sig = normalize_screen_speed_columns(dir_sig)

    print("\nCondition rows by screen_role and speed:")
    print(
        condition_summary
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    units_to_plot = get_units_to_plot(units, sig)

    print(f"\nUnits to plot: {len(units_to_plot)}")
    print(units_to_plot)

    for unit_id in units_to_plot:
        plot_one_unit(
            unit_id=unit_id,
            labeled=labeled,
            trial_summary=trial_summary,
            condition_summary=condition_summary,
            tuning_summary=tuning_summary,
            sig=sig,
            dir_sig=dir_sig,
            out_dir=plot_dir,
        )

    print("\n===== Saved unit plots to =====")
    print(plot_dir)


if __name__ == "__main__":
    main()