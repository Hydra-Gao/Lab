# SB11_plot_units_single_screen.py
#
# Unit-level plots for single-screen 8-direction / 2-speed stimulus.
#
# Expected upstream outputs:
#   labeled_spikes.csv
#   curated_units.csv
#   unit_trial_summary.csv
#   unit_condition_summary.csv
#   unit_direction_summary.csv                 # from updated SB05
#   unit_tuning_summary.csv
#   unit_significance_summary.csv              # from updated SB06
#   unit_direction_significance.csv            # from updated SB06
#
# Main changes from the older unit plotting script:
#   1. PSTH is split by speed.
#   2. Direction panels support 8 directions.
#   3. Main response plot uses signed moving - baseline FR.
#   4. Polar plots include pure moving FR and separated excitation/suppression components.
#   5. Summary text uses updated response_class / PD / DSI / significance columns.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


# =====================
# Plot settings
# =====================

# "all" or "significant"
PLOT_MODE = "all"

# Used only when PLOT_MODE == "significant"
SIGNIFICANCE_COLUMNS = [
    "is_motion_baseline_responsive",
    "is_motion_baseline_suppressed",
    "is_direction_tuned_motion_baseline",
]

TIME_RANGE = (-3.0, 5.0)
BIN_WIDTH = 0.1

# Page layout for 8 direction PSTHs
PSTH_DIRECTION_N_COLS = 4


# =====================
# Small helpers
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


def detect_speed_column(df):
    """Prefer speed_label, otherwise speed. Return None if neither exists."""
    if "speed_label" in df.columns:
        return "speed_label"
    if "speed" in df.columns:
        return "speed"
    return None


def sorted_unique_nonnull(values):
    vals = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


# =====================
# Loading / fallback helpers
# =====================


def get_units_to_plot(units, sig):
    """Choose all units or only significant units."""
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()

    if PLOT_MODE == "significant":
        if sig is None:
            raise FileNotFoundError(
                "unit_significance_summary.csv not found, "
                "but PLOT_MODE='significant'. Run updated SB06 first."
            )

        mask = np.zeros(len(sig), dtype=bool)

        for col in SIGNIFICANCE_COLUMNS:
            if col in sig.columns:
                mask |= sig[col].fillna(False).astype(bool)

        if "unit_id" not in sig.columns:
            raise ValueError("unit_significance_summary.csv must contain unit_id.")

        return sig.loc[mask, "unit_id"].tolist()

    raise ValueError("PLOT_MODE must be 'all' or 'significant'.")


def build_direction_summary_fallback(condition_summary):
    """
    Build a unit x direction summary if updated SB05 has not yet written
    unit_direction_summary.csv.
    """
    required = ["unit_id", "direction"]
    for col in required:
        if col not in condition_summary.columns:
            raise ValueError(f"condition_summary is missing required column: {col}")

    agg_cols = {}
    for col in [
        "n_trials",
        "baseline_fr",
        "static_fr",
        "early_fr",
        "sustained_fr",
        "moving_fr",
        "moving_minus_baseline",
        "moving_minus_static",
    ]:
        if col in condition_summary.columns:
            agg_cols[col] = (col, "mean")

    direction_summary = (
        condition_summary
        .groupby(["unit_id", "direction"], dropna=False)
        .agg(**agg_cols)
        .reset_index()
    )

    if "moving_minus_baseline" in direction_summary.columns:
        direction_summary["motion_baseline_positive"] = (
            direction_summary["moving_minus_baseline"].clip(lower=0)
        )
        direction_summary["motion_baseline_negative_strength"] = (
            -direction_summary["moving_minus_baseline"].clip(upper=0)
        )
    else:
        direction_summary["motion_baseline_positive"] = np.nan
        direction_summary["motion_baseline_negative_strength"] = np.nan

    return direction_summary


# =====================
# Raster
# =====================


def plot_raster(ax, labeled_unit, directions):
    """Raster plot grouped by direction, pooled across speeds."""
    y = 0
    yticks = []
    ylabels = []

    for direction in directions:
        df_dir = labeled_unit[labeled_unit["direction"] == direction]

        for trial_id, df_trial in df_dir.groupby("trial_id"):
            spike_times = df_trial["time_from_moving_onset"].values
            ax.vlines(spike_times, y - 0.4, y + 0.4, linewidth=0.5)
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
    ax.set_title("Raster aligned to moving onset\npooled across speeds")


# =====================
# PSTH by speed
# =====================


def plot_psth_by_direction(ax, labeled_unit, directions, speed_value=None, speed_col=None):
    """Combined PSTH: one line per direction, optionally filtered by speed."""
    df_plot = labeled_unit.copy()
    title_suffix = "pooled across speeds"

    if speed_value is not None and speed_col is not None and speed_col in df_plot.columns:
        df_plot = df_plot[df_plot[speed_col] == speed_value]
        title_suffix = f"speed = {speed_value}"

    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    for direction in directions:
        df_dir = df_plot[df_plot["direction"] == direction]
        n_trials = df_dir["trial_id"].nunique()

        if n_trials == 0:
            continue

        counts, _ = np.histogram(df_dir["time_from_moving_onset"], bins=bins)
        fr = counts / n_trials / BIN_WIDTH
        ax.plot(centers, fr, label=direction_label(direction))

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    ax.set_title(f"PSTH by direction\n{title_suffix}")
    ax.legend(title="Direction", fontsize=7, ncol=2)


def plot_psth_separated_by_speed(fig, gs_cell, labeled_unit, directions, speed_value, speed_col):
    """Small multiples: one PSTH panel per direction for one speed."""
    n = len(directions)
    n_cols = min(PSTH_DIRECTION_N_COLS, max(1, n))
    n_rows = int(np.ceil(n / n_cols))

    sub_gs = gs_cell.subgridspec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    df_speed = labeled_unit.copy()
    if speed_col is not None and speed_col in df_speed.columns:
        df_speed = df_speed[df_speed[speed_col] == speed_value]

    for i, direction in enumerate(directions):
        ax = fig.add_subplot(sub_gs[i // n_cols, i % n_cols])

        df_dir = df_speed[df_speed["direction"] == direction]
        n_trials = df_dir["trial_id"].nunique()

        if n_trials > 0:
            counts, _ = np.histogram(df_dir["time_from_moving_onset"], bins=bins)
            fr = counts / n_trials / BIN_WIDTH
            ax.plot(centers, fr)

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlim(TIME_RANGE)
        ax.set_title(f"{direction_label(direction)}°", fontsize=10)

        if i // n_cols == n_rows - 1:
            ax.set_xlabel("Time (s)")
        if i % n_cols == 0:
            ax.set_ylabel("FR")

    # Hide unused panels if any.
    for j in range(n, n_rows * n_cols):
        ax = fig.add_subplot(sub_gs[j // n_cols, j % n_cols])
        ax.axis("off")


# =====================
# Response plots
# =====================


def plot_signed_motion_baseline_response(ax, trial_unit, dir_sig_unit=None):
    """
    Direction response using signed moving - baseline FR.

    Shows trial dots, mean ± SEM, zero line, and optional direction-level
    significance markers from unit_direction_significance.csv.
    """
    if "moving_minus_baseline" not in trial_unit.columns:
        ax.set_title("Motion - baseline response")
        ax.text(0.5, 0.5, "moving_minus_baseline missing", ha="center", va="center")
        return

    df = trial_unit[["direction", "trial_id", "moving_minus_baseline"]].dropna().copy()

    if df.empty:
        ax.set_title("Motion - baseline response")
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
    rng = np.random.default_rng(42)

    for i, direction in enumerate(directions):
        vals = df.loc[df["direction"] == direction, "moving_minus_baseline"].values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, alpha=0.7)

    ax.errorbar(
        x,
        summary["mean"].values,
        yerr=summary["sem"].values,
        fmt="-o",
        capsize=4,
        linewidth=1.5,
    )

    ax.axhline(0, linewidth=1)

    # Optional direction-level significance markers.
    if dir_sig_unit is not None and not dir_sig_unit.empty:
        y_upper = summary["mean"].values + summary["sem"].values
        y_lower = summary["mean"].values - summary["sem"].values
        y_max = np.nanmax(y_upper) if len(y_upper) else 1
        y_min = np.nanmin(y_lower) if len(y_lower) else -1
        y_range = y_max - y_min if y_max > y_min else 1

        for i, direction in enumerate(directions):
            row = dir_sig_unit.loc[dir_sig_unit["direction"] == direction]
            if row.empty:
                continue
            r = row.iloc[0]

            if bool(r.get("is_direction_excited", False)):
                ax.text(i, y_max + 0.08 * y_range, "*", ha="center", va="bottom", fontsize=14)
            elif bool(r.get("is_direction_suppressed", False)):
                ax.text(i, y_min - 0.08 * y_range, "*", ha="center", va="top", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([direction_label(d) for d in directions])
    ax.set_xlabel("Direction")
    ax.set_ylabel("Moving - baseline FR")
    ax.set_title("Signed motion-baseline response\ntrials + mean ± SEM")


def plot_moving_fr_polar(ax, direction_unit):
    """Polar plot of pure moving firing rate."""
    if "moving_fr" not in direction_unit.columns:
        ax.set_title("Pure moving FR polar")
        return

    df = direction_unit[["direction", "moving_fr"]].dropna().sort_values("direction").copy()

    if df.empty:
        ax.set_title("Pure moving FR polar")
        return

    theta = np.deg2rad(df["direction"].astype(float).values)
    r = df["moving_fr"].clip(lower=0).values
    theta_c, r_c = close_polar(theta, r)

    ax.plot(theta_c, r_c, marker="o", linewidth=1.8)
    ax.fill(theta_c, r_c, alpha=0.15)
    format_polar_axis(ax, "Pure moving FR polar")


def plot_signed_components_polar(ax, direction_unit, dir_sig_unit=None):
    """
    Polar plot splitting motion-baseline response into:
      positive component = max(moving-baseline, 0)
      suppression strength = max(-(moving-baseline), 0)

    Suppression is plotted as a positive radius, but represents negative signed response.
    """
    required_cols = [
        "direction",
        "motion_baseline_positive",
        "motion_baseline_negative_strength",
    ]

    missing = [c for c in required_cols if c not in direction_unit.columns]
    if missing:
        ax.set_title("Motion-baseline components")
        ax.text(0.5, 0.5, "Missing: " + ", ".join(missing), ha="center", va="center")
        return

    df = direction_unit[required_cols].dropna(subset=["direction"]).sort_values("direction").copy()

    if df.empty:
        ax.set_title("Motion-baseline components")
        return

    theta = np.deg2rad(df["direction"].astype(float).values)
    pos_r = df["motion_baseline_positive"].clip(lower=0).values
    neg_r = df["motion_baseline_negative_strength"].clip(lower=0).values

    theta_c, pos_c = close_polar(theta, pos_r)
    _, neg_c = close_polar(theta, neg_r)

    ax.plot(theta_c, pos_c, marker="o", linewidth=1.8, label="excitation")
    ax.fill(theta_c, pos_c, alpha=0.12)

    ax.plot(theta_c, neg_c, marker="o", linewidth=1.8, label="suppression")
    ax.fill(theta_c, neg_c, alpha=0.12)

    # Optional significant direction markers.
    if dir_sig_unit is not None and not dir_sig_unit.empty:
        for _, sig_row in dir_sig_unit.iterrows():
            direction = sig_row["direction"]
            theta_sig = np.deg2rad(float(direction))
            match = df.loc[df["direction"] == direction]

            if match.empty:
                continue

            if bool(sig_row.get("is_direction_excited", False)):
                radius = float(match.iloc[0]["motion_baseline_positive"])
                ax.scatter([theta_sig], [radius], s=80, marker="*")

            if bool(sig_row.get("is_direction_suppressed", False)):
                radius = float(match.iloc[0]["motion_baseline_negative_strength"])
                ax.scatter([theta_sig], [radius], s=80, marker="*")

    format_polar_axis(ax, "Motion-baseline components\npositive vs suppression")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


def plot_baseline_static_moving_polar(ax, direction_unit):
    """Optional context polar: baseline/static/moving FR by direction."""
    cols = ["direction", "baseline_fr", "static_fr", "moving_fr"]
    available_cols = [c for c in cols if c in direction_unit.columns]

    if "direction" not in available_cols or len(available_cols) < 2:
        ax.set_title("Baseline/static/moving FR")
        return

    df = direction_unit[available_cols].dropna(subset=["direction"]).sort_values("direction").copy()

    if df.empty:
        ax.set_title("Baseline/static/moving FR")
        return

    theta = np.deg2rad(df["direction"].astype(float).values)
    theta_c, _ = close_polar(theta, np.zeros(len(theta)))

    for col in ["baseline_fr", "static_fr", "moving_fr"]:
        if col not in df.columns:
            continue
        r = df[col].clip(lower=0).values
        _, r_c = close_polar(theta, r)
        ax.plot(theta_c, r_c, marker="o", linewidth=1.5, label=col)
        ax.fill(theta_c, r_c, alpha=0.08)

    format_polar_axis(ax, "Baseline / static / moving FR")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


# =====================
# Summary text
# =====================


def add_summary_text(ax, unit_id, tuning_row, sig_row, dir_sig_unit=None):
    """Add unit-level summary text."""
    ax.axis("off")
    lines = [f"Unit {unit_id}"]

    if tuning_row is not None and len(tuning_row) > 0:
        r = tuning_row.iloc[0]
        lines += [
            "",
            f"Class: {r.get('response_class', 'NA')}",
            f"PD method: {r.get('pd_method', 'NA')}",
            f"Positive dirs: {format_value(r.get('n_positive_directions', np.nan), 0)}",
            f"Negative dirs: {format_value(r.get('n_negative_directions', np.nan), 0)}",
            "",
            f"Mean baseline FR: {format_value(r.get('mean_baseline_fr', np.nan), 2)}",
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

    if sig_row is not None and len(sig_row) > 0:
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

    if dir_sig_unit is not None and not dir_sig_unit.empty:
        n_sig = int(dir_sig_unit.get("is_direction_response_significant", pd.Series(False)).fillna(False).sum())
        n_exc = int(dir_sig_unit.get("is_direction_excited", pd.Series(False)).fillna(False).sum())
        n_sup = int(dir_sig_unit.get("is_direction_suppressed", pd.Series(False)).fillna(False).sum())
        lines += [
            "",
            f"Significant directions: {n_sig}",
            f"Excited directions: {n_exc}",
            f"Suppressed directions: {n_sup}",
        ]

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)


# =====================
# Unit PDF
# =====================


def plot_one_unit(
    unit_id,
    labeled,
    trial_summary,
    condition_summary,
    direction_summary,
    tuning_summary,
    sig,
    dir_sig,
    out_dir,
):
    """Create a multi-page PDF summary for one unit."""
    labeled_unit = labeled[labeled["unit_id"] == unit_id].copy()
    trial_unit = trial_summary[trial_summary["unit_id"] == unit_id].copy()
    condition_unit = condition_summary[condition_summary["unit_id"] == unit_id].copy()
    direction_unit = direction_summary[direction_summary["unit_id"] == unit_id].copy()

    if labeled_unit.empty or trial_unit.empty or direction_unit.empty:
        print(f"Skipping unit {unit_id}: missing labeled/trial/direction data.")
        return

    directions = sorted_unique_nonnull(direction_unit["direction"])
    if len(directions) == 0:
        directions = sorted_unique_nonnull(condition_unit["direction"])

    if len(directions) == 0:
        print(f"Skipping unit {unit_id}: no directions found.")
        return

    speed_col = detect_speed_column(labeled_unit)
    speeds = sorted_unique_nonnull(labeled_unit[speed_col]) if speed_col else [None]

    tuning_row = tuning_summary[tuning_summary["unit_id"] == unit_id]
    sig_row = None if sig is None else sig[sig["unit_id"] == unit_id]
    dir_sig_unit = None if dir_sig is None else dir_sig[dir_sig["unit_id"] == unit_id].copy()

    out_path = out_dir / f"unit_{unit_id}_single_screen_summary.pdf"

    with PdfPages(out_path) as pdf:

        # Page 1: raster pooled across speed.
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        plot_raster(ax1, labeled_unit, directions)
        fig1.suptitle(f"Unit {unit_id} raster", fontsize=14)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # One PSTH page per speed.
        for speed_value in speeds:
            fig2 = plt.figure(figsize=(18, 10))
            gs2 = fig2.add_gridspec(2, 5)

            ax_combined = fig2.add_subplot(gs2[:, 0])
            plot_psth_by_direction(
                ax_combined,
                labeled_unit,
                directions,
                speed_value=speed_value,
                speed_col=speed_col,
            )

            plot_psth_separated_by_speed(
                fig=fig2,
                gs_cell=gs2[:, 1:],
                labeled_unit=labeled_unit,
                directions=directions,
                speed_value=speed_value,
                speed_col=speed_col,
            )

            speed_title = "pooled across speeds" if speed_value is None else f"speed = {speed_value}"
            fig2.suptitle(f"Unit {unit_id} PSTH summary ({speed_title})", fontsize=14)
            fig2.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig2)
            plt.close(fig2)

        # Final page: response/tuning summaries.
        fig3 = plt.figure(figsize=(18, 11))
        gs3 = fig3.add_gridspec(2, 4)

        ax_signed = fig3.add_subplot(gs3[0, 0])
        plot_signed_motion_baseline_response(
            ax_signed,
            trial_unit,
            dir_sig_unit=dir_sig_unit,
        )

        ax_moving_polar = fig3.add_subplot(gs3[0, 1], projection="polar")
        plot_moving_fr_polar(ax_moving_polar, direction_unit)

        ax_components = fig3.add_subplot(gs3[0, 2], projection="polar")
        plot_signed_components_polar(
            ax_components,
            direction_unit,
            dir_sig_unit=dir_sig_unit,
        )

        ax_fr_context = fig3.add_subplot(gs3[0, 3], projection="polar")
        plot_baseline_static_moving_polar(ax_fr_context, direction_unit)

        ax_text = fig3.add_subplot(gs3[1, :])
        add_summary_text(
            ax_text,
            unit_id,
            tuning_row=tuning_row,
            sig_row=sig_row,
            dir_sig_unit=dir_sig_unit,
        )

        fig3.suptitle(f"Unit {unit_id} tuning / response summary", fontsize=14)
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Saved: {out_path}")


# =====================
# Main
# =====================


def main():
    print("===== Plot single-screen unit summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units_single_screen"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv")
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / "curated_units.csv")
    trial_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv")
    condition_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv")
    tuning_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv")

    direction_summary_path = ANALYSIS_OUTPUT_DIR / "unit_direction_summary.csv"
    direction_summary = safe_read_csv(direction_summary_path)

    if direction_summary is None:
        print("unit_direction_summary.csv not found. Building fallback from unit_condition_summary.csv.")
        direction_summary = build_direction_summary_fallback(condition_summary)

    sig = safe_read_csv(ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv")
    dir_sig = safe_read_csv(ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv")

    units_to_plot = get_units_to_plot(units, sig)

    print(f"Units to plot: {len(units_to_plot)}")
    print(units_to_plot)

    for unit_id in units_to_plot:
        plot_one_unit(
            unit_id=unit_id,
            labeled=labeled,
            trial_summary=trial_summary,
            condition_summary=condition_summary,
            direction_summary=direction_summary,
            tuning_summary=tuning_summary,
            sig=sig,
            dir_sig=dir_sig,
            out_dir=plot_dir,
        )

    print("\n===== Saved unit plots to =====")
    print(plot_dir)


if __name__ == "__main__":
    main()
