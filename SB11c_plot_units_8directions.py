# SB11_plot_units_single_screen_speed_split.py
#
# Unit-level plots for single-screen 8-direction / 2-speed stimulus.
#
# This version assumes ALL summaries are speed-specific:
#   unit_trial_summary.csv              unit × trial, with speed column
#   unit_condition_summary.csv          unit × speed × direction
#   unit_tuning_summary.csv             unit × speed
#   unit_significance_summary.csv       unit × speed
#   unit_direction_significance.csv     unit × speed × direction
#
# For each unit, the PDF contains a complete set of pages for each speed:
#   1. raster by direction for that speed
#   2. PSTH by direction for that speed
#   3. signed response + polar plots + speed-specific summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR

# Things need to be changed:
# def format_polar_axis(ax, title):
# def speed_to_numeric
# def get_speed_order_available
# def plot_cross_speed_summary_page
# def plot_speed_raster_psth_page


# =====================
# Plot settings
# =====================

PLOT_MODE = "all"  # "all" or "significant"

SIGNIFICANCE_COLUMNS = [
    "is_motion_baseline_responsive",
    "is_motion_baseline_suppressed",
    "is_direction_tuned_motion_baseline",
]

TIME_RANGE = (-2.0, 4.0)
BIN_WIDTH = 0.1
PSTH_DIRECTION_N_COLS = 4


# =====================
# Helpers
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


# def format_polar_axis(ax, title):
#     ax.set_theta_zero_location("E")
#     ax.set_theta_direction(-1)
#     ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
#     ax.set_xticklabels(["0", "45", "90", "135", "180", "225", "270", "315"])
#     ax.set_title(title)


def format_polar_axis(ax, title):
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"])
    ax.set_title(title)


def speed_to_numeric(speed):
    if pd.isna(speed):
        return np.nan

    try:
        return float(speed)
    except Exception:
        pass

    s = str(speed)
    s = s.replace("speed_", "").replace("_dps", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def get_speed_order_available(df):
    speeds = pd.Series(df["speed"]).dropna().astype(str).unique().tolist()
    return sorted(speeds, key=lambda x: speed_to_numeric(x))


def plot_cross_speed_summary_page(
    pdf,
    unit_id,
    trial_unit,
    condition_unit,
    tuning_summary,
    sig,
    dir_sig,
    speed_effect_unit=None,
):
    speeds = get_speed_order_available(condition_unit)

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(
        3,
        len(speeds),
        height_ratios=[1.0, 1.05, 1.25],
        hspace=0.45,
        wspace=0.45,
    )

    for col, speed in enumerate(speeds):
        labeled_dummy = None

        trial_speed = trial_unit[
            trial_unit["speed"].astype(str) == str(speed)
        ].copy()

        condition_speed = condition_unit[
            condition_unit["speed"].astype(str) == str(speed)
        ].copy()

        tuning_row = filter_unit_speed(tuning_summary, unit_id, speed)

        sig_row = (
            filter_unit_speed(sig, unit_id, speed)
            if sig is not None
            else None
        )

        dir_sig_speed = (
            filter_unit_speed(dir_sig, unit_id, speed)
            if dir_sig is not None
            else None
        )

        ax_resp = fig.add_subplot(gs[0, col])
        plot_signed_motion_baseline_response(
            ax_resp,
            trial_speed,
            dir_sig_speed,
        )
        ax_resp.set_title(f"{speed}\nresponse")

        ax_polar = fig.add_subplot(gs[1, col], projection="polar")
        plot_baseline_static_moving_polar(
            ax_polar,
            condition_speed,
        )
        ax_polar.set_title(f"{speed}\nbaseline/static/moving")

        ax_text = fig.add_subplot(gs[2, col])
        add_summary_text(
            ax_text,
            unit_id,
            speed,
            tuning_row,
            sig_row,
            dir_sig_speed,
        )

    fig.suptitle(
        f"Unit {unit_id} cross-speed comparison",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def plot_speed_raster_psth_page(
    pdf,
    unit_id,
    speed,
    labeled_speed,
    directions,
):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.1, 1.2],
        hspace=0.4,
    )

    ax_raster = fig.add_subplot(gs[0, 0])
    plot_raster_by_speed(
        ax_raster,
        labeled_speed,
        directions,
        speed,
    )

    ax_psth = fig.add_subplot(gs[1, 0])
    plot_psth_combined(
        ax_psth,
        labeled_speed,
        directions,
        speed,
    )

    fig.suptitle(
        f"Unit {unit_id} raster + PSTH | speed = {speed}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def ensure_speed_column(df, table_name):
    if "speed" not in df.columns:
        raise ValueError(f"{table_name} must contain a 'speed' column for speed-specific plotting.")


def get_units_to_plot(units, sig):
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()

    if PLOT_MODE == "significant":
        if sig is None:
            raise FileNotFoundError("unit_significance_summary.csv not found.")

        mask = np.zeros(len(sig), dtype=bool)
        for col in SIGNIFICANCE_COLUMNS:
            if col in sig.columns:
                mask |= sig[col].fillna(False).astype(bool)

        return sorted_unique_nonnull(sig.loc[mask, "unit_id"])

    raise ValueError("PLOT_MODE must be 'all' or 'significant'.")


def filter_unit_speed(df, unit_id, speed):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[(df["unit_id"] == unit_id) & (df["speed"] == speed)].copy()


# =====================
# Raster / PSTH
# =====================


def plot_raster_by_speed(ax, labeled_unit_speed, directions, speed):
    """Raster plot grouped by direction for one speed."""
    y = 0
    yticks = []
    ylabels = []

    for direction in directions:
        df_dir = labeled_unit_speed[labeled_unit_speed["direction"] == direction]

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
    ax.set_title(f"Raster aligned to moving onset\nspeed = {speed}")


def plot_psth_combined(ax, labeled_unit_speed, directions, speed):
    """Combined PSTH for one speed: one line per direction."""
    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    for direction in directions:
        df_dir = labeled_unit_speed[labeled_unit_speed["direction"] == direction]
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
    ax.set_title(f"PSTH by direction\nspeed = {speed}")
    ax.legend(title="Direction", fontsize=7, ncol=2)


def plot_psth_separated(fig, gs_cell, labeled_unit_speed, directions):
    """Small multiples: one PSTH panel per direction for one speed."""
    n = len(directions)
    n_cols = min(PSTH_DIRECTION_N_COLS, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    sub_gs = gs_cell.subgridspec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    for i, direction in enumerate(directions):
        ax = fig.add_subplot(sub_gs[i // n_cols, i % n_cols])
        df_dir = labeled_unit_speed[labeled_unit_speed["direction"] == direction]
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

    for j in range(n, n_rows * n_cols):
        ax = fig.add_subplot(sub_gs[j // n_cols, j % n_cols])
        ax.axis("off")


# =====================
# Response / polar plots
# =====================


def plot_signed_motion_baseline_response(ax, trial_unit_speed, dir_sig_unit_speed=None):
    """Signed moving-baseline response by direction for one speed."""
    df = trial_unit_speed[["direction", "trial_id", "moving_minus_baseline"]].dropna().copy()

    if df.empty:
        ax.set_title("Signed motion-baseline response")
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

    if dir_sig_unit_speed is not None and not dir_sig_unit_speed.empty:
        y_upper = summary["mean"].values + summary["sem"].values
        y_lower = summary["mean"].values - summary["sem"].values
        y_max = np.nanmax(y_upper) if len(y_upper) else 1
        y_min = np.nanmin(y_lower) if len(y_lower) else -1
        y_range = y_max - y_min if y_max > y_min else 1

        for i, direction in enumerate(directions):
            row = dir_sig_unit_speed.loc[dir_sig_unit_speed["direction"] == direction]
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


def plot_moving_fr_polar(ax, condition_unit_speed):
    """Pure moving firing-rate polar plot for one speed."""
    df = condition_unit_speed[["direction", "moving_fr"]].dropna().sort_values("direction").copy()

    if df.empty:
        ax.set_title("Pure moving FR polar")
        return

    theta = np.deg2rad(df["direction"].astype(float).values)
    r = df["moving_fr"].clip(lower=0).values
    theta_c, r_c = close_polar(theta, r)

    ax.plot(theta_c, r_c, marker="o", linewidth=1.8)
    ax.fill(theta_c, r_c, alpha=0.15)
    format_polar_axis(ax, "Pure moving FR polar")


def plot_signed_components_polar(ax, condition_unit_speed, dir_sig_unit_speed=None):
    """
    Positive/suppression split polar for one speed.

    excitation radius   = max(moving-baseline, 0)
    suppression radius  = max(-(moving-baseline), 0)
    """
    df = condition_unit_speed[
        ["direction", "motion_baseline_positive", "motion_baseline_negative_strength"]
    ].dropna(subset=["direction"]).sort_values("direction").copy()

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

    if dir_sig_unit_speed is not None and not dir_sig_unit_speed.empty:
        for _, sig_row in dir_sig_unit_speed.iterrows():
            direction = sig_row["direction"]
            match = df.loc[df["direction"] == direction]
            if match.empty:
                continue
            theta_sig = np.deg2rad(float(direction))

            if bool(sig_row.get("is_direction_excited", False)):
                radius = float(match.iloc[0]["motion_baseline_positive"])
                ax.scatter([theta_sig], [radius], s=80, marker="*")

            if bool(sig_row.get("is_direction_suppressed", False)):
                radius = float(match.iloc[0]["motion_baseline_negative_strength"])
                ax.scatter([theta_sig], [radius], s=80, marker="*")

    format_polar_axis(ax, "Motion-baseline components\npositive vs suppression")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


def plot_baseline_static_moving_polar(ax, condition_unit_speed):
    """Baseline/static/moving FR context polar for one speed."""
    df = condition_unit_speed[
        ["direction", "baseline_fr", "static_fr", "moving_fr"]
    ].dropna(subset=["direction"]).sort_values("direction").copy()

    if df.empty:
        ax.set_title("Baseline/static/moving FR")
        return

    theta = np.deg2rad(df["direction"].astype(float).values)
    theta_c, _ = close_polar(theta, np.zeros(len(theta)))

    for col in ["baseline_fr", "static_fr", "moving_fr"]:
        r = df[col].clip(lower=0).values
        _, r_c = close_polar(theta, r)
        ax.plot(theta_c, r_c, marker="o", linewidth=1.5, label=col)
        ax.fill(theta_c, r_c, alpha=0.08)

    format_polar_axis(ax, "Baseline / static / moving FR")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


# =====================
# Summary text
# =====================


def add_summary_text(ax, unit_id, speed_value, tuning_row, sig_row, dir_sig_speed):
    ax.axis("off")
    lines = [f"Unit {unit_id}", f"Speed: {speed_value}"]

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

    if dir_sig_speed is not None and not dir_sig_speed.empty:
        n_sig = int(dir_sig_speed.get("is_direction_response_significant", pd.Series(False)).fillna(False).sum())
        n_exc = int(dir_sig_speed.get("is_direction_excited", pd.Series(False)).fillna(False).sum())
        n_sup = int(dir_sig_speed.get("is_direction_suppressed", pd.Series(False)).fillna(False).sum())

        lines += [
            "",
            f"Significant directions: {n_sig}",
            f"Excited directions: {n_exc}",
            f"Suppressed directions: {n_sup}",
            "",
            "Direction p/q, moving-baseline:",
        ]

        dtab = dir_sig_speed.copy()
        dtab["direction"] = dtab["direction"].astype(float)
        dtab = dtab.sort_values("direction")

        for _, row in dtab.iterrows():
            direction = row.get("direction", np.nan)
            mean_resp = row.get("mean_moving_minus_baseline", np.nan)
            p_val = row.get("p_motion_baseline_two_sided", np.nan)
            q_val = row.get("q_motion_baseline_direction", np.nan)

            sig_label = ""
            if bool(row.get("is_direction_excited", False)):
                sig_label = " exc"
            elif bool(row.get("is_direction_suppressed", False)):
                sig_label = " sup"

            lines.append(
                f"{format_value(direction, 0)}°: "
                f"Δ={format_value(mean_resp, 2)}, "
                f"p={format_value(p_val, 4)}, "
                f"q={format_value(q_val, 4)}"
                f"{sig_label}"
            )

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8)


# =====================
# Unit PDF
# =====================


# def plot_one_unit(unit_id, labeled, trial_summary, condition_summary, tuning_summary, sig, dir_sig, out_dir):
#     labeled_unit = labeled[labeled["unit_id"] == unit_id].copy()
#     trial_unit = trial_summary[trial_summary["unit_id"] == unit_id].copy()
#     condition_unit = condition_summary[condition_summary["unit_id"] == unit_id].copy()

#     if labeled_unit.empty or trial_unit.empty or condition_unit.empty:
#         print(f"Skipping unit {unit_id}: missing labeled/trial/condition data.")
#         return

#     speeds = sorted_unique_nonnull(condition_unit["speed"])
#     if len(speeds) == 0:
#         print(f"Skipping unit {unit_id}: no speed values found.")
#         return

#     out_path = out_dir / f"unit_{unit_id}_single_screen_speed_split_summary.pdf"

#     with PdfPages(out_path) as pdf:
#         for speed in speeds:
#             labeled_speed = labeled_unit[labeled_unit["speed"] == speed].copy() if "speed" in labeled_unit.columns else labeled_unit.copy()
#             trial_speed = trial_unit[trial_unit["speed"] == speed].copy()
#             condition_speed = condition_unit[condition_unit["speed"] == speed].copy()
#             tuning_row = filter_unit_speed(tuning_summary, unit_id, speed)
#             sig_row = filter_unit_speed(sig, unit_id, speed) if sig is not None else None
#             dir_sig_speed = filter_unit_speed(dir_sig, unit_id, speed) if dir_sig is not None else None

#             directions = sorted_unique_nonnull(condition_speed["direction"])
#             if len(directions) == 0:
#                 continue

#             # Page A: raster for this speed.
#             fig1, ax1 = plt.subplots(figsize=(12, 8))
#             plot_raster_by_speed(ax1, labeled_speed, directions, speed)
#             fig1.suptitle(f"Unit {unit_id} raster | speed = {speed}", fontsize=14)
#             fig1.tight_layout(rect=[0, 0, 1, 0.96])
#             pdf.savefig(fig1)
#             plt.close(fig1)

#             # Page B: PSTH for this speed.
#             fig2 = plt.figure(figsize=(18, 10))
#             gs2 = fig2.add_gridspec(2, 5)
#             ax_combined = fig2.add_subplot(gs2[:, 0])
#             plot_psth_combined(ax_combined, labeled_speed, directions, speed)
#             plot_psth_separated(fig2, gs2[:, 1:], labeled_speed, directions)
#             fig2.suptitle(f"Unit {unit_id} PSTH | speed = {speed}", fontsize=14)
#             fig2.tight_layout(rect=[0, 0, 1, 0.96])
#             pdf.savefig(fig2)
#             plt.close(fig2)

#             # Page C: response and polar summaries for this speed.
#             fig3 = plt.figure(figsize=(19, 11))
#             gs3 = fig3.add_gridspec(
#                 2, 4,
#                 width_ratios=[1.2, 1.1, 1.1, 1.1],
#                 height_ratios=[1.0, 1.05],
#                 wspace=0.45,
#                 hspace=0.35,
#             )

#             ax_signed = fig3.add_subplot(gs3[0, 0])
#             plot_signed_motion_baseline_response(ax_signed, trial_speed, dir_sig_speed)

#             ax_moving_polar = fig3.add_subplot(gs3[0, 1], projection="polar")
#             plot_moving_fr_polar(ax_moving_polar, condition_speed)

#             ax_components = fig3.add_subplot(gs3[0, 2], projection="polar")
#             plot_signed_components_polar(ax_components, condition_speed, dir_sig_speed)

#             ax_context = fig3.add_subplot(gs3[0, 3], projection="polar")
#             plot_baseline_static_moving_polar(ax_context, condition_speed)

#             ax_text = fig3.add_subplot(gs3[1, 0:2])
#             add_summary_text(ax_text, unit_id, speed, tuning_row, sig_row, None)

#             ax_dir_table = fig3.add_subplot(gs3[1, 2:4])
#             add_direction_pq_table(ax_dir_table, dir_sig_speed)

#             fig3.suptitle(f"Unit {unit_id} response summary | speed = {speed}", fontsize=14)
#             fig3.tight_layout(rect=[0, 0, 1, 0.96])
#             pdf.savefig(fig3)
#             plt.close(fig3)

#     print(f"Saved: {out_path}")


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

    if labeled_unit.empty or trial_unit.empty or condition_unit.empty:
        print(f"Skipping unit {unit_id}: missing labeled/trial/condition data.")
        return

    speeds = get_speed_order_available(condition_unit)

    if len(speeds) == 0:
        print(f"Skipping unit {unit_id}: no speed values found.")
        return

    out_path = out_dir / f"unit_{unit_id}_cross_speed_summary.pdf"

    with PdfPages(out_path) as pdf:

        # Page 1: cross-speed comparison.
        plot_cross_speed_summary_page(
            pdf=pdf,
            unit_id=unit_id,
            trial_unit=trial_unit,
            condition_unit=condition_unit,
            tuning_summary=tuning_summary,
            sig=sig,
            dir_sig=dir_sig,
        )

        # Pages 2+: one raster/PSTH page per speed.
        for speed in speeds:
            labeled_speed = labeled_unit[
                labeled_unit["speed"].astype(str) == str(speed)
            ].copy()

            condition_speed = condition_unit[
                condition_unit["speed"].astype(str) == str(speed)
            ].copy()

            directions = sorted_unique_nonnull(condition_speed["direction"])

            if len(directions) == 0:
                continue

            plot_speed_raster_psth_page(
                pdf=pdf,
                unit_id=unit_id,
                speed=speed,
                labeled_speed=labeled_speed,
                directions=directions,
            )

    print(f"Saved: {out_path}")


def add_direction_pq_table(ax, dir_sig_speed):
    """
    Show direction-level moving-baseline p/q values as a compact table.

    Each row:
        direction
        mean moving-baseline response
        two-sided p value
        FDR q value
        label
    """
    ax.axis("off")

    if dir_sig_speed is None or dir_sig_speed.empty:
        ax.text(
            0.02,
            0.95,
            "Direction p/q\nNo direction-level significance data",
            va="top",
            ha="left",
            fontsize=9,
        )
        return

    dtab = dir_sig_speed.copy()
    dtab["direction"] = dtab["direction"].astype(float)
    dtab = dtab.sort_values("direction")

    rows = []

    for _, row in dtab.iterrows():
        direction = row.get("direction", np.nan)
        mean_resp = row.get("mean_moving_minus_baseline", np.nan)
        p_val = row.get("p_motion_baseline_two_sided", np.nan)
        q_val = row.get("q_motion_baseline_direction", np.nan)

        label = ""
        if bool(row.get("is_direction_excited", False)):
            label = "exc"
        elif bool(row.get("is_direction_suppressed", False)):
            label = "sup"

        rows.append([
            f"{format_value(direction, 0)}°",
            format_value(mean_resp, 2),
            format_value(p_val, 4),
            format_value(q_val, 4),
            label,
        ])

    col_labels = ["Dir", "ΔMB", "p", "q", "Sig"]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.86],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    # Make header slightly clearer.
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")

    ax.set_title("Direction-level moving-baseline test", fontsize=10)


# =====================
# Main
# =====================


def main():
    print("===== Plot speed-specific single-screen unit summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units_single_screen_speed_split"
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

    for df in [labeled, trial_summary, condition_summary, tuning_summary, sig, dir_sig]:
        if df is not None and "speed" in df.columns:
            df["speed"] = df["speed"].astype(str).str.strip()

    for name, df in [
        ("labeled_spikes.csv", labeled),
        ("unit_trial_summary.csv", trial_summary),
        ("unit_condition_summary.csv", condition_summary),
        ("unit_tuning_summary.csv", tuning_summary),
    ]:
        ensure_speed_column(df, name)

    if sig is not None:
        ensure_speed_column(sig, "unit_significance_summary.csv")
    if dir_sig is not None:
        ensure_speed_column(dir_sig, "unit_direction_significance.csv")

    units_to_plot = get_units_to_plot(units, sig)
    print(f"Units to plot: {len(units_to_plot)}")
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
