# 06_plot_units.py

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
    "is_motion_responsive",
    "is_motion_suppressed",
    "is_direction_tuned",
]

TIME_RANGE = (-5.0, 6.0)
BIN_WIDTH = 0.1


def sem(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def get_units_to_plot(units, sig):
    """Choose all units or only significant units."""
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()

    if PLOT_MODE == "significant":
        if sig is None:
            raise FileNotFoundError(
                "unit_significance_summary.csv not found, "
                "but PLOT_MODE='significant'. Run 05b first."
            )

        mask = np.zeros(len(sig), dtype=bool)

        for col in SIGNIFICANCE_COLUMNS:
            if col in sig.columns:
                mask |= sig[col].fillna(False).astype(bool)

        return sig.loc[mask, "unit_id"].tolist()

    raise ValueError("PLOT_MODE must be 'all' or 'significant'.")


def plot_raster(ax, labeled_unit, directions):
    """Raster plot grouped by direction, using thin vertical lines."""
    y = 0
    yticks = []
    ylabels = []

    for direction in directions:
        df_dir = labeled_unit[labeled_unit["direction"] == direction]

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
            ylabels.append(str(direction))

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Direction / trials")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title("Raster aligned to moving onset")


def plot_psth(ax, labeled_unit, directions):
    """PSTH by direction."""
    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    for direction in directions:
        df_dir = labeled_unit[labeled_unit["direction"] == direction]
        n_trials = df_dir["trial_id"].nunique()

        if n_trials == 0:
            continue

        counts, _ = np.histogram(df_dir["time_from_moving_onset"], bins=bins)
        fr = counts / n_trials / BIN_WIDTH

        ax.plot(centers, fr, label=str(direction))

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(TIME_RANGE)
    ax.set_xlabel("Time from moving onset (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    ax.set_title("PSTH by direction")
    ax.legend(title="Direction", fontsize=8)


def plot_psth_separated(fig, gs_cell, labeled_unit, directions):
    """Small multiples: one PSTH panel per direction."""
    sub_gs = gs_cell.subgridspec(2, 2, hspace=0.45, wspace=0.35)

    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    axes = []

    for i, direction in enumerate(directions):
        ax = fig.add_subplot(sub_gs[i // 2, i % 2])
        axes.append(ax)

        df_dir = labeled_unit[labeled_unit["direction"] == direction]
        n_trials = df_dir["trial_id"].nunique()

        if n_trials > 0:
            counts, _ = np.histogram(df_dir["time_from_moving_onset"], bins=bins)
            fr = counts / n_trials / BIN_WIDTH
            ax.plot(centers, fr)

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlim(TIME_RANGE)
        ax.set_title(f"{direction}°", fontsize=10)

        if i // 2 == 1:
            ax.set_xlabel("Time (s)")
        if i % 2 == 0:
            ax.set_ylabel("FR")

    return axes


def sem(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def plot_condition_response(ax, trial_unit):
    """
    Plot moving-static response by direction with:
    - individual trial dots
    - mean ± SEM
    """
    df = trial_unit[["direction", "trial_id", "moving_minus_static"]].dropna().copy()

    if df.empty:
        ax.set_title("Motion-specific response")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    directions = sorted(df["direction"].unique())
    summary = (
        df.groupby("direction", dropna=False)["moving_minus_static"]
        .agg(mean="mean", sem=sem)
        .reindex(directions)
        .reset_index()
    )

    x = np.arange(len(directions))

    # plot individual trial dots
    rng = np.random.default_rng(42)
    for i, direction in enumerate(directions):
        vals = df.loc[df["direction"] == direction, "moving_minus_static"].values
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=25,
            alpha=0.8,
        )

    # plot mean ± SEM
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
    ax.set_xticklabels([str(d) for d in directions])
    ax.set_xlabel("Direction")
    ax.set_ylabel("Moving - static FR")
    ax.set_title("Motion-specific response\n(trials + mean ± SEM)")


def plot_condition_response_polar(ax, trial_unit, condition_unit):
    """
    Polar map showing:
    - static FR as translucent background
    - moving FR as translucent background
    - moving-static response as trial dots + mean ± SEM
    """
    cond = (
        condition_unit[
            ["direction", "static_fr", "moving_fr", "moving_minus_static"]
        ]
        .dropna(subset=["direction"])
        .groupby("direction", as_index=False)
        .mean()
        .sort_values("direction")
    )

    trial = (
        trial_unit[
            ["direction", "trial_id", "moving_minus_static"]
        ]
        .dropna(subset=["direction", "moving_minus_static"])
        .copy()
    )

    if cond.empty:
        ax.set_title("Motion-specific polar response")
        return

    directions = cond["direction"].astype(float).values
    theta = np.deg2rad(directions)

    # Background FR layers: static and moving should be non-negative.
    static_r = cond["static_fr"].clip(lower=0).values
    moving_r = cond["moving_fr"].clip(lower=0).values

    theta_closed = np.r_[theta, theta[0]]
    static_closed = np.r_[static_r, static_r[0]]
    moving_closed = np.r_[moving_r, moving_r[0]]

    ax.fill(theta_closed, static_closed, alpha=0.15, label="static FR")
    ax.fill(theta_closed, moving_closed, alpha=0.15, label="moving FR")

    # Main layer: moving - static, with trial dots + mean ± SEM.
    summary = (
        trial.groupby("direction")["moving_minus_static"]
        .agg(mean="mean", sem=sem)
        .reindex(directions)
        .reset_index()
    )

    # Polar radius cannot show negative values naturally.
    # For plotting radius, clip negative motion-specific responses to 0.
    mean_r = summary["mean"].clip(lower=0).values
    sem_r = summary["sem"].values

    mean_closed = np.r_[mean_r, mean_r[0]]
    ax.plot(theta_closed, mean_closed, marker="o", linewidth=1.8, label="moving - static")

    ax.errorbar(
        theta,
        mean_r,
        yerr=sem_r,
        fmt="none",
        capsize=3,
        linewidth=1,
    )

    # Trial dots with slight angular jitter
    rng = np.random.default_rng(42)

    for direction in directions:
        vals = trial.loc[
            trial["direction"].astype(float) == direction,
            "moving_minus_static",
        ].values

        vals = np.clip(vals, 0, None)

        if len(vals) == 0:
            continue

        theta_jitter = np.deg2rad(
            direction + rng.uniform(-3, 3, size=len(vals))
        )

        ax.scatter(
            theta_jitter,
            vals,
            s=18,
            alpha=0.7,
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"])
    ax.set_title("Static / moving / motion-specific response")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)


def add_summary_text(ax, unit_id, tuning_row, sig_row):
    """Add unit-level summary text."""
    ax.axis("off")

    lines = [f"Unit {unit_id}"]

    if tuning_row is not None and len(tuning_row) > 0:
        r = tuning_row.iloc[0]
        lines += [
            "",
            f"Mean static FR: {r.get('mean_static_fr', np.nan):.2f}",
            f"Mean moving FR: {r.get('mean_moving_fr', np.nan):.2f}",
            f"Preferred direction: {r.get('preferred_direction', np.nan)}",
            f"Vector strength: {r.get('vector_strength', np.nan):.3f}",
            f"DSI: {r.get('dsi', np.nan):.3f}",
        ]

    if sig_row is not None and len(sig_row) > 0:
        r = sig_row.iloc[0]
        lines += [
            "",
            f"Motion responsive: {bool(r.get('is_motion_responsive', False))}",
            f"Motion suppressed: {bool(r.get('is_motion_suppressed', False))}",
            f"Direction tuned: {bool(r.get('is_direction_tuned', False))}",
            "",
            f"q motion-specific: {r.get('q_motion_specific', np.nan):.4f}",
            f"q direction tuning: {r.get('q_direction_tuning', np.nan):.4f}",
        ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
    )


def plot_one_unit(unit_id, labeled, trial_summary, condition_summary, tuning_summary, sig, out_dir):
    """Create a two-page PDF summary for one unit."""
    labeled_unit = labeled[labeled["unit_id"] == unit_id]
    trial_unit = trial_summary[trial_summary["unit_id"] == unit_id]
    condition_unit = condition_summary[condition_summary["unit_id"] == unit_id]

    if labeled_unit.empty or condition_unit.empty:
        print(f"Skipping unit {unit_id}: no labeled spikes or condition summary.")
        return

    directions = sorted(condition_unit["direction"].dropna().unique())

    tuning_row = tuning_summary[tuning_summary["unit_id"] == unit_id]
    sig_row = None if sig is None else sig[sig["unit_id"] == unit_id]

    out_path = out_dir / f"unit_{unit_id}_summary.pdf"

    with PdfPages(out_path) as pdf:

        # Page 1: raster only
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        plot_raster(ax1, labeled_unit, directions)
        fig1.suptitle(f"Unit {unit_id} raster", fontsize=14)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: PSTH + response summaries
        fig2 = plt.figure(figsize=(16, 10))
        gs = fig2.add_gridspec(2, 3)

        ax_psth_combined = fig2.add_subplot(gs[0, 0])
        plot_psth(ax_psth_combined, labeled_unit, directions)

        plot_psth_separated(
            fig=fig2,
            gs_cell=gs[0, 1:],
            labeled_unit=labeled_unit,
            directions=directions,
        )

        ax_response = fig2.add_subplot(gs[1, 0])
        plot_condition_response(ax_response, trial_unit)

        ax_polar = fig2.add_subplot(gs[1, 1], projection="polar")
        plot_condition_response_polar(ax_polar, trial_unit, condition_unit)

        ax_text = fig2.add_subplot(gs[1, 2])
        add_summary_text(ax_text, unit_id, tuning_row, sig_row)

        fig2.suptitle(f"Unit {unit_id} response summary", fontsize=14)
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)


def plot_polar_tuning(ax, condition_unit):
    """
    Polar tuning plot using condition-mean moving-static response.

    Note:
    Polar radius cannot display negative values nicely, so we clip at 0.
    Suppressed responses should still be interpreted mainly from the tuning curve
    and summary text.
    """
    df = condition_unit[["direction", "moving_minus_static"]].dropna().copy()

    if df.empty:
        ax.set_title("Polar tuning")
        return

    df = (
        df.groupby("direction", dropna=False)["moving_minus_static"]
        .mean()
        .reset_index()
        .sort_values("direction")
    )

    # clip negative values for polar radius
    df["response_for_polar"] = df["moving_minus_static"].clip(lower=0)

    if len(df) == 0:
        ax.set_title("Polar tuning")
        return

    theta = np.deg2rad(df["direction"].values)
    r = df["response_for_polar"].values

    # close the curve
    theta = np.r_[theta, theta[0]]
    r = np.r_[r, r[0]]

    ax.plot(theta, r, marker="o")
    ax.fill(theta, r, alpha=0.2)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "180", "270"])
    ax.set_title("Polar tuning\n(max(0, moving-static))")


def main():
    print("===== Plot unit summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv")
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / "curated_units.csv")
    trial_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv")
    condition_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv")
    tuning_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv")

    sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() else None

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
            out_dir=plot_dir,
        )

    print("\n===== Saved unit plots to =====")
    print(plot_dir)


if __name__ == "__main__":
    main()