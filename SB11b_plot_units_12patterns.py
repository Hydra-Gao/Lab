# SB06b_plot_vbc_units.py

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

# Used only when PLOT_MODE == "significant".
# This works with unit_pattern_significance.csv from SB05d.
SIGNIFICANCE_COLUMNS = [
    "is_motion_specific",
    "is_motion_responsive",
    "is_motion_suppressed",
    "is_responsive_two_sided_gate",
    "is_suppressed_two_sided_gate",
]

# VbC three-screen timing:
# static = -3 to 0 sec relative to moving onset
# moving = 0 to 5 sec
TIME_RANGE = (-3.0, 5.0)
BIN_WIDTH = 0.1

ALPHA = 0.05


# =====================
# Pattern order / grouping
# =====================

PATTERN_ORDER = [
    # vertical axis
    "VAr",
    "VAl",

    # horizontal axis / corner
    "HA_leftcorner_clockwise",
    "HA_leftcorner_anticlockwise",
    "HA_rightcorner_clockwise",
    "HA_rightcorner_anticlockwise",

    # vertical translation
    "Ascent",
    "Descent",

    # expansion / contraction
    "EXPANSION_l",
    "EXPANSION_r",
    "CONTRACTION_left",
    "CONTRACTION_right",
]

PATTERN_LABELS_SHORT = {
    "VAr": "VAr",
    "VAl": "VAl",

    "HA_leftcorner_clockwise": "HA L-cw",
    "HA_leftcorner_anticlockwise": "HA L-acw",
    "HA_rightcorner_clockwise": "HA R-cw",
    "HA_rightcorner_anticlockwise": "HA R-acw",

    "Ascent": "Ascent",
    "Descent": "Descent",

    "EXPANSION_l": "Exp L",
    "EXPANSION_r": "Exp R",
    "CONTRACTION_left": "Con L",
    "CONTRACTION_right": "Con R",
}

PATTERN_GROUPS = {
    "Vertical axis": [
        "VAr",
        "VAl",
    ],
    "HA": [
        "HA_leftcorner_clockwise",
        "HA_leftcorner_anticlockwise",
        "HA_rightcorner_clockwise",
        "HA_rightcorner_anticlockwise",
    ],
    "Ascent / Descent": [
        "Ascent",
        "Descent",
    ],
    "Expansion / Contraction": [
        "EXPANSION_l",
        "EXPANSION_r",
        "CONTRACTION_left",
        "CONTRACTION_right",
    ],
}


# =====================
# Small utilities
# =====================

def require_columns(df, cols, name):
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def sem(x):
    """Standard error of the mean."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def safe_float(value):
    """Convert value to float, returning nan if conversion fails."""
    try:
        return float(value)
    except Exception:
        return np.nan


def fmt_p(value):
    """Format p/q values compactly."""
    value = safe_float(value)

    if np.isnan(value):
        return "nan"

    if value < 0.0001:
        return "<1e-4"

    return f"{value:.4f}"


def short_pattern_name(pattern):
    return PATTERN_LABELS_SHORT.get(pattern, str(pattern))


def available_pattern_order(df):
    """
    Return PATTERN_ORDER plus any unexpected patterns appended at the end.
    This prevents silent data loss if a pattern name changes slightly.
    """
    observed = [p for p in df["pattern"].dropna().unique().tolist()]

    ordered = [p for p in PATTERN_ORDER if p in observed]
    extras = sorted([p for p in observed if p not in PATTERN_ORDER])

    return ordered + extras


def add_epoch_background(ax):
    """Shade static and moving windows."""
    ax.axvspan(TIME_RANGE[0], 0, alpha=0.08)
    ax.axvspan(0, TIME_RANGE[1], alpha=0.08)
    ax.axvline(0, linestyle="--", linewidth=1)


# =====================
# Unit selection
# =====================

def get_units_to_plot(units, sig):
    """Choose all units or only units with at least one significant pattern."""
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()

    if PLOT_MODE == "significant":
        if sig is None:
            raise FileNotFoundError(
                "unit_pattern_significance.csv not found, "
                "but PLOT_MODE='significant'. Run SB05d first."
            )

        mask = np.zeros(len(sig), dtype=bool)

        for col in SIGNIFICANCE_COLUMNS:
            if col in sig.columns:
                mask |= sig[col].fillna(False).astype(bool).to_numpy()

        return sorted(sig.loc[mask, "unit_id"].dropna().unique().tolist())

    raise ValueError("PLOT_MODE must be 'all' or 'significant'.")


# =====================
# Page 1: grouped raster
# =====================

def plot_grouped_raster(ax, labeled_unit, pattern_list, title):
    """
    Raster plot for one pattern group.

    Rows:
        trial rows, grouped by pattern.

    x:
        time from moving onset.
    """
    y = 0
    yticks = []
    ylabels = []

    for pattern in pattern_list:
        df_pat = labeled_unit[labeled_unit["pattern"] == pattern].copy()

        trial_ids = sorted(df_pat["trial_id"].dropna().unique())

        if len(trial_ids) == 0:
            continue

        pattern_start_y = y

        for trial_id in trial_ids:
            df_trial = df_pat[df_pat["trial_id"] == trial_id]
            spike_times = df_trial["time_from_moving_onset"].to_numpy(dtype=float)

            spike_times = spike_times[
                (spike_times >= TIME_RANGE[0])
                & (spike_times <= TIME_RANGE[1])
            ]

            ax.vlines(
                spike_times,
                y - 0.4,
                y + 0.4,
                linewidth=0.5,
            )

            y += 1

        pattern_end_y = y - 1

        yticks.append((pattern_start_y + pattern_end_y) / 2)
        ylabels.append(short_pattern_name(pattern))

        # separator between patterns
        ax.axhline(y - 0.5, linewidth=0.5, alpha=0.4)
        y += 1

    add_epoch_background(ax)

    ax.set_xlim(TIME_RANGE)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylabel("Pattern / trials")
    ax.set_title(title)

    if y == 0:
        ax.text(
            0.5,
            0.5,
            "No spikes / trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


# =====================
# Page 2: 12 PSTHs
# =====================

def plot_psth_one_pattern(ax, labeled_unit, pattern):
    """Plot PSTH for one pattern."""
    df_pat = labeled_unit[labeled_unit["pattern"] == pattern].copy()

    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2

    n_trials = df_pat["trial_id"].nunique()

    if n_trials > 0:
        spike_times = df_pat["time_from_moving_onset"].to_numpy(dtype=float)
        spike_times = spike_times[
            (spike_times >= TIME_RANGE[0])
            & (spike_times <= TIME_RANGE[1])
        ]

        counts, _ = np.histogram(spike_times, bins=bins)
        fr = counts / n_trials / BIN_WIDTH

        ax.plot(centers, fr, linewidth=1.2)

    add_epoch_background(ax)

    ax.set_xlim(TIME_RANGE)
    ax.set_title(short_pattern_name(pattern), fontsize=9)

    if n_trials == 0:
        ax.text(
            0.5,
            0.5,
            "No trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )


# =====================
# Page 3: response summary
# =====================

def plot_pattern_response_summary(ax, trial_unit, pattern_unit, pattern_order):
    """
    Plot moving-baseline response by pattern.

    Data:
        individual trial dots from unit_trial_summary.csv
        mean ± SEM from unit_pattern_summary.csv
    """
    require_columns(
        trial_unit,
        ["pattern", "trial_id", "moving_minus_baseline"],
        "trial_unit",
    )

    df_trial = trial_unit[
        ["pattern", "trial_id", "moving_minus_baseline"]
    ].dropna(subset=["pattern", "moving_minus_baseline"]).copy()

    x = np.arange(len(pattern_order))
    rng = np.random.default_rng(42)

    # Individual trial dots
    for i, pattern in enumerate(pattern_order):
        vals = df_trial.loc[
            df_trial["pattern"] == pattern,
            "moving_minus_baseline",
        ].to_numpy(dtype=float)

        if len(vals) == 0:
            continue

        jitter = rng.uniform(-0.10, 0.10, size=len(vals))

        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=20,
            alpha=0.7,
        )

    # Mean ± SEM
    mean_vals = []
    sem_vals = []

    for pattern in pattern_order:
        row = pattern_unit[pattern_unit["pattern"] == pattern]

        if len(row) == 0:
            # Fallback to compute from trial table if pattern summary is missing
            vals = df_trial.loc[
                df_trial["pattern"] == pattern,
                "moving_minus_baseline",
            ].to_numpy(dtype=float)

            mean_vals.append(np.nanmean(vals) if len(vals) > 0 else np.nan)
            sem_vals.append(sem(vals) if len(vals) > 0 else np.nan)
        else:
            r = row.iloc[0]
            mean_vals.append(
                safe_float(r.get("moving_minus_baseline_mean", np.nan))
            )
            sem_vals.append(
                safe_float(r.get("moving_minus_baseline_sem", np.nan))
            )

    ax.errorbar(
        x,
        mean_vals,
        yerr=sem_vals,
        fmt="-o",
        capsize=4,
        linewidth=1.5,
    )

    ax.axhline(0, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_pattern_name(p) for p in pattern_order],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Moving - baseline FR")
    ax.set_title("Pattern response summary\n(trials + mean ± SEM)")


def get_pattern_label_from_sig_row(row):
    """
    Label response direction using raw two-sided p-value.

    resp*:
        p_motion_specific_two_sided < ALPHA and mean > 0

    supp*:
        p_motion_specific_two_sided < ALPHA and mean < 0
    """
    mean_val = safe_float(row.get("mean_moving_minus_baseline", np.nan))
    p2 = safe_float(row.get("p_motion_specific_two_sided", np.nan))

    if np.isnan(mean_val) or np.isnan(p2):
        return "ns"

    if p2 < ALPHA and mean_val > 0:
        return "resp*"

    if p2 < ALPHA and mean_val < 0:
        return "supp*"

    return "ns"


def add_pattern_stats_text(ax, unit_id, sig_unit, pattern_order):
    """Add pattern-specific raw p-values as text."""
    ax.axis("off")

    lines = [
        f"Unit {unit_id}",
        "",
        "Pattern-level motion test:",
        "diff = moving FR - baseline/static FR",
        "",
        "Columns:",
        "mean | p2 | pR | pS | label",
        "",
    ]

    if sig_unit is None or sig_unit.empty:
        lines.append("No unit_pattern_significance.csv row found.")
        ax.text(
            0.01,
            0.99,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
        )
        return

    for pattern in pattern_order:
        row = sig_unit[sig_unit["pattern"] == pattern]

        if len(row) == 0:
            lines.append(f"{short_pattern_name(pattern):<10} no data")
            continue

        r = row.iloc[0]

        mean_val = safe_float(r.get("mean_moving_minus_baseline", np.nan))
        p2 = r.get("p_motion_specific_two_sided", np.nan)
        p_resp = r.get("p_motion_responsive", np.nan)
        p_supp = r.get("p_motion_suppressed", np.nan)

        label = get_pattern_label_from_sig_row(r)

        lines.append(
            f"{short_pattern_name(pattern):<10} "
            f"{mean_val:>7.2f} | "
            f"{fmt_p(p2):>6} | "
            f"{fmt_p(p_resp):>6} | "
            f"{fmt_p(p_supp):>6} | "
            f"{label}"
        )

    # Compact raw-p counts
    lines += [""]

    n_resp = 0
    n_supp = 0
    n_specific = 0

    for _, r in sig_unit.iterrows():
        mean_val = safe_float(r.get("mean_moving_minus_baseline", np.nan))
        p2 = safe_float(r.get("p_motion_specific_two_sided", np.nan))

        if np.isnan(mean_val) or np.isnan(p2):
            continue

        if p2 < ALPHA:
            n_specific += 1

            if mean_val > 0:
                n_resp += 1
            elif mean_val < 0:
                n_supp += 1

    lines += [
        f"Patterns with p2<{ALPHA}: {n_specific}",
        f"Responsive by p2 + mean>0: {n_resp}",
        f"Suppressed by p2 + mean<0: {n_supp}",
    ]

    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

# =====================
# One-unit PDF
# =====================

def plot_one_unit(
    unit_id,
    labeled,
    trial_summary,
    pattern_summary,
    sig,
    out_dir,
):
    """Create a three-page PDF summary for one unit."""
    labeled_unit = labeled[labeled["unit_id"] == unit_id].copy()
    trial_unit = trial_summary[trial_summary["unit_id"] == unit_id].copy()
    pattern_unit = pattern_summary[pattern_summary["unit_id"] == unit_id].copy()

    if sig is None:
        sig_unit = pd.DataFrame()
    else:
        sig_unit = sig[sig["unit_id"] == unit_id].copy()

    if trial_unit.empty:
        print(f"Skipping unit {unit_id}: no trial summary.")
        return

    # Use all observed patterns for this unit, but keep the planned order.
    pattern_order = available_pattern_order(trial_unit)

    if len(pattern_order) == 0:
        print(f"Skipping unit {unit_id}: no pattern data.")
        return

    out_path = out_dir / f"unit_{unit_id}_vbc_summary.pdf"

    with PdfPages(out_path) as pdf:

        # -------------------------
        # Page 1: grouped raster
        # -------------------------
        fig1, axes = plt.subplots(
            4,
            1,
            figsize=(14, 16),
            sharex=True,
        )

        for ax, (group_name, patterns) in zip(axes, PATTERN_GROUPS.items()):
            plot_grouped_raster(
                ax=ax,
                labeled_unit=labeled_unit,
                pattern_list=patterns,
                title=group_name,
            )

        axes[-1].set_xlabel("Time from moving onset (s)")

        fig1.suptitle(f"Unit {unit_id} | grouped raster", fontsize=14)
        fig1.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig1)
        plt.close(fig1)

        # -------------------------
        # Page 2: 12 PSTHs
        # -------------------------
        fig2, axes = plt.subplots(
            3,
            4,
            figsize=(16, 10),
            sharex=True,
            sharey=True,
        )

        for ax, pattern in zip(axes.flat, PATTERN_ORDER):
            plot_psth_one_pattern(
                ax=ax,
                labeled_unit=labeled_unit,
                pattern=pattern,
            )

        # Hide extra axes if pattern list ever changes
        for ax in axes.flat[len(PATTERN_ORDER):]:
            ax.axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("Time from moving onset (s)")

        for ax in axes[:, 0]:
            ax.set_ylabel("FR (spikes/s)")

        fig2.suptitle(f"Unit {unit_id} | 12-pattern PSTHs", fontsize=14)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig2)
        plt.close(fig2)

        # -------------------------
        # Page 3: response summary + stats
        # -------------------------
        fig3 = plt.figure(figsize=(17, 10))
        gs = fig3.add_gridspec(
            1,
            2,
            width_ratios=[1.55, 1.0],
            wspace=0.25,
        )

        ax_response = fig3.add_subplot(gs[0, 0])
        plot_pattern_response_summary(
            ax=ax_response,
            trial_unit=trial_unit,
            pattern_unit=pattern_unit,
            pattern_order=PATTERN_ORDER,
        )

        ax_text = fig3.add_subplot(gs[0, 1])
        add_pattern_stats_text(
            ax=ax_text,
            unit_id=unit_id,
            sig_unit=sig_unit,
            pattern_order=PATTERN_ORDER,
        )

        fig3.suptitle(f"Unit {unit_id} | pattern response and statistics", fontsize=14)
        fig3.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig3)
        plt.close(fig3)

    print(f"Saved: {out_path}")


# =====================
# Main
# =====================

def main():
    print("===== Plot VbC three-screen unit summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "units_vbc"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"
    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    pattern_summary_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_summary.csv"
    sig_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_significance.csv"

    labeled = pd.read_csv(labeled_path)
    units = pd.read_csv(units_path)
    trial_summary = pd.read_csv(trial_summary_path)
    pattern_summary = pd.read_csv(pattern_summary_path)

    sig = pd.read_csv(sig_path) if sig_path.exists() else None

    # -----------------------------
    # Check required columns
    # -----------------------------

    require_columns(
        labeled,
        [
            "unit_id",
            "trial_id",
            "pattern",
            "time_from_moving_onset",
        ],
        "labeled_spikes.csv",
    )

    require_columns(
        units,
        [
            "unit_id",
        ],
        "curated_units.csv",
    )

    require_columns(
        trial_summary,
        [
            "unit_id",
            "trial_id",
            "pattern",
            "moving_minus_baseline",
        ],
        "unit_trial_summary.csv",
    )

    require_columns(
        pattern_summary,
        [
            "unit_id",
            "pattern",
            "moving_minus_baseline_mean",
            "moving_minus_baseline_sem",
        ],
        "unit_pattern_summary.csv",
    )

    if sig is not None:
        require_columns(
            sig,
            [
                "unit_id",
                "pattern",
                "mean_moving_minus_baseline",
                "p_motion_specific_two_sided",
                "p_motion_responsive",
                "p_motion_suppressed",
            ],
            "unit_pattern_significance.csv",
        )

    units_to_plot = get_units_to_plot(units, sig)

    print(f"Units to plot: {len(units_to_plot)}")
    print(units_to_plot)

    if len(units_to_plot) == 0:
        print("No units selected. Nothing to plot.")
        return

    for unit_id in units_to_plot:
        plot_one_unit(
            unit_id=unit_id,
            labeled=labeled,
            trial_summary=trial_summary,
            pattern_summary=pattern_summary,
            sig=sig,
            out_dir=plot_dir,
        )

    print("\n===== Saved VbC unit plots to =====")
    print(plot_dir)


if __name__ == "__main__":
    main()