# 07_plot_population.py

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

SIGNIFICANCE_COLUMNS = [
    "is_motion_responsive",
    "is_motion_suppressed",
    "is_direction_tuned",
]


def build_heatmap_table(condition_summary, units_to_plot):
    df = condition_summary[
        condition_summary["unit_id"].isin(units_to_plot)
    ].copy()

    if df.empty:
        return None

    heat = df.pivot_table(
        index="unit_id",
        columns="direction",
        values="moving_minus_static",
        aggfunc="mean",
    )

    # Sort rows by preferred direction, then by max response
    pref_dir = heat.idxmax(axis=1)
    max_resp = heat.max(axis=1)

    sort_df = pd.DataFrame(
        {
            "unit_id": heat.index,
            "preferred_direction": pref_dir,
            "max_response": max_resp,
        }
    ).sort_values(["preferred_direction", "max_response"])

    heat = heat.loc[sort_df["unit_id"]]
    return heat


def zscore_rows(heat):
    """
    Z-score each row (unit) across directions.
    """
    row_mean = heat.mean(axis=1)
    row_std = heat.std(axis=1).replace(0, np.nan)

    heat_z = heat.sub(row_mean, axis=0).div(row_std, axis=0)
    return heat_z.fillna(0)


def get_units_to_plot(tuning, sig):
    """Choose all units or significant units only."""

    if PLOT_MODE == "all":
        return tuning["unit_id"].tolist()

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


def build_heatmap_table(condition_summary, units_to_plot):
    """Build unit × direction response matrix."""
    df = condition_summary[
        condition_summary["unit_id"].isin(units_to_plot)
    ].copy()

    if df.empty:
        return None

    heat = df.pivot_table(
        index="unit_id",
        columns="direction",
        values="moving_minus_static",
        aggfunc="mean",
    )

    # Sort rows by preferred direction, then by max response
    pref_dir = heat.idxmax(axis=1)
    max_resp = heat.max(axis=1)

    sort_df = pd.DataFrame(
        {
            "unit_id": heat.index,
            "preferred_direction": pref_dir,
            "max_response": max_resp,
        }
    ).sort_values(["preferred_direction", "max_response"])

    heat = heat.loc[sort_df["unit_id"]]

    return heat


def zscore_rows(heat):
    """Z-score each unit across directions."""
    row_mean = heat.mean(axis=1)
    row_std = heat.std(axis=1).replace(0, np.nan)

    heat_z = heat.sub(row_mean, axis=0).div(row_std, axis=0)
    return heat_z.fillna(0)


def plot_motion_response_heatmap(ax, condition_summary, units_to_plot):
    """
    Raw heatmap:
        rows = units
        columns = directions
        values = moving - static firing rate
    """
    heat = build_heatmap_table(condition_summary, units_to_plot)

    if heat is None or heat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Raw heatmap")
        return None

    im = ax.imshow(
        heat.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(x) for x in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([str(x) for x in heat.index], fontsize=6)

    ax.set_xlabel("Direction")
    ax.set_ylabel("Unit ID")
    ax.set_title("Raw response\nmoving FR - static FR")

    return im


def plot_motion_response_heatmap_zscore(ax, condition_summary, units_to_plot):
    """
    Z-scored heatmap:
        each unit normalized across directions
    """
    heat = build_heatmap_table(condition_summary, units_to_plot)

    if heat is None or heat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Z-scored heatmap")
        return None

    heat_z = zscore_rows(heat)

    im = ax.imshow(
        heat_z.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(heat_z.columns)))
    ax.set_xticklabels([str(x) for x in heat_z.columns])
    ax.set_yticks(np.arange(len(heat_z.index)))
    ax.set_yticklabels([str(x) for x in heat_z.index], fontsize=6)

    ax.set_xlabel("Direction")
    ax.set_ylabel("Unit ID")
    ax.set_title("Normalized response\nz-scored within unit")

    return im


def plot_preferred_direction_distribution(ax, tuning, units_to_plot):
    """Histogram of preferred directions."""
    df = tuning[tuning["unit_id"].isin(units_to_plot)].copy()
    df = df.dropna(subset=["preferred_direction"])

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Preferred direction")
        return

    directions = sorted(df["preferred_direction"].dropna().unique())
    counts = df["preferred_direction"].value_counts().reindex(directions, fill_value=0)

    ax.bar(np.arange(len(directions)), counts.values)

    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels([str(x) for x in directions])

    ax.set_xlabel("Preferred direction")
    ax.set_ylabel("Number of units")
    ax.set_title("Preferred direction distribution")


def plot_dsi_distribution(ax, tuning, units_to_plot):
    """Histogram of DSI."""
    df = tuning[tuning["unit_id"].isin(units_to_plot)].copy()
    df = df.dropna(subset=["dsi"])

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("DSI")
        return

    ax.hist(df["dsi"], bins=15)

    ax.set_xlabel("DSI")
    ax.set_ylabel("Number of units")
    ax.set_title("DSI distribution")


def plot_vector_strength_distribution(ax, tuning, units_to_plot):
    """Histogram of vector strength."""
    df = tuning[tuning["unit_id"].isin(units_to_plot)].copy()
    df = df.dropna(subset=["vector_strength"])

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Vector strength")
        return

    ax.hist(df["vector_strength"], bins=15)

    ax.set_xlabel("Vector strength")
    ax.set_ylabel("Number of units")
    ax.set_title("Vector strength distribution")


def plot_significance_summary(ax, sig):
    """Bar plot showing number of significant units by category."""
    if sig is None:
        ax.text(0.5, 0.5, "No significance file", ha="center", va="center")
        ax.set_title("Significant unit counts")
        return

    categories = {
        "motion\nresponsive": "is_motion_responsive",
        "motion\nsuppressed": "is_motion_suppressed",
        "direction\ntuned": "is_direction_tuned",
    }

    counts = []

    for _, col in categories.items():
        if col in sig.columns:
            counts.append(int(sig[col].fillna(False).astype(bool).sum()))
        else:
            counts.append(0)

    ax.bar(np.arange(len(categories)), counts)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(list(categories.keys()))
    ax.set_ylabel("Number of units")
    ax.set_title("Significant unit counts")

    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom")


def save_population_table(tuning, condition_summary, sig, units_to_plot, out_dir):
    """Save a compact table of plotted units."""

    pop = tuning[tuning["unit_id"].isin(units_to_plot)].copy()

    if sig is not None:
        sig_cols = [
            "unit_id",
            "is_motion_responsive",
            "is_motion_suppressed",
            "is_direction_tuned",
            "q_motion_specific",
            "q_direction_tuning",
        ]

        sig_cols = [c for c in sig_cols if c in sig.columns]

        pop = pop.merge(
            sig[sig_cols],
            on="unit_id",
            how="left",
        )

    out_path = out_dir / "population_units_plotted.csv"
    pop.to_csv(out_path, index=False)

    print(f"Saved table: {out_path}")


def main():
    print("===== Plot population summaries =====")
    print(f"PLOT_MODE = {PLOT_MODE}")

    plot_dir = ANALYSIS_OUTPUT_DIR / "plots" / "population"
    plot_dir.mkdir(parents=True, exist_ok=True)

    condition_summary = pd.read_csv(
        ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv"
    )

    tuning = pd.read_csv(
        ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"
    )

    sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() else None

    units_to_plot = get_units_to_plot(tuning, sig)

    print(f"Units to plot: {len(units_to_plot)}")
    print(units_to_plot)

    if len(units_to_plot) == 0:
        print("No units selected. Nothing to plot.")
        return

    pdf_path = plot_dir / "population_summary_one_page.pdf"

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    im_raw = plot_motion_response_heatmap(
        ax=axes[0, 0],
        condition_summary=condition_summary,
        units_to_plot=units_to_plot,
    )

    im_z = plot_motion_response_heatmap_zscore(
        ax=axes[0, 1],
        condition_summary=condition_summary,
        units_to_plot=units_to_plot,
    )

    plot_preferred_direction_distribution(
        ax=axes[1, 0],
        tuning=tuning,
        units_to_plot=units_to_plot,
    )

    plot_dsi_distribution(
        ax=axes[1, 1],
        tuning=tuning,
        units_to_plot=units_to_plot,
    )

    plot_vector_strength_distribution(
        ax=axes[2, 0],
        tuning=tuning,
        units_to_plot=units_to_plot,
    )

    plot_significance_summary(
        ax=axes[2, 1],
        sig=sig,
    )

    # Add colorbars only for the two heatmaps
    if im_raw is not None:
        cbar = fig.colorbar(im_raw, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar.set_label("Moving - static FR")

    if im_z is not None:
        cbar = fig.colorbar(im_z, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Z-score")

    fig.suptitle(
        f"Population summary | {PLOT_MODE} units | n = {len(units_to_plot)}",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(pdf_path)
    plt.close(fig)

    save_population_table(
        tuning=tuning,
        condition_summary=condition_summary,
        sig=sig,
        units_to_plot=units_to_plot,
        out_dir=plot_dir,
    )

    print("\n===== Saved population plots to =====")
    print(plot_dir)
    print(pdf_path)


if __name__ == "__main__":
    main()