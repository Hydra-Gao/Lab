# SB07_plot_population_12patterns.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from adjustText import adjust_text

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


ALPHA = 0.05
P_COL = "p_ttest_two_sided"
EPS_BASELINE_FR = 1e-6
MAX_EFFECT_RATIO_FOR_SIZE = 5.0
MIN_MARKER_SIZE = 100
MAX_MARKER_SIZE = 500

X_UNIT_OFFSET_MAX = 0.25     # fixed offset by unit
X_JITTER = 0.1              # additional random jitter in x
Y_JITTER_UM = 1           # random jitter in depth (um)
JITTER_SEED = 42
TEXT_X_OFFSET = 0.05        # small text shift to the right of each point
TEXT_Y_OFFSET = 0.5          # small text shift in depth

# Change this if your exact pattern names differ in the CSV.
PREFERRED_PATTERN_ORDER = [
    "VAl",
    "VAr",
    "HA_leftcorner_clockwise",
    "HA_leftcorner_anticlockwise",
    "HA_rightcorner_clockwise",
    "HA_rightcorner_anticlockwise",
    "Ascent",
    "Descent",
    "EXPANSION_l",
    "EXPANSION_r",
    "CONTRACTION_left",
    "CONTRACTION_right",
]


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def apply_collision_aware_dodge(
    df,
    x_col="pattern_x",
    y_col="depth_um",
    y_threshold_um=5.0,
    max_spread=0.22,
    seed=42,
):
    """
    For each pattern, if points have very similar depth values,
    spread them horizontally within a small local range.

    This is a soft collision-aware dodge, not a rigid arrangement.
    """
    rng = np.random.default_rng(seed)
    out = []

    for pattern_x, sub in df.groupby(x_col, sort=False):
        sub = sub.sort_values(y_col).copy()
        ys = sub[y_col].to_numpy()

        # build local clusters of nearby y values
        cluster_ids = np.zeros(len(sub), dtype=int)
        cluster_id = 0
        if len(sub) > 0:
            cluster_ids[0] = cluster_id
            for i in range(1, len(sub)):
                if abs(ys[i] - ys[i - 1]) <= y_threshold_um:
                    cluster_ids[i] = cluster_id
                else:
                    cluster_id += 1
                    cluster_ids[i] = cluster_id

        sub["_cluster_id"] = cluster_ids

        cluster_parts = []
        for _, g in sub.groupby("_cluster_id", sort=False):
            g = g.copy()
            n = len(g)

            if n == 1:
                g["pattern_x_plot"] = g[x_col].astype(float)
            else:
                # spread within a small range, plus tiny randomness
                offsets = np.linspace(-max_spread, max_spread, n)

                # small extra shuffle so it does not look too rigid
                offsets = offsets + rng.uniform(-0.015, 0.015, size=n)

                # randomize which point gets which offset, to avoid “ranked look”
                rng.shuffle(offsets)

                g["pattern_x_plot"] = g[x_col].astype(float) + offsets

            cluster_parts.append(g)

        sub2 = pd.concat(cluster_parts, ignore_index=True)
        out.append(sub2.drop(columns="_cluster_id"))

    return pd.concat(out, ignore_index=True)


def add_unit_labels_with_adjustment(ax, df, x_col="pattern_x_plot", y_col="depth_um_plot"):
    texts = []

    for _, row in df.iterrows():
        txt = ax.text(
            row[x_col],
            row[y_col],
            str(row["unit_id"]),
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
            ),
        )
        texts.append(txt)

    if texts:
        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "xy", "texts": "xy"},
            expand_points=(1.15, 1.25),
            expand_text=(1.15, 1.25),
            force_text=(0.4, 0.6),
            force_points=(0.2, 0.3),
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.5,
                alpha=0.7,
            ),
        )


def make_unit_x_offsets(unit_ids, max_offset=X_UNIT_OFFSET_MAX):
    """
    Give each unit a fixed x-offset so units at the same pattern
    do not completely overlap.
    """
    unit_ids = list(unit_ids)

    if len(unit_ids) == 1:
        return {unit_ids[0]: 0.0}

    offsets = np.linspace(-max_offset, max_offset, len(unit_ids))
    return {u: off for u, off in zip(unit_ids, offsets)}


def make_pattern_order(patterns):
    """Use preferred order when names match; append unexpected names at the end."""
    patterns = [str(p) for p in pd.Series(patterns).dropna().unique().tolist()]
    ordered = [p for p in PREFERRED_PATTERN_ORDER if p in patterns]
    remaining = sorted([p for p in patterns if p not in ordered])
    return ordered + remaining


def effect_ratio_to_size(effect_ratio_abs):
    """
    Convert absolute fractional FR change into marker area.

    effect_ratio_abs = abs((moving_fr_mean - baseline_fr_mean) / baseline_fr_mean)

    Example:
        0.5 means +50% or -50% relative change.
        1.0 means +100% or -100% relative change.
    """
    x = np.asarray(effect_ratio_abs, dtype=float)
    x = np.clip(x, 0, MAX_EFFECT_RATIO_FOR_SIZE)
    return MIN_MARKER_SIZE + (x / MAX_EFFECT_RATIO_FOR_SIZE) * (
        MAX_MARKER_SIZE - MIN_MARKER_SIZE
    )


def add_size_legend(ax):
    legend_effects = [0.5, 1.0, 2.0, 5.0]
    handles = []
    labels = []

    for e in legend_effects:
        handles.append(
            plt.scatter(
                [],
                [],
                s=effect_ratio_to_size(e),
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
            )
        )
        labels.append(f"{e:g}")

    leg = ax.legend(
        handles,
        labels,
        title="|moving - baseline|\n/ |baseline|",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        labelspacing=1.2,
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(leg)


def add_effect_ratio_colorbar(fig, ax, norm, cmap):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
        fraction=0.05,
    )
    cbar.set_label("(moving - baseline) / baseline", rotation=90)
    return cbar


def plot_one_speed(df_speed, pattern_order, unit_ids, speed_label, out_path):
    sig = df_speed.loc[df_speed["is_sig_ttest_p05"]].copy()

    fig_width = max(14, 0.9 * len(pattern_order) + 5)
    fig_height = max(8, 0.35 * len(unit_ids) + 5.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("turbo")

    if sig.empty:
        color_limit = 1.0
    else:
        color_limit = max(np.nanpercentile(sig["effect_ratio"].abs(), 95), 0.5)

    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    if sig.empty:
        ax.text(
            0.5,
            0.5,
            f"No significant unit × pattern points at p < {ALPHA}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        # apply collision-aware dodge within each pattern
        sig_plot = apply_collision_aware_dodge(
            sig,
            x_col="pattern_x",
            y_col="depth_um",
            y_threshold_um=4.0,   # tune if needed
            max_spread=0.22,      # tune if needed
            seed=42,
        ).copy()

        # keep depth mostly real
        sig_plot["depth_um_plot"] = sig_plot["depth_um"]

        ax.scatter(
            sig_plot["pattern_x_plot"],
            sig_plot["depth_um_plot"],
            s=120,
            c=sig_plot["effect_ratio"],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.92,
            zorder=3,
        )

        add_unit_labels_with_adjustment(
            ax,
            sig_plot,
            x_col="pattern_x_plot",
            y_col="depth_um_plot",
        )

    ax.set_xticks(range(len(pattern_order)))
    ax.set_xticklabels(pattern_order, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(pattern_order) - 0.4)

    ax.set_xlabel("12-pattern stimulus")
    ax.set_ylabel("Depth (um)")
    ax.set_title(
        f"Population significant responses by depth\n"
        f"p_ttest_two_sided < {ALPHA}; color = (moving - baseline) / baseline"
        f"{speed_label}"
    )

    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("(moving - baseline) / baseline", rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("===== Plot 12-pattern population significance by depth =====")

    sig_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_significance.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    sig = pd.read_csv(sig_path)
    units = pd.read_csv(units_path)

    require_columns(
        sig,
        [
            "unit_id",
            "pattern",
            "speed_deg_per_sec",
            P_COL,
            "baseline_fr_mean",
            "moving_fr_mean",
        ],
        "unit_pattern_significance.csv",
    )
    require_columns(units, ["unit_id", "depth_um"], "curated_units.csv")

    # SB06 may not carry depth_um, so merge it explicitly from curated_units.csv.
    if "depth_um" in sig.columns:
        sig = sig.drop(columns=["depth_um"])

    sig = sig.merge(
        units[["unit_id", "depth_um"]].drop_duplicates(),
        on="unit_id",
        how="left",
    )

    
    # Fractional response change.
    # Positive = moving FR above baseline; negative = moving FR below baseline.
    sig["effect_ratio"] = (
        sig["moving_fr_mean"] - sig["baseline_fr_mean"]
    ) / (sig["baseline_fr_mean"].abs() + EPS_BASELINE_FR)

    sig["effect_ratio_abs"] = sig["effect_ratio"].abs()
    sig["marker_size"] = effect_ratio_to_size(sig["effect_ratio_abs"])

    sig["is_sig_ttest_p05"] = sig[P_COL] < ALPHA

    pattern_order = make_pattern_order(sig["pattern"])
    pattern_to_x = {p: i for i, p in enumerate(pattern_order)}
    sig["pattern_x"] = sig["pattern"].map(pattern_to_x)

    # Sort units by depth for stable coloring/legend order.
    unit_depth = (
        sig[["unit_id", "depth_um"]]
        .drop_duplicates()
        .sort_values(["depth_um", "unit_id"], na_position="last")
    )
    unit_ids = unit_depth["unit_id"].tolist()

    out_dir = ANALYSIS_OUTPUT_DIR / "population_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_csv_path = out_dir / "population_12patterns_effect_ratio_table.csv"
    sig.to_csv(annotated_csv_path, index=False)

    speed_values = sorted(sig["speed_deg_per_sec"].dropna().unique().tolist())

    saved = []
    if len(speed_values) == 0:
        out_path = out_dir / "population_12patterns_by_depth.png"
        plot_one_speed(sig, pattern_order, unit_ids, "", out_path)
        saved.append(out_path)
    else:
        for speed in speed_values:
            df_speed = sig.loc[sig["speed_deg_per_sec"] == speed].copy()
            speed_safe = str(speed).replace(".", "p")
            out_path = out_dir / f"population_12patterns_by_depth_speed_{speed_safe}.png"
            speed_label = f"\nspeed = {speed:g} deg/s"
            plot_one_speed(
                df_speed,
                pattern_order,
                unit_ids,
                speed_label,
                out_path,
            )
            saved.append(out_path)

    print("\nSaved annotated effect-ratio table:")
    print(annotated_csv_path)

    print("\nSaved figures:")
    for p in saved:
        print(p)

    print("\nSignificant points per speed:")
    print(
        sig.groupby("speed_deg_per_sec", dropna=False)["is_sig_ttest_p05"]
        .sum()
        .rename("n_sig_points")
        .reset_index()
    )


if __name__ == "__main__":
    main()
