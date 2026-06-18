# SB12c_plot_population_8directions.py
#
# Population plots for single-screen 8-direction / 2-speed data.
#
# Expected inputs in ANALYSIS_OUTPUT_DIR:
#   curated_units.csv
#   unit_condition_summary.csv          from SB05a1
#   unit_direction_significance.csv     from SB06a1
#
# Main output:
#   population_plots_8directions/population_8directions_summary.pdf
#
# Pages:
#   Page 1: significant unit x screen x direction points for speed 1
#   Page 2: significant unit x screen x direction points for speed 2
#   Page 3: polar population plots for speed 1
#   Page 4: polar population plots for speed 2

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


# =====================
# User settings
# =====================

ALPHA = 0.05
P_COL = "p_motion_baseline_two_sided"
EPS_BASELINE_FR = 1e-6

DIRECTION_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]

# For the first two pages.
Y_COLLISION_THRESHOLD_UM = 4.0
X_DODGE_MAX = 0.22
LABEL_FONT_SIZE = 7
POINT_SIZE = 120

# For polar pages.
RAW_POLAR_POINT_SIZE = 28
MEAN_POLAR_LINEWIDTH = 2.2

# If True, collapse across screen_role before population averaging in polar plots.
# This prevents a unit recorded on 3 screens from being weighted 3x more than a
# unit with only 1 screen. Raw polar points will be unit-level after this collapse.
COLLAPSE_SCREEN_ROLE_FOR_POLAR = False

# If you prefer only FDR-corrected significant direction points on pages 1-2,
# change this to "q_motion_baseline_direction".
SIGNIFICANCE_COL_FOR_PAGES_1_2 = P_COL


# =====================
# Small helpers
# =====================


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "screen_role" not in df.columns:
        df["screen_role"] = "unknown"

    for col in ["unit_id", "screen_role", "speed"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360
    df["direction"] = df["direction"].round(6)

    return df


def direction_label(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if abs(float(x) - round(float(x))) < 1e-6:
        return f"{int(round(float(x)))}°"
    return f"{float(x):g}°"


def direction_to_x_map(directions: pd.Series) -> tuple[list[float], dict[float, int]]:
    observed = sorted(pd.to_numeric(directions, errors="coerce").dropna().unique().tolist())
    preferred = [float(d) for d in DIRECTION_ORDER if float(d) in observed]
    remaining = [float(d) for d in observed if float(d) not in preferred]
    order = preferred + remaining
    return order, {d: i for i, d in enumerate(order)}


def apply_collision_aware_dodge(
    df: pd.DataFrame,
    x_col: str = "direction_x",
    y_col: str = "depth_um",
    y_threshold_um: float = Y_COLLISION_THRESHOLD_UM,
    max_spread: float = X_DODGE_MAX,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Within each x-bin, horizontally spread points with very similar depths.
    This keeps the y-axis interpretable while reducing overplotting.
    """
    rng = np.random.default_rng(seed)
    parts = []

    for x_value, sub in df.groupby(x_col, sort=False):
        sub = sub.sort_values(y_col, na_position="last").copy()
        ys = sub[y_col].to_numpy(dtype=float)

        cluster_ids = np.zeros(len(sub), dtype=int)
        cluster_id = 0
        for i in range(1, len(sub)):
            if np.isfinite(ys[i]) and np.isfinite(ys[i - 1]) and abs(ys[i] - ys[i - 1]) <= y_threshold_um:
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
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(-max_spread, max_spread, n)
                offsets += rng.uniform(-0.015, 0.015, size=n)
                rng.shuffle(offsets)
            g["direction_x_plot"] = g[x_col].astype(float) + offsets
            cluster_parts.append(g)

        parts.append(pd.concat(cluster_parts, ignore_index=True).drop(columns="_cluster_id"))

    if not parts:
        return df.copy()
    return pd.concat(parts, ignore_index=True)


def add_unit_labels_with_adjustment(ax, df: pd.DataFrame) -> None:
    texts = []
    for _, row in df.iterrows():
        screen_suffix = ""
        if "screen_role" in row.index and str(row["screen_role"]) not in ["unknown", "nan"]:
            screen_suffix = f"/{row['screen_role']}"
        txt = ax.text(
            row["direction_x_plot"],
            row["depth_um"],
            f"{row['unit_id']}{screen_suffix}",
            fontsize=LABEL_FONT_SIZE,
            color="black",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.65),
        )
        texts.append(txt)

    if texts and adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "xy", "texts": "xy"},
            expand_points=(1.15, 1.25),
            expand_text=(1.15, 1.25),
            force_text=(0.4, 0.6),
            force_points=(0.2, 0.3),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
        )


def get_screen_markers(screen_roles: list[str]) -> dict[str, str]:
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    return {screen: markers[i % len(markers)] for i, screen in enumerate(screen_roles)}


def add_effect_colorbar(fig, ax, norm, cmap) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("(moving - baseline) / baseline", rotation=90)


def make_effect_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["effect_ratio"] = df["moving_minus_baseline"] / (df["baseline_fr"].abs() + EPS_BASELINE_FR)
    df["effect_ratio_abs"] = df["effect_ratio"].abs()
    return df


def get_polar_baseline_col(df: pd.DataFrame) -> str:
    """
    Choose baseline column for baseline-vs-moving polar plots.

    Priority:
      1. pooled_static_fr: preferred, if generated by the relative-direction pooled baseline logic
      2. static_fr
      3. baseline_fr: fallback from current population table
    """
    for col in ["pooled_static_fr", "static_fr", "baseline_fr"]:
        if col in df.columns:
            return col

    raise ValueError(
        "No baseline column found. Expected one of: "
        "pooled_static_fr, static_fr, baseline_fr"
    )


# =====================
# Data loading / merging
# =====================


def load_population_table() -> pd.DataFrame:
    condition_path = ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv"
    direction_sig_path = ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    condition = pd.read_csv(condition_path)
    direction_sig = pd.read_csv(direction_sig_path)
    units = pd.read_csv(units_path)

    require_columns(
        condition,
        ["unit_id", "screen_role", "speed", "direction", "baseline_fr", "moving_fr", "moving_minus_baseline"],
        "unit_condition_summary.csv",
    )
    require_columns(
        direction_sig,
        ["unit_id", "screen_role", "speed", "direction", P_COL],
        "unit_direction_significance.csv",
    )
    require_columns(units, ["unit_id", "depth_um"], "curated_units.csv")

    condition = normalize_key_columns(condition)
    direction_sig = normalize_key_columns(direction_sig)
    units = units.copy()
    units["unit_id"] = units["unit_id"].astype(str).str.strip()

    keep_condition_cols = [
        "unit_id", "screen_role", "speed", "direction",
        "baseline_fr", "moving_fr", "moving_minus_baseline",

        "pooled_static_fr", "static_fr",
        "speed_label", "speed_deg_per_sec", "tf_hz", "sf_cpd",
    ]
    keep_condition_cols = [c for c in keep_condition_cols if c in condition.columns]

    keep_sig_cols = [
        "unit_id", "screen_role", "speed", "direction",
        P_COL, "q_motion_baseline_direction",
        "mean_moving_minus_baseline",
    ]
    keep_sig_cols = [c for c in keep_sig_cols if c in direction_sig.columns]

    df = condition[keep_condition_cols].merge(
        direction_sig[keep_sig_cols],
        on=["unit_id", "screen_role", "speed", "direction"],
        how="left",
        suffixes=("", "_sig"),
    )

    df = df.merge(
        units[["unit_id", "depth_um"]].drop_duplicates(),
        on="unit_id",
        how="left",
    )

    df = make_effect_ratio(df)

    # Use all units in curated_units.csv as the good-unit universe.
    good_units = units["unit_id"].dropna().unique().tolist()
    df = df.loc[df["unit_id"].isin(good_units)].copy()

    return df


# =====================
# Pages 1-2: direction x depth scatter
# =====================


def plot_significant_depth_page(
    fig,
    df_speed: pd.DataFrame,
    speed: str,
    direction_order: list[float],
    direction_to_x: dict[float, int],
) -> None:
    ax = fig.add_subplot(111)

    sig = df_speed.loc[df_speed[SIGNIFICANCE_COL_FOR_PAGES_1_2] < ALPHA].copy()
    sig["direction_x"] = sig["direction"].map(direction_to_x)
    sig = sig.dropna(subset=["direction_x", "depth_um"])

    cmap = plt.get_cmap("turbo")
    if sig.empty:
        color_limit = 1.0
    else:
        color_limit = max(float(np.nanpercentile(sig["effect_ratio"].abs(), 95)), 0.5)

    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    if sig.empty:
        ax.text(
            0.5, 0.5,
            f"No significant unit x direction points at {SIGNIFICANCE_COL_FOR_PAGES_1_2} < {ALPHA}",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        sig_plot = apply_collision_aware_dodge(sig).copy()

        screen_roles = sorted(sig_plot["screen_role"].dropna().astype(str).unique().tolist())
        marker_map = get_screen_markers(screen_roles)

        for screen_role in screen_roles:
            sub = sig_plot.loc[sig_plot["screen_role"].astype(str) == screen_role]
            ax.scatter(
                sub["direction_x_plot"],
                sub["depth_um"],
                s=POINT_SIZE,
                c=sub["effect_ratio"],
                cmap=cmap,
                norm=norm,
                marker=marker_map[screen_role],
                edgecolors="black",
                linewidths=0.45,
                alpha=0.92,
                zorder=3,
                label=screen_role,
            )

        add_unit_labels_with_adjustment(ax, sig_plot)

        if len(screen_roles) > 1:
            leg = ax.legend(
                title="screen_role",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                frameon=False,
                fontsize=9,
                title_fontsize=10,
            )
            ax.add_artist(leg)

    ax.set_xticks(range(len(direction_order)))
    ax.set_xticklabels([direction_label(d) for d in direction_order], rotation=0)
    ax.set_xlim(-0.6, len(direction_order) - 0.4)
    ax.set_xlabel("Motion direction")
    ax.set_ylabel("Depth (um)")
    ax.set_title(
        "Population significant responses by depth\n"
        f"speed = {speed}; {SIGNIFICANCE_COL_FOR_PAGES_1_2} < {ALPHA}; "
        "color = (moving - baseline) / baseline"
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    add_effect_colorbar(fig, ax, norm, cmap)
    fig.tight_layout()


# =====================
# Pages 3-4: polar plots
# =====================


def collapse_for_polar(df_speed: pd.DataFrame) -> pd.DataFrame:
    df = df_speed.copy()

    if COLLAPSE_SCREEN_ROLE_FOR_POLAR:
        group_cols = ["unit_id", "speed", "direction"]
        df = (
            df.groupby(group_cols, dropna=False)
            .agg(
                baseline_fr=("baseline_fr", "mean"),
                moving_fr=("moving_fr", "mean"),
                moving_minus_baseline=("moving_minus_baseline", "mean"),
                depth_um=("depth_um", "first"),
            )
            .reset_index()
        )
        df = make_effect_ratio(df)

    return df


def close_polar_curve(values_by_direction: pd.Series, direction_order: list[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([values_by_direction.get(float(d), np.nan) for d in direction_order], dtype=float)
    theta = np.deg2rad(np.array(direction_order, dtype=float))

    if len(theta) > 0:
        theta = np.r_[theta, theta[0]]
        values = np.r_[values, values[0]]

    return theta, values


def close_polar(theta, r):
    """
    Compatibility helper for closing polar curves.

    theta: already in radians
    r: radial values
    """
    theta = np.asarray(theta)
    r = np.asarray(r)

    if len(theta) == 0:
        return theta, r

    return np.r_[theta, theta[0]], np.r_[r, r[0]]


def scatter_raw_polar_points(ax, raw: pd.DataFrame, value_col: str, direction_order: list[float]) -> None:
    unit_ids = sorted(raw["unit_id"].dropna().astype(str).unique().tolist())
    if not unit_ids:
        return

    cmap = plt.get_cmap("tab20", max(len(unit_ids), 1))
    unit_to_color = {u: cmap(i) for i, u in enumerate(unit_ids)}

    for i, unit_id in enumerate(unit_ids):
        sub = raw.loc[raw["unit_id"].astype(str) == unit_id].copy()
        sub = sub.dropna(subset=["direction", value_col])
        if sub.empty:
            continue
        ax.scatter(
            np.deg2rad(sub["direction"].astype(float).to_numpy()),
            sub[value_col].astype(float).to_numpy(),
            s=RAW_POLAR_POINT_SIZE,
            alpha=0.65,
            color=unit_to_color[unit_id],
            label=str(unit_id),
        )


def set_polar_format(ax, title: str) -> None:
    ax.set_title(title, pad=16)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(DIRECTION_ORDER, labels=[f"{d}°" for d in DIRECTION_ORDER])
    ax.grid(True, alpha=0.3)


def screen_sort_key(screen: str) -> int:
    """
    Force screen order: left, front, right.
    Unknown screens go to the end.
    """
    s = str(screen).lower()
    if "left" in s:
        return 0
    if "front" in s:
        return 1
    if "right" in s:
        return 2
    return 99


def plot_screen_split_polar_page(
    fig,
    df_speed: pd.DataFrame,
    speed: str,
    direction_order: list[float],
) -> None:
    """
    2 x 3 polar layout:
      Top row:
        left / front / right fraction change
        fraction change = (moving - baseline) / baseline

      Bottom row:
        left / front / right baseline vs moving FR
        baseline uses pooled_static_fr if available, then static_fr, then baseline_fr
    """

    raw = df_speed.copy()
    raw = raw.dropna(subset=["direction", "screen_role"])
    raw["direction"] = raw["direction"].astype(float)
    raw["screen_role"] = raw["screen_role"].astype(str)

    baseline_col = get_polar_baseline_col(raw)

    screen_roles = sorted(
        raw["screen_role"].dropna().unique().tolist(),
        key=screen_sort_key,
    )

    # Usually this should be left/front/right.
    screen_roles = screen_roles[:3]

    if len(screen_roles) == 0:
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No valid screen_role found for polar plots",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Use same unit colors across all six plots.
    unit_ids = sorted(raw["unit_id"].dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(unit_ids), 1))
    unit_to_color = {u: cmap(i) for i, u in enumerate(unit_ids)}

    axes = []

    for col_idx, screen_role in enumerate(screen_roles):
        screen_raw = raw.loc[raw["screen_role"] == screen_role].copy()

        # =========================
        # Top row: fraction change
        # =========================
        ax_ratio = fig.add_subplot(2, 3, col_idx + 1, projection="polar")

        ratio_mean = screen_raw.groupby("direction")["effect_ratio"].mean()
        theta_ratio, ratio_values = close_polar_curve(ratio_mean, direction_order)

        for unit_id in unit_ids:
            sub = screen_raw.loc[screen_raw["unit_id"].astype(str) == unit_id].copy()
            sub = sub.dropna(subset=["direction", "effect_ratio"])

            if sub.empty:
                continue

            ax_ratio.scatter(
                np.deg2rad(sub["direction"].astype(float).to_numpy()),
                sub["effect_ratio"].astype(float).to_numpy(),
                s=RAW_POLAR_POINT_SIZE,
                alpha=0.65,
                color=unit_to_color[unit_id],
                label=str(unit_id),
            )

        ax_ratio.plot(
            theta_ratio,
            ratio_values,
            color="black",
            lw=MEAN_POLAR_LINEWIDTH,
            label="population mean",
        )
        ax_ratio.scatter(theta_ratio, ratio_values, color="black", s=24)

        set_polar_format(
            ax_ratio,
            f"{screen_role}\nFraction change",
        )

        axes.append(ax_ratio)

        # =====================================
        # Bottom row: baseline FR vs moving FR
        # =====================================
        ax_fr = fig.add_subplot(2, 3, col_idx + 4, projection="polar")

        fr_df = screen_raw[
            ["direction", baseline_col, "moving_fr"]
        ].dropna(subset=["direction"]).copy()

        # Population mean across units for each direction.
        fr_mean = (
            fr_df
            .groupby("direction", as_index=False)
            .agg(
                baseline_fr_for_polar=(baseline_col, "mean"),
                moving_fr=("moving_fr", "mean"),
            )
            .sort_values("direction")
        )

        theta = np.deg2rad(fr_mean["direction"].astype(float).to_numpy())

        if len(theta) > 0:
            theta_c, baseline_c = close_polar(
                theta,
                fr_mean["baseline_fr_for_polar"].clip(lower=0).to_numpy(),
            )
            _, moving_c = close_polar(
                theta,
                fr_mean["moving_fr"].clip(lower=0).to_numpy(),
            )

            ax_fr.plot(
                theta_c,
                baseline_c,
                marker="o",
                linewidth=1.8,
                label=f"baseline: {baseline_col}",
            )
            ax_fr.fill(theta_c, baseline_c, alpha=0.08)

            ax_fr.plot(
                theta_c,
                moving_c,
                marker="o",
                linewidth=1.8,
                label="moving FR",
            )
            ax_fr.fill(theta_c, moving_c, alpha=0.08)

        # Optional raw unit points for baseline and moving.
        # These are lighter than the mean lines.
        for unit_id in unit_ids:
            sub = screen_raw.loc[screen_raw["unit_id"].astype(str) == unit_id].copy()
            sub = sub.dropna(subset=["direction", baseline_col, "moving_fr"])

            if sub.empty:
                continue

            theta_raw = np.deg2rad(sub["direction"].astype(float).to_numpy())

            ax_fr.scatter(
                theta_raw,
                sub[baseline_col].clip(lower=0).astype(float).to_numpy(),
                s=RAW_POLAR_POINT_SIZE * 0.65,
                alpha=0.25,
                color=unit_to_color[unit_id],
            )

            ax_fr.scatter(
                theta_raw,
                sub["moving_fr"].clip(lower=0).astype(float).to_numpy(),
                s=RAW_POLAR_POINT_SIZE * 0.65,
                alpha=0.45,
                color=unit_to_color[unit_id],
            )

        set_polar_format(
            ax_fr,
            f"{screen_role}\nBaseline vs moving FR",
        )

        ax_fr.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            fontsize=7,
            frameon=False,
        )

        axes.append(ax_fr)

    fig.suptitle(
        f"Population polar plots by screen, speed = {speed}",
        y=0.98,
        fontsize=14,
    )

    # Unit legend for raw points, from the fraction-change row.
    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    # Only keep unit_id labels, not baseline/moving labels.
    unit_handles = []
    unit_labels = []
    for h, lab in zip(handles, labels):
        if lab in unit_ids:
            unit_handles.append(h)
            unit_labels.append(lab)

    if unit_handles:
        fig.legend(
            unit_handles,
            unit_labels,
            title="raw points: unit_id",
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=False,
            fontsize=7,
            title_fontsize=9,
        )

    fig.text(
        0.03,
        0.73,
        "Fraction change\n(moving - baseline) / baseline",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
    )

    fig.text(
        0.03,
        0.28,
        f"Firing rate\nbaseline = {baseline_col}\nmoving = moving_fr",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
    )

    fig.tight_layout(rect=(0.05, 0, 0.9, 0.94))


# =====================
# Main
# =====================


def main() -> None:
    print("===== Plot 8-direction population summary =====")

    df = load_population_table()

    direction_order, direction_to_x = direction_to_x_map(df["direction"])
    if not direction_order:
        raise ValueError("No valid directions found in merged table.")

    speed_values = sorted(df["speed"].dropna().astype(str).unique().tolist())
    if len(speed_values) == 0:
        raise ValueError("No speed values found in merged table.")

    if len(speed_values) > 2:
        print(f"Warning: found more than 2 speeds: {speed_values}. Plotting all speeds in order.")

    out_dir = ANALYSIS_OUTPUT_DIR / "population_plots_8directions"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_csv_path = out_dir / "population_8directions_effect_ratio_table.csv"
    pdf_path = out_dir / "population_8directions_summary.pdf"

    df.to_csv(annotated_csv_path, index=False)

    with PdfPages(pdf_path) as pdf:
        # Pages 1-2: one depth/significance page per speed.
        for speed in speed_values:
            df_speed = df.loc[df["speed"].astype(str) == str(speed)].copy()
            fig = plt.figure(figsize=(14, 9))
            plot_significant_depth_page(fig, df_speed, speed, direction_order, direction_to_x)
            pdf.savefig(fig, bbox_inches="tight")
            png_path = out_dir / f"population_8directions_by_depth_speed_{str(speed).replace('.', 'p')}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Pages 3-4: one screen-split polar page per speed.
        for speed in speed_values:
            df_speed = df.loc[df["speed"].astype(str) == str(speed)].copy()
            fig = plt.figure(figsize=(16, 10))

            plot_screen_split_polar_page(
                fig,
                df_speed,
                speed,
                direction_order,
            )

            pdf.savefig(fig, bbox_inches="tight")

            png_path = out_dir / f"population_8directions_polar_by_screen_speed_{str(speed).replace('.', 'p')}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")

            plt.close(fig)

    print("\nSaved annotated table:")
    print(annotated_csv_path)

    print("\nSaved multipage PDF:")
    print(pdf_path)

    print("\nSignificant points per speed:")
    print(
        df.assign(is_sig=df[SIGNIFICANCE_COL_FOR_PAGES_1_2] < ALPHA)
        .groupby("speed", dropna=False)["is_sig"]
        .sum()
        .rename("n_sig_points")
        .reset_index()
    )

    print("\nRows by speed and screen_role:")
    print(
        df.groupby(["speed", "screen_role"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )


if __name__ == "__main__":
    main()
