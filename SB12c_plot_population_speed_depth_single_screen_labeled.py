# SB12c_plot_population_speed_depth_single_screen.py
#
# Population plots for single-screen 4-direction / 6-speed data.
#
# Expected inputs in ANALYSIS_OUTPUT_DIR:
#   curated_units.csv
#   unit_condition_summary.csv
#   unit_direction_significance.csv
#
# Main output:
#   population_plots_speed_depth/population_speed_depth_summary.pdf
#
# Pages:
#   Page 1:
#       Population depth-vs-speed scatter
#       x = speed
#       y = depth
#       color = effect ratio = (moving - baseline) / baseline
#       marker = direction
#
#   Page 2+:
#       One population polar page per speed
#       Left polar: effect ratio by direction
#       Right polar: baseline/static FR vs moving FR by direction

from __future__ import annotations

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

# Use "p_motion_baseline_two_sided" for uncorrected significance,
# or "q_motion_baseline_direction" for FDR-corrected direction-level significance.
P_COL = "p_motion_baseline_two_sided"

SIGNIFICANCE_COL_FOR_PAGES_1_2 = P_COL

EPS_BASELINE_FR = 1e-6

DIRECTION_ORDER = [0, 90, 180, 270]

SPEED_ORDER = [
    "speed_1_dps",
    "speed_4_dps",
    "speed_16_dps",
    "speed_64_dps",
    "speed_128_dps",
    "speed_256_dps",
]

POINT_SIZE = 110
LABEL_FONT_SIZE = 7
SHOW_UNIT_LABELS = True
RAW_POLAR_POINT_SIZE = 24
MEAN_POLAR_LINEWIDTH = 2.2

Y_COLLISION_THRESHOLD_UM = 4.0

# Main depth-vs-speed page:
ONLY_SIGNIFICANT_MAIN_PAGE = True

# For the main scatter, directions are separated within each speed bin.
DIRECTION_OFFSET_MAP = {
    0.0: -0.18,
    90.0: -0.06,
    180.0: 0.06,
    270.0: 0.18,
}


# =====================
# Helpers
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

    if "unit_id" in df.columns:
        df["unit_id"] = df["unit_id"].astype(str).str.strip()

    if "speed" in df.columns:
        df["speed"] = df["speed"].astype(str).str.strip()

    if "direction" in df.columns:
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360
        df["direction"] = df["direction"].round(6)

    return df


def direction_label(x: float) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))}°"
    return f"{x:g}°"


def speed_to_numeric(speed) -> float:
    if pd.isna(speed):
        return np.nan

    s = str(speed).strip()

    try:
        return float(s)
    except Exception:
        pass

    s = s.replace("speed_", "").replace("_dps", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def speed_label_clean(speed: str) -> str:
    x = speed_to_numeric(speed)

    if np.isnan(x):
        return str(speed)

    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))

    return f"{x:g}"


def speed_to_x_map(speeds: pd.Series) -> tuple[list[str], dict[str, int]]:
    observed = pd.Series(speeds).dropna().astype(str).unique().tolist()

    preferred = [s for s in SPEED_ORDER if s in observed]
    remaining = [s for s in observed if s not in preferred]
    remaining = sorted(remaining, key=speed_to_numeric)

    order = preferred + remaining
    return order, {s: i for i, s in enumerate(order)}


def get_speed_order_available(df: pd.DataFrame) -> list[str]:
    return speed_to_x_map(df["speed"])[0]


def get_direction_markers() -> dict[float, str]:
    return {
        0.0: "o",
        90.0: "s",
        180.0: "^",
        270.0: "D",
    }


def make_effect_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["effect_ratio"] = (
        df["moving_minus_baseline"]
        / (df["baseline_fr"].abs() + EPS_BASELINE_FR)
    )
    df["effect_ratio_abs"] = df["effect_ratio"].abs()
    return df


def get_polar_baseline_col(df: pd.DataFrame) -> str:
    for col in ["pooled_static_fr", "static_fr", "baseline_fr"]:
        if col in df.columns:
            return col

    raise ValueError(
        "No baseline column found. Expected one of: "
        "pooled_static_fr, static_fr, baseline_fr"
    )


def apply_collision_aware_dodge(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "depth_um",
    y_threshold_um: float = Y_COLLISION_THRESHOLD_UM,
    max_spread: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Within each x-bin, horizontally spread points with very similar depths.
    """
    rng = np.random.default_rng(seed)
    parts = []

    for x_value, sub in df.groupby(x_col, sort=False):
        sub = sub.sort_values(y_col, na_position="last").copy()
        ys = sub[y_col].to_numpy(dtype=float)

        cluster_ids = np.zeros(len(sub), dtype=int)
        cluster_id = 0

        for i in range(1, len(sub)):
            if (
                np.isfinite(ys[i])
                and np.isfinite(ys[i - 1])
                and abs(ys[i] - ys[i - 1]) <= y_threshold_um
            ):
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
                offsets += rng.uniform(-0.005, 0.005, size=n)
                rng.shuffle(offsets)

            g[f"{x_col}_plot"] = g[x_col].astype(float) + offsets
            cluster_parts.append(g)

        parts.append(
            pd.concat(cluster_parts, ignore_index=True)
            .drop(columns="_cluster_id")
        )

    if not parts:
        return df.copy()

    return pd.concat(parts, ignore_index=True)


def close_polar(theta, r):
    theta = np.asarray(theta)
    r = np.asarray(r)

    if len(theta) == 0:
        return theta, r

    return np.r_[theta, theta[0]], np.r_[r, r[0]]


def close_polar_curve(values_by_direction: pd.Series, direction_order: list[float]):
    values = np.array(
        [values_by_direction.get(float(d), np.nan) for d in direction_order],
        dtype=float,
    )
    theta = np.deg2rad(np.array(direction_order, dtype=float))

    return close_polar(theta, values)


def set_polar_format(ax, title: str) -> None:
    ax.set_title(title, pad=16)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        DIRECTION_ORDER,
        labels=[f"{d}°" for d in DIRECTION_ORDER],
    )
    ax.grid(True, alpha=0.3)



def add_unit_labels_with_adjustment(
    ax,
    df: pd.DataFrame,
    x_col: str = "speed_x_base_plot",
    y_col: str = "depth_um",
) -> None:
    """
    Add unit_id labels to the main population scatter.

    If adjustText is installed, labels are adjusted to reduce overlap.
    If not installed, labels are drawn at the point positions.
    """
    texts = []

    for _, row in df.iterrows():
        if x_col not in row.index or y_col not in row.index:
            continue

        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue

        unit_label = str(row.get("unit_id", ""))

        txt = ax.text(
            row[x_col],
            row[y_col],
            unit_label,
            fontsize=LABEL_FONT_SIZE,
            color="black",
            ha="center",
            va="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
            ),
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
            arrowprops=dict(
                arrowstyle="-",
                color="gray",
                lw=0.5,
                alpha=0.7,
            ),
        )


def add_effect_colorbar(fig, ax, norm, cmap) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("(moving - baseline) / baseline", rotation=90)


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

    # New single-screen outputs do not contain screen_role.
    require_columns(
        condition,
        [
            "unit_id",
            "speed",
            "direction",
            "baseline_fr",
            "moving_fr",
            "moving_minus_baseline",
        ],
        "unit_condition_summary.csv",
    )

    require_columns(
        direction_sig,
        [
            "unit_id",
            "speed",
            "direction",
            P_COL,
        ],
        "unit_direction_significance.csv",
    )

    require_columns(
        units,
        ["unit_id", "depth_um"],
        "curated_units.csv",
    )

    condition = normalize_key_columns(condition)
    direction_sig = normalize_key_columns(direction_sig)

    units = units.copy()
    units["unit_id"] = units["unit_id"].astype(str).str.strip()

    keep_condition_cols = [
        "unit_id",
        "speed",
        "direction",
        "baseline_fr",
        "moving_fr",
        "moving_minus_baseline",
        "pooled_static_fr",
        "static_fr",
        "speed_label",
        "speed_deg_per_sec",
        "tf_hz",
        "sf_cpd",
    ]
    keep_condition_cols = [c for c in keep_condition_cols if c in condition.columns]

    keep_sig_cols = [
        "unit_id",
        "speed",
        "direction",
        P_COL,
        "q_motion_baseline_direction",
        "mean_moving_minus_baseline",
        "is_direction_response_significant",
        "is_direction_excited",
        "is_direction_suppressed",
    ]
    keep_sig_cols = [c for c in keep_sig_cols if c in direction_sig.columns]

    df = condition[keep_condition_cols].merge(
        direction_sig[keep_sig_cols],
        on=["unit_id", "speed", "direction"],
        how="left",
        suffixes=("", "_sig"),
    )

    df = df.merge(
        units[["unit_id", "depth_um"]].drop_duplicates(),
        on="unit_id",
        how="left",
    )

    df = make_effect_ratio(df)

    good_units = units["unit_id"].dropna().unique().tolist()
    df = df.loc[df["unit_id"].isin(good_units)].copy()

    return df


# =====================
# Page 1: speed x depth scatter
# =====================

def plot_speed_depth_page(
    fig,
    df: pd.DataFrame,
    only_significant: bool = True,
) -> None:
    ax = fig.add_subplot(111)

    if only_significant:
        plot_df = df.loc[df[SIGNIFICANCE_COL_FOR_PAGES_1_2] < ALPHA].copy()
    else:
        plot_df = df.copy()

    plot_df = plot_df.dropna(subset=["speed", "direction", "depth_um"]).copy()

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No valid points to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Population responses by depth and speed")
        return

    speed_order, speed_to_x = speed_to_x_map(plot_df["speed"])
    plot_df["speed_x"] = plot_df["speed"].astype(str).map(speed_to_x)

    plot_df["speed_x_base"] = (
        plot_df["speed_x"].astype(float)
        + plot_df["direction"].map(DIRECTION_OFFSET_MAP).fillna(0.0)
    )

    plot_df = apply_collision_aware_dodge(
        plot_df,
        x_col="speed_x_base",
        y_col="depth_um",
        y_threshold_um=Y_COLLISION_THRESHOLD_UM,
        max_spread=0.03,
    )

    cmap = plt.get_cmap("turbo")

    if plot_df.empty or plot_df["effect_ratio"].dropna().empty:
        color_limit = 1.0
    else:
        color_limit = max(
            float(np.nanpercentile(plot_df["effect_ratio"].abs(), 95)),
            0.5,
        )

    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    marker_map = get_direction_markers()

    for direction in DIRECTION_ORDER:
        sub = plot_df.loc[
            np.isclose(plot_df["direction"], float(direction))
        ].copy()

        if sub.empty:
            continue

        ax.scatter(
            sub["speed_x_base_plot"],
            sub["depth_um"],
            s=POINT_SIZE,
            c=sub["effect_ratio"],
            cmap=cmap,
            norm=norm,
            marker=marker_map.get(float(direction), "o"),
            edgecolors="black",
            linewidths=0.45,
            alpha=0.92,
            zorder=3,
            label=direction_label(direction),
        )

    if SHOW_UNIT_LABELS:
        add_unit_labels_with_adjustment(
            ax,
            plot_df,
            x_col="speed_x_base_plot",
            y_col="depth_um",
        )

    ax.set_xticks(range(len(speed_order)))
    ax.set_xticklabels([speed_label_clean(s) for s in speed_order])
    ax.set_xlim(-0.6, len(speed_order) - 0.4)
    ax.set_xlabel("Speed (deg/s)")
    ax.set_ylabel("Depth (um)")

    if only_significant:
        title_prefix = "Population significant responses by depth"
        sig_text = f"{SIGNIFICANCE_COL_FOR_PAGES_1_2} < {ALPHA}"
    else:
        title_prefix = "Population responses by depth"
        sig_text = "all unit × speed × direction points"

    ax.set_title(
        f"{title_prefix}\n"
        f"{sig_text}; color = (moving - baseline) / baseline; marker = direction"
    )

    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.legend(
        title="Direction",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    add_effect_colorbar(fig, ax, norm, cmap)

    fig.tight_layout()


# =====================
# Page 2+: single-screen polar pages
# =====================

def plot_single_screen_polar_page(
    fig,
    df_speed: pd.DataFrame,
    speed: str,
    direction_order: list[float],
) -> None:
    """
    One population polar page per speed.

    Left:
        effect ratio = (moving - baseline) / baseline
        raw points = each unit × direction
        black line = population mean

    Right:
        baseline/static FR vs moving FR
        raw points = each unit × direction
        thick lines = population mean
    """
    raw = df_speed.copy()
    raw = raw.dropna(subset=["direction"])
    raw["direction"] = raw["direction"].astype(float)

    if raw.empty:
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"No valid data for speed = {speed}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    baseline_col = get_polar_baseline_col(raw)

    unit_ids = sorted(raw["unit_id"].dropna().astype(str).unique().tolist())
    cmap_units = plt.get_cmap("tab20", max(len(unit_ids), 1))
    unit_to_color = {u: cmap_units(i) for i, u in enumerate(unit_ids)}

    # -------------------------
    # Left: effect ratio polar
    # -------------------------
    ax_ratio = fig.add_subplot(1, 2, 1, projection="polar")

    ratio_mean = raw.groupby("direction")["effect_ratio"].mean()
    theta_ratio, ratio_values = close_polar_curve(ratio_mean, direction_order)

    for unit_id in unit_ids:
        sub = raw.loc[raw["unit_id"].astype(str) == unit_id].copy()
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
        f"Speed {speed_label_clean(speed)} deg/s\nFraction change",
    )

    # -------------------------
    # Right: baseline vs moving
    # -------------------------
    ax_fr = fig.add_subplot(1, 2, 2, projection="polar")

    fr_df = raw[
        ["direction", baseline_col, "moving_fr"]
    ].dropna(subset=["direction"]).copy()

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
            linewidth=1.9,
            label=f"baseline: {baseline_col}",
        )
        ax_fr.fill(theta_c, baseline_c, alpha=0.08)

        ax_fr.plot(
            theta_c,
            moving_c,
            marker="o",
            linewidth=1.9,
            label="moving FR",
        )
        ax_fr.fill(theta_c, moving_c, alpha=0.08)

    for unit_id in unit_ids:
        sub = raw.loc[raw["unit_id"].astype(str) == unit_id].copy()
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
        f"Speed {speed_label_clean(speed)} deg/s\nBaseline/static vs moving FR",
    )

    ax_fr.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        f"Population polar plots, speed = {speed_label_clean(speed)} deg/s",
        y=0.98,
        fontsize=14,
    )

    fig.text(
        0.03,
        0.50,
        "Left: fraction change\nRight: firing rate",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
    )

    fig.tight_layout(rect=(0.05, 0, 0.98, 0.94))


# =====================
# Main
# =====================

def main() -> None:
    print("===== Plot population speed-depth summary =====")

    df = load_population_table()

    direction_order = [float(d) for d in DIRECTION_ORDER]

    speed_values = get_speed_order_available(df)
    if len(speed_values) == 0:
        raise ValueError("No speed values found in merged table.")

    out_dir = ANALYSIS_OUTPUT_DIR / "population_plots_speed_depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_csv_path = out_dir / "population_speed_depth_effect_ratio_table.csv"
    pdf_path = out_dir / "population_speed_depth_summary.pdf"
    main_png_path = out_dir / "population_speed_depth_main.png"

    df.to_csv(annotated_csv_path, index=False)

    with PdfPages(pdf_path) as pdf:
        # Page 1: combined speed-vs-depth population page.
        fig = plt.figure(figsize=(14, 9))
        plot_speed_depth_page(
            fig,
            df,
            only_significant=ONLY_SIGNIFICANT_MAIN_PAGE,
        )
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(main_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Page 2+: one single-screen polar page per speed.
        for speed in speed_values:
            df_speed = df.loc[df["speed"].astype(str) == str(speed)].copy()

            fig = plt.figure(figsize=(15, 8))
            plot_single_screen_polar_page(
                fig,
                df_speed,
                speed,
                direction_order,
            )

            pdf.savefig(fig, bbox_inches="tight")

            png_path = out_dir / f"population_polar_speed_{str(speed).replace('.', 'p')}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")

            plt.close(fig)

    print("\nSaved annotated table:")
    print(annotated_csv_path)

    print("\nSaved multipage PDF:")
    print(pdf_path)

    print("\nSaved main PNG:")
    print(main_png_path)

    if SIGNIFICANCE_COL_FOR_PAGES_1_2 in df.columns:
        print("\nSignificant points total:")
        print(int((df[SIGNIFICANCE_COL_FOR_PAGES_1_2] < ALPHA).sum()))

        print("\nSignificant points by speed:")
        print(
            df.assign(is_sig=df[SIGNIFICANCE_COL_FOR_PAGES_1_2] < ALPHA)
            .groupby("speed", dropna=False)["is_sig"]
            .sum()
            .rename("n_sig_points")
            .reset_index()
        )

    print("\nRows by speed and direction:")
    print(
        df.groupby(["speed", "direction"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )


if __name__ == "__main__":
    main()
