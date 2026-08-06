from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat


# =============================================================================
# USER SETTINGS
# =============================================================================

ANALYSIS_OUTPUT_DIR = Path(r"C:\Lab\Processing\TG963_site2_population")
OUTPUT_PDF_NAME = "population_12patterns_neuropixels.pdf"

# -----------------------------------------------------------------------------
# Source specification
# One dict per recording/source.
#
# Required keys:
#   source_name
#   significance_csv
#   pattern_summary_csv
#   curated_units_csv
#   chanmap_files
#
# Optional keys:
#   curated_shank_id_offset
#       Added to curated_units.csv shank_id before matching chanmap kcoords.
#       Use 1 when curated shank_id is 0-based but chanmap kcoords is 1-based.
#       Default: 0 (already aligned).
#
# Notes:
# - chanmap_files can contain one or multiple .mat files.
# - Different map files are assumed to occupy non-overlapping x/y coordinate
#   spaces, so they are concatenated directly by their native xcoords/ycoords.
# - unit IDs from different sources may overlap. The script creates a unique
#   key: f"{source_name}_{unit_id}".
# -----------------------------------------------------------------------------
SOURCE_SPECS = [
    {
        "source_name": "site1_1",
        "significance_csv": Path(r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\unit_pattern_significance.csv"),
        "pattern_summary_csv": Path(r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\unit_pattern_summary.csv"),
        "curated_units_csv": Path(r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\curated_units.csv"),
        "chanmap_files": [Path(r"C:\Lab\Processing\TG963_site1-1\Cb_2026_07_14_1site_1_correct_g0_t0.imec0.ap_kilosortChanMap.mat")],
        "curated_to_chanmap_shank": {4: 4},
    },
    {
        "source_name": "site1_2",
        "significance_csv": Path(r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\unit_pattern_significance.csv"),
        "pattern_summary_csv": Path(r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\unit_pattern_summary.csv"),
        "curated_units_csv": Path(r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\curated_units.csv"),
        "chanmap_files": [Path(r"C:\Lab\Processing\TG963_site1-2\Cb_2026_07_14_1site_2_g0_t0.imec0.ap_kilosortChanMap.mat")],
        "curated_to_chanmap_shank": {0: 1, 1: 2},
    },
    {   
        "source_name": "site1_3",
        "significance_csv": Path(r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\unit_pattern_significance.csv"),
        "pattern_summary_csv": Path(r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\unit_pattern_summary.csv"),
        "curated_units_csv": Path(r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\curated_units.csv"),
        "chanmap_files": [Path(r"C:\Lab\Processing\TG963_site1-3\Cb_2026_07_14_1site_3_g0_t0.imec0.ap_kilosortChanMap.mat")],
        "curated_to_chanmap_shank": {0: 3},
    },
]

# -----------------------------------------------------------------------------
# Pattern layout
# Edit PAGE1_PATTERNS freely. PAGE2_PATTERNS can be set explicitly, or left as
# None to auto-fill with the remaining patterns.
# -----------------------------------------------------------------------------
PAGE1_PATTERNS = [
    "VAl",
    "VAr",
    "EXPANSION_l",
    "EXPANSION_r",
    "CONTRACTION_left",
    "CONTRACTION_right",
]

PAGE2_PATTERNS: list[str] | None = None

# -----------------------------------------------------------------------------
# Significance rule interface
#
# A unit is plotted for a pattern if ANY selected p/q column satisfies
# value < SIGNIFICANCE_THRESHOLD.
#
# Examples:
#   ["p_motion_specific_two_sided", "p_ttest_two_sided"]
#   ["q_motion_specific_global", "q_ttest_two_sided_global"]
#   ["q_motion_specific_within_pattern"]
# -----------------------------------------------------------------------------
SIGNIFICANCE_COLUMNS = [
    "p_motion_specific_two_sided",
    "p_ttest_two_sided",
]
SIGNIFICANCE_THRESHOLD = 0.05

# -----------------------------------------------------------------------------
# Fraction change settings
# Fully follow the earlier SB12 definition by editing the function
# compute_fraction_change() below if needed.
# -----------------------------------------------------------------------------
FRACTION_CHANGE_EPS = 1e-9

# -----------------------------------------------------------------------------
# Plot style settings
# -----------------------------------------------------------------------------
FIGSIZE = (14, 9)
PANELS_PER_PAGE = 6
NROWS = 2
NCOLS = 3
POINT_SIZE = 32
POINT_ALPHA = 0.95
JITTER_STD_UM = 3.0
TITLE_FONTSIZE = 11
SHOW_AXES = True
DRAW_CONNECTED_CHANNELS_BACKGROUND = True
BACKGROUND_CHANNEL_SIZE = 8
BACKGROUND_CHANNEL_ALPHA = 0.18
# COLORMAP = "coolwarm"
COLORMAP = "seismic"
BACKGROUND_CHANNEL_COLOR = "lightgray"

LABEL_TOP_FRACTION = 0.20   
LABEL_FONT_SIZE = 4
LABEL_X_OFFSET_UM = 8.0   
LABEL_Y_OFFSET_UM = 0.0
LABEL_BOX_ALPHA = 0.65

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SourceData:
    source_name: str
    significance: pd.DataFrame
    summary: pd.DataFrame
    units: pd.DataFrame
    channels: pd.DataFrame


# =============================================================================
# IO / MAP PARSING
# =============================================================================


def _flatten_numeric_array(value) -> np.ndarray:
    arr = np.asarray(value)
    return np.ravel(arr)



def load_chanmap_mat(mat_path: Path) -> pd.DataFrame:
    """Load one kilosort channel map .mat file into a channel-position table.

    Expected/handled fields when present:
        chanMap / chanMap0ind / connected / xcoords / ycoords / kcoords

    Returns columns:
        map_channel_index      # 0-based channel index used by results tables
        x_um
        y_um
        shank_local
        connected
        map_file
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    chan_map = None
    if "chanMap0ind" in mat:
        chan_map = _flatten_numeric_array(mat["chanMap0ind"]).astype(int)
    elif "chanMap" in mat:
        chan_map = _flatten_numeric_array(mat["chanMap"]).astype(int)
        # Many Kilosort chanMap files store chanMap as 1-based indices.
        if np.min(chan_map) == 1:
            chan_map = chan_map - 1
    else:
        raise KeyError(f"{mat_path.name}: missing chanMap/chanMap0ind")

    n = len(chan_map)

    if "xcoords" not in mat or "ycoords" not in mat:
        raise KeyError(f"{mat_path.name}: missing xcoords/ycoords")

    xcoords = _flatten_numeric_array(mat["xcoords"]).astype(float)
    ycoords = _flatten_numeric_array(mat["ycoords"]).astype(float)

    if len(xcoords) != n or len(ycoords) != n:
        raise ValueError(
            f"{mat_path.name}: coordinate length mismatch with chanMap "
            f"(n={n}, len(x)={len(xcoords)}, len(y)={len(ycoords)})"
        )

    if "kcoords" in mat:
        kcoords = _flatten_numeric_array(mat["kcoords"]).astype(int)
        if len(kcoords) != n:
            kcoords = np.full(n, 1, dtype=int)
    else:
        kcoords = np.full(n, 1, dtype=int)

    if "connected" in mat:
        connected = _flatten_numeric_array(mat["connected"])
        if len(connected) != n:
            connected = np.ones(n, dtype=bool)
        connected = connected.astype(bool)
    else:
        connected = np.ones(n, dtype=bool)

    df = pd.DataFrame(
        {
            "map_channel_index": chan_map,
            "x_um": xcoords,
            "y_um": ycoords,
            "shank_local": kcoords,
            "connected": connected,
            "map_file": mat_path.name,
        }
    )

    # If multiple rows share the same channel index, keep them all for now and
    # let the merge validation catch ambiguous best_channel assignments.
    return df



def load_source(spec: dict) -> SourceData:
    source_name = str(spec["source_name"])
    significance_csv = Path(spec["significance_csv"])
    pattern_summary_csv = Path(spec["pattern_summary_csv"])
    curated_units_csv = Path(spec["curated_units_csv"])
    chanmap_files = [Path(p) for p in spec["chanmap_files"]]

    significance = pd.read_csv(significance_csv)
    summary = pd.read_csv(pattern_summary_csv)
    units = pd.read_csv(curated_units_csv)


    # Convert curated shank IDs to numeric values.
    units["shank_id"] = pd.to_numeric(
        units["shank_id"],
        errors="coerce",
    )

    # Explicit per-source mapping:
    # curated_units shank_id -> chanmap kcoords/shank_local
    shank_map = spec.get("curated_to_chanmap_shank", {})

    # Normalize mapping keys and values to int.
    shank_map = {
        int(curated_id): int(chanmap_id)
        for curated_id, chanmap_id in shank_map.items()
    }

    units["shank_id_for_chanmap"] = units["shank_id"].map(shank_map)


    # Detect curated IDs that were not included in the mapping.
    missing_mask = (
        units["shank_id"].notna()
        & units["shank_id_for_chanmap"].isna()
    )

    if missing_mask.any():
        missing_shanks = sorted(
            units.loc[missing_mask, "shank_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{source_name}: curated shank IDs are missing from "
            f"curated_to_chanmap_shank: {missing_shanks}. "
            f"Current mapping: {shank_map}"
        )

    channel_tables = []
    for map_path in chanmap_files:
        channel_tables.append(load_chanmap_mat(map_path))
    channels = pd.concat(channel_tables, ignore_index=True)


    print(f"\n===== {source_name}: shank ID check =====")

    curated_shanks = sorted(
        units["shank_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    map_shanks = sorted(
        pd.to_numeric(channels["shank_local"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    aligned_shanks = sorted(
        units["shank_id_for_chanmap"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(f"curated_units.csv shank_id: {curated_shanks}")
    print(f"chanmap kcoords / shank_local: {map_shanks}")
    print(f"configured curated -> chanmap mapping: {shank_map}")
    print(f"curated shank_id after mapping: {aligned_shanks}")

    invalid_targets = sorted(set(aligned_shanks) - set(map_shanks))

    if invalid_targets:
        print(
            "WARNING: mapped curated shank IDs not found in chanmap: "
            f"{invalid_targets}"
        )
    else:
        print("Shank ID mapping targets: OK")

    
    significance["source_name"] = source_name
    summary["source_name"] = source_name
    units["source_name"] = source_name
    channels["source_name"] = source_name

    return SourceData(
        source_name=source_name,
        significance=significance,
        summary=summary,
        units=units,
        channels=channels,
    )


# =============================================================================
# MERGING / METRICS
# =============================================================================


def require_columns(df: pd.DataFrame, cols: Iterable[str], table_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")



def compute_fraction_change(df: pd.DataFrame) -> pd.Series:
    """Fraction change definition.

    This is the place to enforce full compatibility with the previous SB12
    definition. The current default is:

        (moving_fr_mean - baseline_fr_mean) / (baseline_fr_mean + eps)

    If your earlier SB12 script used a different exact formula, replace it here.
    """
    require_columns(
        df,
        ["moving_fr_mean", "baseline_fr_mean"],
        "pattern summary for fraction change",
    )
    baseline = pd.to_numeric(df["baseline_fr_mean"], errors="coerce")
    moving = pd.to_numeric(df["moving_fr_mean"], errors="coerce")
    return (moving - baseline) / (baseline + FRACTION_CHANGE_EPS)



def passes_significance(row: pd.Series) -> bool:
    for col in SIGNIFICANCE_COLUMNS:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and float(val) < SIGNIFICANCE_THRESHOLD:
                return True
    return False



def build_combined_table(sources: list[SourceData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig_frames = []
    chan_frames = []

    for src in sources:
        sig = src.significance.copy()
        summ = src.summary.copy()
        units = src.units.copy()
        chans = src.channels.copy()

        require_columns(sig, ["unit_id", "pattern"], f"{src.source_name} significance")
        require_columns(summ, ["unit_id", "pattern"], f"{src.source_name} summary")
        require_columns(
            units,
            ["unit_id", "best_channel", "shank_id", "shank_id_for_chanmap"],
            f"{src.source_name} curated_units",
        )
        require_columns(chans, ["map_channel_index", "x_um", "y_um"], f"{src.source_name} chanmap")

        sig["unit_uid"] = sig["source_name"].astype(str) + "_" + sig["unit_id"].astype(str)
        summ["unit_uid"] = summ["source_name"].astype(str) + "_" + summ["unit_id"].astype(str)
        units["unit_uid"] = units["source_name"].astype(str) + "_" + units["unit_id"].astype(str)

        merged = sig.merge(
            summ,
            on=["source_name", "unit_id", "unit_uid", "pattern"],
            how="left",
            suffixes=("", "_summary"),
        )

        merged = merged.merge(
            units,
            on=["source_name", "unit_id", "unit_uid"],
            how="left",
            suffixes=("", "_unit"),
        )

        # Merge by channel and aligned shank ID.
        # best_channel is matched directly to map_channel_index.
        # shank_id_for_chanmap is curated shank_id plus the per-source offset
        # configured in SOURCE_SPECS.
        merged["best_channel"] = pd.to_numeric(merged["best_channel"], errors="coerce")
        merged["shank_id_for_chanmap"] = pd.to_numeric(
            merged["shank_id_for_chanmap"], errors="coerce"
        )
        chans["map_channel_index"] = pd.to_numeric(chans["map_channel_index"], errors="coerce")
        chans["shank_local"] = pd.to_numeric(chans["shank_local"], errors="coerce")

        # Give channel-map coordinates explicit names so they cannot collide
        # with x_um/depth_um already present in curated_units.csv.
        chans_for_merge = chans.rename(
            columns={
                "x_um": "chanmap_x_um",
                "y_um": "chanmap_y_um",
            }
        ).copy()

        merged = merged.merge(
            chans_for_merge,
            left_on=[
                "source_name",
                "best_channel",
                "shank_id_for_chanmap",
            ],
            right_on=[
                "source_name",
                "map_channel_index",
                "shank_local",
            ],
            how="left",
            validate="many_to_one",
        )

        # These are the coordinates that should be used for plotting.
        merged["x_um"] = merged["chanmap_x_um"]
        merged["y_um"] = merged["chanmap_y_um"]

        print(
            chans.groupby("map_channel_index").size().value_counts()
        )

        merged["fraction_change"] = compute_fraction_change(merged)
        merged["passes_significance"] = merged.apply(passes_significance, axis=1)

        sig_frames.append(merged)
        chan_frames.append(chans)

    combined = pd.concat(sig_frames, ignore_index=True)
    combined_channels = pd.concat(chan_frames, ignore_index=True)

    return combined, combined_channels


# =============================================================================
# PATTERN PAGINATION
# =============================================================================


def resolve_pattern_pages(all_patterns: list[str]) -> tuple[list[str], list[str]]:
    page1 = [p for p in PAGE1_PATTERNS if p in all_patterns]

    if len(page1) != len(PAGE1_PATTERNS):
        missing = [p for p in PAGE1_PATTERNS if p not in all_patterns]
        print(f"Warning: PAGE1_PATTERNS not found in data and will be skipped: {missing}")

    if PAGE2_PATTERNS is None:
        remaining = [p for p in all_patterns if p not in page1]
        page2 = remaining[:PANELS_PER_PAGE]
    else:
        page2 = [p for p in PAGE2_PATTERNS if p in all_patterns]
        if len(page2) != len(PAGE2_PATTERNS):
            missing = [p for p in PAGE2_PATTERNS if p not in all_patterns]
            print(f"Warning: PAGE2_PATTERNS not found in data and will be skipped: {missing}")

    return page1[:PANELS_PER_PAGE], page2[:PANELS_PER_PAGE]


# =============================================================================
# PLOTTING
# =============================================================================


def jitter_positions(df: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = df["x_um"].to_numpy(dtype=float)
    y = df["y_um"].to_numpy(dtype=float)
    xj = x + rng.normal(0, JITTER_STD_UM, size=len(df))
    yj = y + rng.normal(0, JITTER_STD_UM, size=len(df))
    return xj, yj



def make_panel_title(pattern: str, df_pattern: pd.DataFrame) -> str:
    n_units = int(df_pattern["unit_uid"].nunique())
    return f"{pattern}\n(n={n_units} units)"



def get_label_row_indices(
    df_pattern: pd.DataFrame,
    top_fraction: float = LABEL_TOP_FRACTION,
) -> list[int]:

    if df_pattern.empty:
        return []

    df = df_pattern.copy()

    # Convert configured significance columns to numeric.
    available_p_cols = [
        col for col in SIGNIFICANCE_COLUMNS
        if col in df.columns
    ]

    if not available_p_cols:
        return []

    for col in available_p_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strongest significance = smallest p-value from either selected test.
    df["_label_p_value"] = df[available_p_cols].min(
        axis=1,
        skipna=True,
    )
    
    label_indices = []

    # # Positive responders: largest fraction_change
    # df_pos = df_pattern.loc[df_pattern["fraction_change"] > 0].copy()
    # if not df_pos.empty:
    #     n_pos = int(np.ceil(len(df_pos) * top_fraction))
    #     n_pos = max(n_pos, 1)
    #     pos_idx = (
    #         df_pos.nlargest(n_pos, "fraction_change")
    #         .index
    #         .tolist()
    #     )
    #     label_indices.extend(pos_idx)

    # # Negative responders: most negative fraction_change
    # df_neg = df_pattern.loc[df_pattern["fraction_change"] < 0].copy()
    # if not df_neg.empty:
    #     n_neg = int(np.ceil(len(df_neg) * top_fraction))
    #     n_neg = max(n_neg, 1)
    #     neg_idx = (
    #         df_neg.nsmallest(n_neg, "fraction_change")
    #         .index
    #         .tolist()
    #     )
    #     label_indices.extend(neg_idx)

    # Positive units, ranked by smallest p-value.
    df_pos = df.loc[
        (df["fraction_change"] > 0)
        & df["_label_p_value"].notna()
    ].copy()

    if not df_pos.empty:
        n_pos = max(
            1,
            int(np.ceil(len(df_pos) * top_fraction)),
        )

        pos_idx = (
            df_pos
            .nsmallest(n_pos, "_label_p_value")
            .index
            .tolist()
        )

        label_indices.extend(pos_idx)

    # Negative units, also ranked by smallest p-value.
    df_neg = df.loc[
        (df["fraction_change"] < 0)
        & df["_label_p_value"].notna()
    ].copy()

    if not df_neg.empty:
        n_neg = max(
            1,
            int(np.ceil(len(df_neg) * top_fraction)),
        )

        neg_idx = (
            df_neg
            .nsmallest(n_neg, "_label_p_value")
            .index
            .tolist()
        )

        label_indices.extend(neg_idx)

    return list(dict.fromkeys(label_indices))

    # # 去重并保持顺序
    # seen = set()
    # out = []
    # for idx in label_indices:
    #     if idx not in seen:
    #         seen.add(idx)
    #         out.append(idx)

    # return out



def plot_page(
    pdf: PdfPages,
    page_patterns: list[str],
    combined: pd.DataFrame,
    all_channels: pd.DataFrame,
    page_label: str,
):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
    axes = np.ravel(axes)

    for ax in axes:
        ax.set_visible(False)

    for i, pattern in enumerate(page_patterns):
        ax = axes[i]
        ax.set_visible(True)

        df_pattern = combined.loc[
            (combined["pattern"] == pattern) & (combined["passes_significance"])
        ].copy()

        if DRAW_CONNECTED_CHANNELS_BACKGROUND:
            bg = all_channels.copy()
            if "connected" in bg.columns:
                bg = bg.loc[bg["connected"]]
            ax.scatter(
                bg["x_um"],
                bg["y_um"],
                s=BACKGROUND_CHANNEL_SIZE,
                alpha=BACKGROUND_CHANNEL_ALPHA,
                linewidths=0,
                color=BACKGROUND_CHANNEL_COLOR,
            )

        if df_pattern.empty:
            ax.set_title(f"{pattern}\n(no significant units)", fontsize=TITLE_FONTSIZE)
            if SHOW_AXES:
                ax.set_xlabel("x (um)")
                ax.set_ylabel("depth / y (um)")
            continue

        # Drop rows with missing coordinates or fraction change.
        df_pattern = df_pattern.dropna(subset=["x_um", "y_um", "fraction_change"]).copy().reset_index(drop=True)
        if df_pattern.empty:
            ax.set_title(f"{pattern}\n(no plottable units)", fontsize=TITLE_FONTSIZE)
            if SHOW_AXES:
                ax.set_xlabel("x (um)")
                ax.set_ylabel("depth / y (um)")
            continue

        xj, yj = jitter_positions(df_pattern, seed=i)
        vals = df_pattern["fraction_change"].to_numpy(dtype=float)

        vmax = np.nanmax(np.abs(vals))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        sc = ax.scatter(
            xj,
            yj,
            c=vals,
            cmap=COLORMAP,
            norm=norm,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
        )

        # ---- add unit-id labels only for strongest positive/negative top 10% ----
        label_indices = get_label_row_indices(df_pattern)

        for idx in label_indices:
            row = df_pattern.iloc[idx]
            ax.text(
                xj[idx] + LABEL_X_OFFSET_UM,
                yj[idx] + LABEL_Y_OFFSET_UM,
                str(row["unit_id"]),
                fontsize=LABEL_FONT_SIZE,
                ha="left",
                va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=LABEL_BOX_ALPHA,
                ),
                clip_on=True,
            )

        ax.set_title(make_panel_title(pattern, df_pattern), fontsize=TITLE_FONTSIZE)
        if SHOW_AXES:
            ax.set_xlabel("x (um)")
            ax.set_ylabel("depth / y (um)")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("fraction change")

    fig.suptitle(
        f"12-pattern population map ({page_label})\n"
        f"Significance: any([{', '.join(SIGNIFICANCE_COLUMNS)}] < {SIGNIFICANCE_THRESHOLD})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main():
    if len(SOURCE_SPECS) == 0:
        raise ValueError(
            "SOURCE_SPECS is empty. Please specify at least one source dict."
        )

    sources = [load_source(spec) for spec in SOURCE_SPECS]
    combined, all_channels = build_combined_table(sources)

    # Keep only 12-pattern rows if mixed files are supplied.
    if "trial_kind" in combined.columns:
        before = len(combined)
        combined = combined.loc[
            combined["trial_kind"].eq("optimal_3screen_12pattern")
            | combined["trial_kind"].isna()
        ].copy()
        after = len(combined)
        if after != before:
            print(f"Filtered non-12-pattern rows: {before} -> {after}")

    patterns = [p for p in combined["pattern"].dropna().unique().tolist()]
    if len(patterns) == 0:
        raise ValueError("No pattern values were found in the combined table.")

    page1, page2 = resolve_pattern_pages(patterns)
    print("Page 1 patterns:", page1)
    print("Page 2 patterns:", page2)

    out_path = ANALYSIS_OUTPUT_DIR / OUTPUT_PDF_NAME
    with PdfPages(out_path) as pdf:
        if page1:
            plot_page(pdf, page1, combined, all_channels, page_label="Page 1")
        if page2:
            plot_page(pdf, page2, combined, all_channels, page_label="Page 2")
    
    print(f"Saved population map PDF: {out_path}")


if __name__ == "__main__":
    main()
