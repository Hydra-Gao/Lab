from __future__ import annotations

"""Prepare CSV tables for the marimo Neuropixels population explorer.

Outputs
-------
1. interactive_population_points.csv
   One row per source x unit x 12-pattern condition. Contains probe-map
   coordinates, stable display jitter, response values, and all available
   p/q/significance columns.

2. interactive_12pattern_responses.csv
   One row per source x unit x 12-pattern condition x trial. Used for raw
   trial points, the selected unit's full 12-pattern response curve, and p/q
   values.

3. interactive_8direction_responses.csv
   One row per source x unit x screen x speed x direction x trial. Used for
   raw trial points, the three polar plots, three line plots, and p/q values.

This script deliberately performs all complicated cross-file merging before
marimo starts. The marimo app should only load, filter, select, and plot these
three tables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTPUT_DIR = Path(r"C:\Lab\Processing\TG963_site2_population\interactive_tables")

# One entry per recording/source. source_name must be identical for the
# corresponding 12-pattern and 8-direction datasets so a selected unit can be
# matched through unit_uid = source_name + "_" + unit_id.
#
# Required for the population + 12-pattern tables:
#   pattern_significance_csv
#   pattern_summary_csv
#   curated_units_csv
#   chanmap_files
#   curated_to_chanmap_shank
#
# For the 8-direction table, normally you only need one directory:
#   direction_analysis_dir
#
# The script automatically looks for these fixed filenames inside it:
#   unit_condition_summary.csv
#   unit_significance_summary.csv
#   unit_direction_significance.csv
#   unit_tuning_summary.csv
#   unit_trial_summary.csv
#
# When direction_analysis_dir is None, the script automatically uses the parent
# folder of pattern_significance_csv. This matches the current pipeline layout
# where the 8-direction and 12-pattern output CSVs live in the same folder.
# Individual direction_*_csv paths are still supported as optional overrides.
#
# The 12-pattern raw points are also read from unit_trial_summary.csv in the
# same analysis directory. An optional pattern_trial_summary_csv entry may be
# supplied per source when that file lives elsewhere.
SOURCE_SPECS = [
    {
        "source_name": "site2_1",
        "pattern_significance_csv": Path(
            r"C:\Lab\Processing\TG963_site2-1\Output_dir_almost\analysis_TG963_site2-1_kilosort4\unit_pattern_significance.csv"
        ),
        "pattern_summary_csv": Path(
            r"C:\Lab\Processing\TG963_site2-1\Output_dir_almost\analysis_TG963_site2-1_kilosort4\unit_pattern_summary.csv"
        ),
        "curated_units_csv": Path(
            r"C:\Lab\Processing\TG963_site2-1\Output_dir_almost\analysis_TG963_site2-1_kilosort4\curated_units.csv"
        ),
        "chanmap_files": [
            Path(
                r"C:\Lab\Processing\TG963_site2-1\Cb_2026_07_14_site2_1_g0_t0.imec0.ap_kilosortChanMap.mat"
            )
        ],
        "curated_to_chanmap_shank": {0: 3, 1: 4},
    
        # None = automatically use pattern_significance_csv.parent.
        # Set a separate folder here only when 8-direction outputs live elsewhere.
        "direction_analysis_dir": None,
    },
    {
        "source_name": "site2_2",
        "pattern_significance_csv": Path(
            r"C:\Lab\Processing\TG963_site2-2\Output_dir_almost\analysis_TG963_site2-2_kilosort4\unit_pattern_significance.csv"
        ),
        "pattern_summary_csv": Path(
            r"C:\Lab\Processing\TG963_site2-2\Output_dir_almost\analysis_TG963_site2-2_kilosort4\unit_pattern_summary.csv"
        ),
        "curated_units_csv": Path(
            r"C:\Lab\Processing\TG963_site2-2\Output_dir_almost\analysis_TG963_site2-2_kilosort4\curated_units.csv"
        ),
        "chanmap_files": [
            Path(
                r"C:\Lab\Processing\TG963_site2-2\Cb_2026_07_14_site2_2_g0_t0.imec0.ap_kilosortChanMap.mat"
            )
        ],
        "curated_to_chanmap_shank": {1: 1, 2: 2},
    
        # None = automatically use pattern_significance_csv.parent.
        # Set a separate folder here only when 8-direction outputs live elsewhere.
        "direction_analysis_dir": None,
    },
    # {
    #     "source_name": "site1_1",
    #     "pattern_significance_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\unit_pattern_significance.csv"
    #     ),
    #     "pattern_summary_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\unit_pattern_summary.csv"
    #     ),
    #     "curated_units_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-1\Output_dir_almost\analysis_TG963_site1-1_kilosort4\curated_units.csv"
    #     ),
    #     "chanmap_files": [
    #         Path(
    #             r"C:\Lab\Processing\TG963_site1-1\Cb_2026_07_14_1site_1_correct_g0_t0.imec0.ap_kilosortChanMap.mat"
    #         )
    #     ],
    #     "curated_to_chanmap_shank": {4: 4},

    #     # None = automatically use pattern_significance_csv.parent.
    #     # Set a separate folder here only when 8-direction outputs live elsewhere.
    #     "direction_analysis_dir": None,
    # },
    # {
    #     "source_name": "site1_2",
    #     "pattern_significance_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\unit_pattern_significance.csv"
    #     ),
    #     "pattern_summary_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\unit_pattern_summary.csv"
    #     ),
    #     "curated_units_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-2\Output_dir_almost\analysis_TG963_site1-2_kilosort4\curated_units.csv"
    #     ),
    #     "chanmap_files": [
    #         Path(
    #             r"C:\Lab\Processing\TG963_site1-2\Cb_2026_07_14_1site_2_g0_t0.imec0.ap_kilosortChanMap.mat"
    #         )
    #     ],
    #     "curated_to_chanmap_shank": {0: 1, 1: 2},
    #     "direction_analysis_dir": None,
    # },
    # {
    #     "source_name": "site1_3",
    #     "pattern_significance_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\unit_pattern_significance.csv"
    #     ),
    #     "pattern_summary_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\unit_pattern_summary.csv"
    #     ),
    #     "curated_units_csv": Path(
    #         r"C:\Lab\Processing\TG963_site1-3\Output_dir_almost\analysis_TG963_site1-3_kilosort4\curated_units.csv"
    #     ),
    #     "chanmap_files": [
    #         Path(
    #             r"C:\Lab\Processing\TG963_site1-3\Cb_2026_07_14_1site_3_g0_t0.imec0.ap_kilosortChanMap.mat"
    #         )
    #     ],
    #     "curated_to_chanmap_shank": {0: 3},
    #     "direction_analysis_dir": None,
    # },
]

PATTERN_ORDER = [
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

SCREEN_ORDER = ["left", "front", "right"]
DIRECTION_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]

FRACTION_CHANGE_EPS = 1e-9
SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_SIGNIFICANCE_COLUMNS = [
    "p_motion_specific_two_sided",
    "p_ttest_two_sided",
]

# Stable jitter is generated once here, not every time the marimo page opens.
JITTER_SEED = 42
JITTER_STD_UM = 3.0


# =============================================================================
# BASIC HELPERS
# =============================================================================


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def read_required_csv(path_value: object, label: str) -> pd.DataFrame:
    if path_value is None:
        raise ValueError(f"Missing required path for {label}")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path_value: object, label: str) -> pd.DataFrame | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value)
    if not path.exists():
        print(f"WARNING: optional {label} not found; skipping: {path}")
        return None
    return pd.read_csv(path)


def normalize_unit_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "unit_id" in df.columns:
        # String normalization prevents int-versus-string merge failures while
        # preserving labels such as "12.1" or nonnumeric IDs.
        df["unit_id"] = df["unit_id"].astype(str).str.strip()
    return df


def add_source_keys(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = normalize_unit_id(df)
    df["source_name"] = str(source_name)
    require_columns(df, ["unit_id"], f"{source_name} table")
    df["unit_uid"] = df["source_name"] + "_" + df["unit_id"]
    return df


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((c for c in candidates if c in available), None)


def numeric_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    col = first_existing(df.columns, candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def ordered_index(values: pd.Series, preferred_order: list) -> pd.Series:
    mapping = {str(v): i for i, v in enumerate(preferred_order)}
    text = values.astype(str)
    extras = sorted(v for v in text.dropna().unique() if v not in mapping)
    mapping.update({v: len(mapping) + i for i, v in enumerate(extras)})
    return text.map(mapping).astype("Int64")


def suffix_nonkeys(df: pd.DataFrame, keys: list[str], suffix: str) -> pd.DataFrame:
    rename = {c: f"{c}{suffix}" for c in df.columns if c not in keys}
    return df.rename(columns=rename)


def assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicate_mask = df.duplicated(keys, keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, keys].head(10)
        raise ValueError(
            f"{name} is not unique on {keys}. Example duplicate keys:\n"
            f"{examples.to_string(index=False)}"
        )


def compute_sem_from_trials(
    trial_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Compute trial-level mean/SEM columns when an optional trial table exists."""
    value_cols = [
        c
        for c in ["baseline_fr", "moving_fr", "moving_minus_baseline"]
        if c in trial_df.columns
    ]
    if not value_cols:
        return pd.DataFrame(columns=group_cols)

    agg_spec: dict[str, tuple[str, str]] = {}
    for col in value_cols:
        agg_spec[f"{col}_trial_mean"] = (col, "mean")
        agg_spec[f"{col}_trial_sem"] = (col, "sem")
    if "trial_id" in trial_df.columns:
        agg_spec["n_trials_from_trial_table"] = ("trial_id", "nunique")

    return trial_df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


# =============================================================================
# CHANNEL MAP AND UNIT COORDINATES
# =============================================================================


def _flatten_numeric_array(value: object) -> np.ndarray:
    return np.ravel(np.asarray(value))


def load_chanmap_mat(mat_path: Path) -> pd.DataFrame:
    if not mat_path.exists():
        raise FileNotFoundError(f"Channel map not found: {mat_path}")

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if "chanMap0ind" in mat:
        chan_map = _flatten_numeric_array(mat["chanMap0ind"]).astype(int)
    elif "chanMap" in mat:
        chan_map = _flatten_numeric_array(mat["chanMap"]).astype(int)
        if len(chan_map) and np.min(chan_map) == 1:
            chan_map = chan_map - 1
    else:
        raise KeyError(f"{mat_path.name}: missing chanMap/chanMap0ind")

    n = len(chan_map)
    if "xcoords" not in mat or "ycoords" not in mat:
        raise KeyError(f"{mat_path.name}: missing xcoords/ycoords")

    xcoords = _flatten_numeric_array(mat["xcoords"]).astype(float)
    ycoords = _flatten_numeric_array(mat["ycoords"]).astype(float)
    if len(xcoords) != n or len(ycoords) != n:
        raise ValueError(f"{mat_path.name}: coordinate length mismatch")

    if "kcoords" in mat:
        kcoords = _flatten_numeric_array(mat["kcoords"]).astype(int)
        if len(kcoords) != n:
            kcoords = np.ones(n, dtype=int)
    else:
        kcoords = np.ones(n, dtype=int)

    if "connected" in mat:
        connected = _flatten_numeric_array(mat["connected"])
        if len(connected) != n:
            connected = np.ones(n, dtype=bool)
        connected = connected.astype(bool)
    else:
        connected = np.ones(n, dtype=bool)

    return pd.DataFrame(
        {
            "map_channel_index": chan_map,
            "x_um": xcoords,
            "y_um": ycoords,
            "shank_local": kcoords,
            "connected": connected,
            "map_file": mat_path.name,
        }
    )


def prepare_units_with_coordinates(spec: dict) -> pd.DataFrame:
    source_name = str(spec["source_name"])
    units = add_source_keys(
        read_required_csv(spec["curated_units_csv"], f"{source_name} curated_units"),
        source_name,
    )
    require_columns(units, ["best_channel", "shank_id"], f"{source_name} curated_units")

    shank_map = {
        int(curated_id): int(chanmap_id)
        for curated_id, chanmap_id in spec.get("curated_to_chanmap_shank", {}).items()
    }
    units["shank_id"] = pd.to_numeric(units["shank_id"], errors="coerce")
    units["shank_id_for_chanmap"] = units["shank_id"].map(shank_map)

    missing_mapping = units["shank_id"].notna() & units["shank_id_for_chanmap"].isna()
    if missing_mapping.any():
        missing = sorted(units.loc[missing_mapping, "shank_id"].astype(int).unique())
        raise ValueError(
            f"{source_name}: curated shank IDs missing from "
            f"curated_to_chanmap_shank: {missing}; mapping={shank_map}"
        )

    channel_tables = [load_chanmap_mat(Path(p)) for p in spec["chanmap_files"]]
    channels = pd.concat(channel_tables, ignore_index=True)
    channels["source_name"] = source_name

    units["best_channel"] = pd.to_numeric(units["best_channel"], errors="coerce")
    channels["map_channel_index"] = pd.to_numeric(
        channels["map_channel_index"], errors="coerce"
    )
    channels["shank_local"] = pd.to_numeric(channels["shank_local"], errors="coerce")

    assert_unique(
        channels,
        ["source_name", "map_channel_index", "shank_local"],
        f"{source_name} channel map",
    )

    units = units.merge(
        channels.rename(columns={"x_um": "chanmap_x_um", "y_um": "chanmap_y_um"}),
        left_on=["source_name", "best_channel", "shank_id_for_chanmap"],
        right_on=["source_name", "map_channel_index", "shank_local"],
        how="left",
        validate="many_to_one",
    )
    units["x_um"] = units["chanmap_x_um"]
    units["y_um"] = units["chanmap_y_um"]

    missing_coords = units["x_um"].isna() | units["y_um"].isna()
    print(
        f"{source_name}: units={len(units)}, "
        f"matched coordinates={(~missing_coords).sum()}, missing={missing_coords.sum()}"
    )
    if missing_coords.any():
        examples = units.loc[
            missing_coords,
            ["unit_id", "best_channel", "shank_id", "shank_id_for_chanmap"],
        ].head(10)
        print("Missing-coordinate examples:\n", examples.to_string(index=False))

    return units


# =============================================================================
# 12-PATTERN TABLES
# =============================================================================


def compute_fraction_change(df: pd.DataFrame) -> pd.Series:
    baseline = numeric_series(df, ["baseline_fr_mean", "baseline_fr"])
    moving = numeric_series(df, ["moving_fr_mean", "moving_fr"])
    return (moving - baseline) / (baseline + FRACTION_CHANGE_EPS)


def add_significance_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    usable = [c for c in DEFAULT_SIGNIFICANCE_COLUMNS if c in df.columns]
    if usable:
        numeric = df[usable].apply(pd.to_numeric, errors="coerce")
        df["passes_default_significance"] = numeric.lt(SIGNIFICANCE_THRESHOLD).any(axis=1)
        df["min_default_p_value"] = numeric.min(axis=1, skipna=True)
    else:
        df["passes_default_significance"] = False
        df["min_default_p_value"] = np.nan

    p_cols = [c for c in df.columns if c.startswith("p_")]
    q_cols = [c for c in df.columns if c.startswith("q_")]
    if p_cols:
        df["min_p_value_any"] = df[p_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
    else:
        df["min_p_value_any"] = np.nan
    if q_cols:
        df["min_q_value_any"] = df[q_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
    else:
        df["min_q_value_any"] = np.nan
    return df


def add_stable_jitter(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable display coordinates separately within every pattern panel."""
    df = df.copy()
    df["x_plot_um"] = pd.to_numeric(df["x_um"], errors="coerce")
    df["y_plot_um"] = pd.to_numeric(df["y_um"], errors="coerce")

    rng = np.random.default_rng(JITTER_SEED)
    # Sort first so jitter assignment is reproducible even if CSV row order changes.
    order_cols = [c for c in ["source_name", "pattern_order", "pattern", "unit_uid"] if c in df]
    sorted_index = df.sort_values(order_cols, kind="stable").index
    valid = df.loc[sorted_index, ["x_um", "y_um"]].notna().all(axis=1)
    valid_index = valid.index[valid]
    df.loc[valid_index, "x_plot_um"] = (
        pd.to_numeric(df.loc[valid_index, "x_um"], errors="coerce")
        + rng.normal(0, JITTER_STD_UM, size=len(valid_index))
    )
    df.loc[valid_index, "y_plot_um"] = (
        pd.to_numeric(df.loc[valid_index, "y_um"], errors="coerce")
        + rng.normal(0, JITTER_STD_UM, size=len(valid_index))
    )
    return df


def build_12pattern_source(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a summary-level population table and trial-level pattern table.

    Returns
    -------
    population
        One row per source x unit x pattern. This remains summary-level because
        each clickable probe-map point represents one unit-condition response.

    response_trials
        One row per source x unit x pattern x trial. Pattern-level summary,
        coordinate, and p/q columns are repeated onto every matching trial row.
        This lets marimo draw raw points without performing runtime merges.
    """
    source_name = str(spec["source_name"])
    sig = add_source_keys(
        read_required_csv(
            spec["pattern_significance_csv"],
            f"{source_name} unit_pattern_significance",
        ),
        source_name,
    )
    summary = add_source_keys(
        read_required_csv(
            spec["pattern_summary_csv"],
            f"{source_name} unit_pattern_summary",
        ),
        source_name,
    )
    units = prepare_units_with_coordinates(spec)

    require_columns(sig, ["pattern"], f"{source_name} pattern significance")
    require_columns(summary, ["pattern"], f"{source_name} pattern summary")

    summary_keys = ["source_name", "unit_id", "unit_uid", "pattern"]

    # Include speed in the summary key only when both summary/significance files
    # contain it. The current experiment uses one speed, but this keeps the
    # preparation script compatible with future multi-speed pattern datasets.
    speed_col = first_existing(sig.columns, ["speed_deg_per_sec", "speed"])
    summary_speed_col = first_existing(
        summary.columns,
        ["speed_deg_per_sec", "speed"],
    )
    if speed_col and summary_speed_col:
        if speed_col != "speed_value":
            sig = sig.rename(columns={speed_col: "speed_value"})
        if summary_speed_col != "speed_value":
            summary = summary.rename(columns={summary_speed_col: "speed_value"})
        summary_keys.append("speed_value")

    assert_unique(sig, summary_keys, f"{source_name} pattern significance")
    assert_unique(summary, summary_keys, f"{source_name} pattern summary")

    # Keep the canonical unit_pattern_summary version for overlapping response
    # columns; significance-only columns are then added from SB06b.
    overlap = [
        c for c in summary.columns
        if c in sig.columns and c not in summary_keys
    ]
    sig_for_merge = sig.drop(columns=overlap)
    response_summary = summary.merge(
        sig_for_merge,
        on=summary_keys,
        how="outer",
        validate="one_to_one",
    )

    unit_metadata_cols = [
        c
        for c in units.columns
        if c not in {"map_channel_index", "shank_local", "connected", "map_file"}
    ]
    response_summary = response_summary.merge(
        units[unit_metadata_cols],
        on=["source_name", "unit_id", "unit_uid"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_unit"),
    )

    # Use the valid curated-unit/channel-map coordinates even when the upstream
    # pattern tables already contain empty x_um/y_um columns.
    if "x_um_unit" in response_summary.columns:
        response_summary["x_um"] = pd.to_numeric(
            response_summary["x_um_unit"],
            errors="coerce",
        )
    elif "chanmap_x_um" in response_summary.columns:
        response_summary["x_um"] = pd.to_numeric(
            response_summary["chanmap_x_um"],
            errors="coerce",
        )

    if "y_um_unit" in response_summary.columns:
        response_summary["y_um"] = pd.to_numeric(
            response_summary["y_um_unit"],
            errors="coerce",
        )
    elif "chanmap_y_um" in response_summary.columns:
        response_summary["y_um"] = pd.to_numeric(
            response_summary["chanmap_y_um"],
            errors="coerce",
        )

    response_summary["pattern_order"] = ordered_index(
        response_summary["pattern"],
        PATTERN_ORDER,
    )
    response_summary["fraction_change"] = compute_fraction_change(response_summary)
    response_summary = add_significance_flags(response_summary)

    # The population table remains one row per unit x pattern.
    population = add_stable_jitter(response_summary.copy())
    population = population.sort_values(
        ["pattern_order", "source_name", "unit_uid"],
        kind="stable",
    ).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Trial-level 12-pattern response table
    # ---------------------------------------------------------------------
    explicit_trial_path = spec.get("pattern_trial_summary_csv")
    if explicit_trial_path is not None and str(explicit_trial_path).strip() != "":
        trial_path = Path(explicit_trial_path)
    else:
        trial_path = Path(spec["pattern_significance_csv"]).parent / "unit_trial_summary.csv"

    trial = read_required_csv(
        trial_path,
        f"{source_name} 12-pattern unit_trial_summary",
    )
    trial = add_source_keys(trial, source_name)
    require_columns(
        trial,
        ["pattern", "trial_id"],
        f"{source_name} 12-pattern trial summary",
    )

    # unit_trial_summary can contain both 8-direction and 12-pattern rows.
    # A valid pattern label is the least ambiguous way to retain only pattern
    # trials without relying on experiment-specific trial_kind spelling.
    trial["pattern"] = trial["pattern"].astype("string").str.strip()
    trial = trial[
        trial["pattern"].notna()
        & trial["pattern"].isin(PATTERN_ORDER)
    ].copy()

    if trial.empty:
        raise ValueError(
            f"{source_name}: no rows with recognized 12-pattern labels were "
            f"found in {trial_path}"
        )

    trial_keys = [
        "source_name",
        "unit_id",
        "unit_uid",
        "pattern",
        "trial_id",
    ]

    # Preserve speed in the trial key when both sides expose it.
    trial_speed_col = first_existing(
        trial.columns,
        ["speed_value", "speed_deg_per_sec", "speed"],
    )
    if "speed_value" in summary_keys and trial_speed_col is not None:
        if trial_speed_col != "speed_value":
            trial = trial.rename(columns={trial_speed_col: "speed_value"})
        trial_keys.append("speed_value")

    assert_unique(trial, trial_keys, f"{source_name} 12-pattern trial summary")

    # Canonical trial values used by the marimo plot.
    trial["baseline_fr_trial"] = numeric_series(
        trial,
        ["baseline_fr", "baseline_window_fr", "static_fr"],
    )
    trial["moving_fr_trial"] = numeric_series(
        trial,
        ["moving_fr"],
    )
    trial["response_trial"] = numeric_series(
        trial,
        ["moving_minus_baseline", "moving_fr_minus_baseline"],
    )
    missing_response = trial["response_trial"].isna()
    trial.loc[missing_response, "response_trial"] = (
        trial.loc[missing_response, "moving_fr_trial"]
        - trial.loc[missing_response, "baseline_fr_trial"]
    )
    trial["fraction_change_trial"] = (
        trial["moving_fr_trial"] - trial["baseline_fr_trial"]
    ) / (trial["baseline_fr_trial"] + FRACTION_CHANGE_EPS)

    # Merge one summary row onto each trial. Only overlapping non-key summary
    # columns receive "_summary"; trial-level baseline/moving/response fields
    # remain unchanged and therefore cannot be mistaken for raw observations.
    merge_keys = ["source_name", "unit_id", "unit_uid", "pattern"]
    if "speed_value" in summary_keys and "speed_value" in trial.columns:
        merge_keys.append("speed_value")

    if merge_keys == ["source_name", "unit_id", "unit_uid", "pattern"]:
        assert_unique(
            response_summary,
            merge_keys,
            f"{source_name} pattern summary for trial merge",
        )

    response_trials = trial.merge(
        response_summary,
        on=merge_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_summary"),
    )

    # Re-establish canonical order after the merge.
    response_trials["pattern_order"] = ordered_index(
        response_trials["pattern"],
        PATTERN_ORDER,
    )

    response_trials = response_trials.sort_values(
        [
            "source_name",
            "unit_uid",
            "pattern_order",
            "pattern",
            "trial_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return population, response_trials


# =============================================================================
# 8-DIRECTION TABLE
# =============================================================================


DIRECTION_AUTO_FILENAMES = {
    "direction_condition_summary_csv": "unit_condition_summary.csv",
    "direction_significance_summary_csv": "unit_significance_summary.csv",
    "direction_significance_csv": "unit_direction_significance.csv",
    "direction_tuning_summary_csv": "unit_tuning_summary.csv",
    "direction_trial_summary_csv": "unit_trial_summary.csv",
}


def resolve_direction_paths(spec: dict) -> dict[str, Path | None]:
    """Resolve 8-direction files from one analysis directory.

    Resolution order for each file:
      1. An explicitly configured direction_*_csv path, when provided.
      2. direction_analysis_dir / expected filename.
      3. pattern_significance_csv.parent / expected filename.

    The condition summary is required to build the 8-direction table. The other
    four files are optional and are returned as paths even when absent so the
    existing optional-file reader can print a clear warning.
    """
    explicit_dir = spec.get("direction_analysis_dir")

    if explicit_dir is not None and str(explicit_dir).strip() != "":
        analysis_dir = Path(explicit_dir)
    else:
        pattern_sig = spec.get("pattern_significance_csv")
        analysis_dir = Path(pattern_sig).parent if pattern_sig is not None else None

    resolved: dict[str, Path | None] = {}

    for config_key, filename in DIRECTION_AUTO_FILENAMES.items():
        explicit_path = spec.get(config_key)
        if explicit_path is not None and str(explicit_path).strip() != "":
            resolved[config_key] = Path(explicit_path)
        elif analysis_dir is not None:
            resolved[config_key] = analysis_dir / filename
        else:
            resolved[config_key] = None

    source_name = str(spec.get("source_name", "unknown"))
    print(f"{source_name}: 8-direction analysis directory = {analysis_dir}")
    for config_key, path in resolved.items():
        status = "FOUND" if path is not None and path.exists() else "missing"
        print(f"  {config_key}: {status} | {path}")

    return resolved


def normalize_direction_keys(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = add_source_keys(df, source_name)
    # Current SB05/SB06 outputs use active_screen_role.
    # Normalize it to the common internal name screen_role.
    if "screen_role" not in df.columns:
        if "active_screen_role" in df.columns:
            df["screen_role"] = df["active_screen_role"]
        else:
            df["screen_role"] = "unknown"
    if "speed" not in df.columns:
        df["speed"] = "unknown"
    df["screen_role"] = df["screen_role"].astype(str).str.strip().str.lower()
    df["speed"] = df["speed"].astype(str).str.strip()
    if "direction" in df.columns:
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360
        df["direction"] = df["direction"].round(6)
    return df


def build_8direction_source(spec: dict) -> pd.DataFrame | None:
    """Build a trial-level 8-direction table and attach summary statistics.

    Output grain:
        source x unit x screen x speed x direction x trial

    The trial table is the plotting source of truth. Direction-, screen-, and
    tuning-level results are repeated onto each matching trial row so the
    marimo app can show raw points and p/q values without runtime merges.

    File paths are resolved through resolve_direction_paths(), preserving the
    convenient direction_analysis_dir / automatic-parent-directory behavior.
    """
    source_name = str(spec["source_name"])
    direction_paths = resolve_direction_paths(spec)

    trial = read_optional_csv(
        direction_paths["direction_trial_summary_csv"],
        f"{source_name} unit_trial_summary",
    )
    if trial is None:
        print(f"{source_name}: no 8-direction trial table found; skipping")
        return None

    trial = normalize_direction_keys(trial, source_name)
    require_columns(
        trial,
        ["screen_role", "speed", "direction", "trial_id"],
        f"{source_name} 8-direction trial summary",
    )

    keys_trial = [
        "source_name",
        "unit_id",
        "unit_uid",
        "screen_role",
        "speed",
        "direction",
        "trial_id",
    ]
    keys_dir = [
        "source_name",
        "unit_id",
        "unit_uid",
        "screen_role",
        "speed",
        "direction",
    ]
    keys_screen = [
        "source_name",
        "unit_id",
        "unit_uid",
        "screen_role",
        "speed",
    ]

    assert_unique(trial, keys_trial, f"{source_name} 8-direction trial summary")
    result = trial.copy()

    # Optional direction-condition summary. Trial-level baseline_fr/moving_fr
    # remain the canonical plotting values; summary columns receive a suffix.
    condition = read_optional_csv(
        direction_paths["direction_condition_summary_csv"],
        f"{source_name} unit_condition_summary",
    )
    if condition is not None:
        condition = normalize_direction_keys(condition, source_name)
        require_columns(
            condition,
            ["screen_role", "speed", "direction"],
            f"{source_name} 8-direction condition summary",
        )
        assert_unique(condition, keys_dir, f"{source_name} 8-direction condition summary")
        condition = suffix_nonkeys(condition, keys_dir, "_condition")
        result = result.merge(
            condition,
            on=keys_dir,
            how="left",
            validate="many_to_one",
        )

    dir_sig = read_optional_csv(
        direction_paths["direction_significance_csv"],
        f"{source_name} unit_direction_significance",
    )
    if dir_sig is not None:
        dir_sig = normalize_direction_keys(dir_sig, source_name)
        assert_unique(dir_sig, keys_dir, f"{source_name} direction significance")
        dir_sig = suffix_nonkeys(dir_sig, keys_dir, "_direction_sig")
        result = result.merge(
            dir_sig,
            on=keys_dir,
            how="left",
            validate="many_to_one",
        )

    screen_sig = read_optional_csv(
        direction_paths["direction_significance_summary_csv"],
        f"{source_name} unit_significance_summary",
    )
    if screen_sig is not None:
        screen_sig = normalize_direction_keys(screen_sig, source_name)
        assert_unique(screen_sig, keys_screen, f"{source_name} screen significance")
        screen_sig = suffix_nonkeys(screen_sig, keys_screen, "_screen")
        result = result.merge(
            screen_sig,
            on=keys_screen,
            how="left",
            validate="many_to_one",
        )

    tuning = read_optional_csv(
        direction_paths["direction_tuning_summary_csv"],
        f"{source_name} unit_tuning_summary",
    )
    if tuning is not None:
        tuning = normalize_direction_keys(tuning, source_name)
        assert_unique(tuning, keys_screen, f"{source_name} tuning summary")
        tuning = suffix_nonkeys(tuning, keys_screen, "_tuning")
        result = result.merge(
            tuning,
            on=keys_screen,
            how="left",
            validate="many_to_one",
        )

    result["screen_order"] = ordered_index(result["screen_role"], SCREEN_ORDER)
    direction_map = {float(d): i for i, d in enumerate(DIRECTION_ORDER)}
    result["direction_order"] = pd.to_numeric(
        result["direction"],
        errors="coerce",
    ).map(direction_map).astype("Int64")

    # Canonical trial-level columns consumed by the marimo app.
    result["baseline_fr_trial"] = numeric_series(
        result,
        ["baseline_fr", "baseline_window_fr", "static_fr"],
    )
    result["moving_fr_trial"] = numeric_series(
        result,
        ["moving_fr"],
    )
    result["response_trial"] = numeric_series(
        result,
        ["moving_minus_baseline", "moving_fr_minus_baseline"],
    )

    missing_response = result["response_trial"].isna()
    result.loc[missing_response, "response_trial"] = (
        result.loc[missing_response, "moving_fr_trial"]
        - result.loc[missing_response, "baseline_fr_trial"]
    )

    result["fraction_change_trial"] = (
        result["moving_fr_trial"] - result["baseline_fr_trial"]
    ) / (result["baseline_fr_trial"] + FRACTION_CHANGE_EPS)

    return result.sort_values(
        [
            "source_name",
            "unit_uid",
            "screen_order",
            "speed",
            "direction_order",
            "direction",
            "trial_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# OUTPUT VALIDATION AND MAIN
# =============================================================================


def report_table(df: pd.DataFrame, name: str, keys: list[str]) -> None:
    print(f"\n{name}")
    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns):,}")
    if "unit_uid" in df.columns:
        print(f"  unique units: {df['unit_uid'].nunique(dropna=True):,}")
    if "source_name" in df.columns:
        print("  rows by source:")
        print(df.groupby("source_name", dropna=False).size().to_string())
    assert_unique(df, keys, name)


def main() -> None:
    print("===== Prepare interactive Neuropixels tables =====")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    population_frames: list[pd.DataFrame] = []
    pattern_frames: list[pd.DataFrame] = []
    direction_frames: list[pd.DataFrame] = []

    for spec in SOURCE_SPECS:
        source_name = str(spec["source_name"])
        print(f"\n===== Source: {source_name} =====")

        population, pattern_response = build_12pattern_source(spec)
        population_frames.append(population)
        pattern_frames.append(pattern_response)

        direction_response = build_8direction_source(spec)
        if direction_response is not None:
            direction_frames.append(direction_response)

    population_all = pd.concat(population_frames, ignore_index=True)
    pattern_all = pd.concat(pattern_frames, ignore_index=True)
    direction_all = (
        pd.concat(direction_frames, ignore_index=True)
        if direction_frames
        else pd.DataFrame(
            columns=[
                "source_name",
                "unit_id",
                "unit_uid",
                "screen_role",
                "speed",
                "direction",
                "trial_id",
                "screen_order",
                "direction_order",
                "baseline_fr_trial",
                "moving_fr_trial",
                "response_trial",
                "fraction_change_trial",
            ]
        )
    )

    pop_keys = ["source_name", "unit_id", "unit_uid", "pattern"]
    pat_keys = [
        "source_name",
        "unit_id",
        "unit_uid",
        "pattern",
        "trial_id",
    ]
    if "speed_value" in population_all.columns:
        pop_keys.append("speed_value")
    if "speed_value" in pattern_all.columns:
        pat_keys.append("speed_value")
    dir_keys = [
        "source_name",
        "unit_id",
        "unit_uid",
        "screen_role",
        "speed",
        "direction",
        "trial_id",
    ]

    report_table(population_all, "interactive_population_points", pop_keys)
    report_table(pattern_all, "interactive_12pattern_responses", pat_keys)
    report_table(direction_all, "interactive_8direction_responses", dir_keys)

    output_paths = {
        "population": OUTPUT_DIR / "interactive_population_points.csv",
        "patterns": OUTPUT_DIR / "interactive_12pattern_responses.csv",
        "directions": OUTPUT_DIR / "interactive_8direction_responses.csv",
    }
    population_all.to_csv(output_paths["population"], index=False)
    pattern_all.to_csv(output_paths["patterns"], index=False)
    direction_all.to_csv(output_paths["directions"], index=False)

    print("\n===== Saved =====")
    for label, path in output_paths.items():
        print(f"{label:>12}: {path}")

    if direction_all.empty:
        print(
            "\nNOTE: the 8-direction CSV contains headers only because no "
            "direction_trial_summary_csv files were found. Check "
            "direction_analysis_dir or the automatic analysis-directory paths and rerun."
        )


if __name__ == "__main__":
    main()
