# Inputs from SB05NP_compute_tuning_summary.py:
#   unit_trial_summary.csv
#   unit_condition_summary.csv          optional merge source
#   unit_tuning_summary.csv             optional merge source
#   unit_pattern_summary.csv            optional merge source
#   curated_units.csv                   optional merge source
#
# Outputs:
#   unit_significance_summary.csv
#       8-direction: unit x active_screen_role
#
#   unit_direction_significance.csv
#       8-direction: unit x active_screen_role x direction
#
#   unit_pattern_significance.csv
#       12-pattern: unit x pattern
#
# This recording contains only one speed. Speed/speed_label are retained as
# metadata, but are intentionally NOT used as grouping dimensions.

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


N_PERMUTATIONS = 5000
RANDOM_SEED = 42
ALPHA = 0.05

TRIAL_KIND_8D = "single_screen_8direction"
TRIAL_KIND_12P = "optimal_3screen_12pattern"


# ============================================================================
# Generic helpers
# ============================================================================


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def first_nonnull(series: pd.Series, default=np.nan):
    s = pd.Series(series).dropna()
    return default if s.empty else s.iloc[0]


def stable_seed(*parts) -> int:
    """Deterministic but group-specific RNG seed."""
    text = "|".join(str(x) for x in parts)
    return (RANDOM_SEED + zlib.crc32(text.encode("utf-8"))) % (2**32 - 1)


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)

    valid = np.isfinite(p_values)
    p = p_values[valid]
    if len(p) == 0:
        return q_values

    order = np.argsort(p)
    ranked_p = p[order]
    m = len(ranked_p)

    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty_like(ranked_q)
    q[order] = ranked_q
    q_values[valid] = q
    return q_values


def add_groupwise_fdr(
    df: pd.DataFrame,
    p_col: str,
    q_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Apply BH correction independently inside each group."""
    df = df.copy()
    df[q_col] = np.nan

    if df.empty:
        return df

    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        df.loc[idx, q_col] = bh_fdr(df.loc[idx, p_col].to_numpy(dtype=float))

    return df


def descriptive_diff_stats(diff) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    n = len(diff)
    if n == 0:
        return {
            "n_valid_trials": 0,
            "mean_moving_minus_baseline": np.nan,
            "sd_moving_minus_baseline": np.nan,
            "sem_moving_minus_baseline": np.nan,
            "cohen_dz_moving_minus_baseline": np.nan,
            "n_pos": 0,
            "n_neg": 0,
            "n_zero": 0,
        }

    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n >= 2 else np.nan
    sem_diff = sd_diff / np.sqrt(n) if n >= 2 else np.nan

    if n < 2 or not np.isfinite(sd_diff):
        cohen_dz = np.nan
    elif sd_diff == 0:
        if mean_diff == 0:
            cohen_dz = np.nan
        else:
            cohen_dz = np.inf if mean_diff > 0 else -np.inf
    else:
        cohen_dz = mean_diff / sd_diff

    return {
        "n_valid_trials": n,
        "mean_moving_minus_baseline": mean_diff,
        "sd_moving_minus_baseline": sd_diff,
        "sem_moving_minus_baseline": sem_diff,
        "cohen_dz_moving_minus_baseline": cohen_dz,
        "n_pos": int(np.sum(diff > 0)),
        "n_neg": int(np.sum(diff < 0)),
        "n_zero": int(np.sum(diff == 0)),
    }


def paired_sign_flip_test(diff, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Sign-flip permutation test on trial-level differences.

    Returns:
        observed mean, two-sided p, greater p, less p
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return np.nan, np.nan, np.nan, np.nan

    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)

    # Generate in chunks to avoid a large temporary array if trial count grows.
    chunk_size = 1000
    n_done = 0
    n_two = 0
    n_greater = 0
    n_less = 0

    while n_done < n_perm:
        n_chunk = min(chunk_size, n_perm - n_done)
        signs = rng.choice((-1.0, 1.0), size=(n_chunk, len(diff)))
        null = np.mean(signs * diff, axis=1)

        n_two += int(np.sum(np.abs(null) >= abs(obs)))
        n_greater += int(np.sum(null >= obs))
        n_less += int(np.sum(null <= obs))
        n_done += n_chunk

    p_two = (n_two + 1) / (n_perm + 1)
    p_greater = (n_greater + 1) / (n_perm + 1)
    p_less = (n_less + 1) / (n_perm + 1)

    return obs, p_two, p_greater, p_less


def run_ttest(diff) -> dict:
    """One-sample t-test on paired trial differences."""
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return {
            "t_stat_moving_minus_baseline": np.nan,
            "p_ttest_two_sided": np.nan,
            "p_ttest_responsive": np.nan,
            "p_ttest_suppressed": np.nan,
        }

    mean_diff = float(np.mean(x))
    sd_diff = float(np.std(x, ddof=1))

    if sd_diff == 0:
        if mean_diff == 0:
            t_stat = np.nan
            p_two = p_greater = p_less = 1.0
        elif mean_diff > 0:
            t_stat = np.inf
            p_two, p_greater, p_less = 0.0, 0.0, 1.0
        else:
            t_stat = -np.inf
            p_two, p_greater, p_less = 0.0, 1.0, 0.0
    else:
        t_stat, p_two = stats.ttest_1samp(
            x, popmean=0, alternative="two-sided"
        )
        _, p_greater = stats.ttest_1samp(
            x, popmean=0, alternative="greater"
        )
        _, p_less = stats.ttest_1samp(
            x, popmean=0, alternative="less"
        )

    return {
        "t_stat_moving_minus_baseline": float(t_stat),
        "p_ttest_two_sided": float(p_two),
        "p_ttest_responsive": float(p_greater),
        "p_ttest_suppressed": float(p_less),
    }

def one_way_f_stat(values, groups):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    valid = np.isfinite(values) & ~pd.isna(groups)
    values = values[valid]
    groups = groups[valid]

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return np.nan

    grand_mean = np.mean(values)
    ss_between = 0.0
    ss_within = 0.0

    for group in unique_groups:
        v = values[groups == group]
        if len(v) == 0:
            continue
        ss_between += len(v) * (np.mean(v) - grand_mean) ** 2
        ss_within += np.sum((v - np.mean(v)) ** 2)

    df_between = len(unique_groups) - 1
    df_within = len(values) - len(unique_groups)

    if df_within <= 0 or ss_within <= 0:
        return np.nan

    return (ss_between / df_between) / (ss_within / df_within)


def direction_permutation_test(
    values,
    directions,
    n_perm=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    """Permutation ANOVA for direction tuning."""
    values = np.asarray(values, dtype=float)
    directions = np.asarray(directions)

    valid = np.isfinite(values) & ~pd.isna(directions)
    values = values[valid]
    directions = directions[valid]

    if len(np.unique(directions)) < 2:
        return np.nan, np.nan

    observed_f = one_way_f_stat(values, directions)
    if not np.isfinite(observed_f):
        return observed_f, np.nan

    rng = np.random.default_rng(seed)
    n_extreme = 0
    n_valid = 0

    for _ in range(n_perm):
        perm_f = one_way_f_stat(values, rng.permutation(directions))
        if np.isfinite(perm_f):
            n_valid += 1
            n_extreme += int(perm_f >= observed_f)

    if n_valid == 0:
        return observed_f, np.nan

    return observed_f, (n_extreme + 1) / (n_valid + 1)


def metadata_from_group(df: pd.DataFrame) -> dict:
    fields = [
        "speed",
        "speed_label",
        "baseline_pooling_mode",
        "recording_site_side",
        "ipsilateral_screen_role",
        "contralateral_screen_role",
    ]
    return {field: first_nonnull(df[field]) for field in fields if field in df.columns}


def merge_optional(
    left: pd.DataFrame,
    path: Path,
    keys: list[str],
    preferred_columns: list[str] | None = None,
    suffix: str = "_summary",
) -> pd.DataFrame:
    """Merge an optional downstream summary without duplicating existing columns."""
    if not path.exists() or left.empty:
        return left

    right = pd.read_csv(path)
    if any(key not in right.columns for key in keys):
        print(f"Skipping optional merge from {path.name}: missing keys {keys}")
        return left

    if preferred_columns is None:
        keep = list(right.columns)
    else:
        keep = keys + [
            c for c in preferred_columns
            if c in right.columns and c not in keys
        ]

    right = right[keep].drop_duplicates(subset=keys)
    return left.merge(right, on=keys, how="left", suffixes=("", suffix))


# ============================================================================
# 8-direction analysis
# ============================================================================


def analyze_8direction(unit_trial: pd.DataFrame):
    df8 = unit_trial.loc[
        unit_trial["trial_kind"] == TRIAL_KIND_8D
    ].copy()

    if df8.empty:
        print("No 8-direction trials found.")
        return pd.DataFrame(), pd.DataFrame()

    require_columns(
        df8,
        [
            "unit_id",
            "trial_id",
            "active_screen_role",
            "direction",
            "moving_minus_baseline",
        ],
        "8-direction unit_trial_summary",
    )

    df8["active_screen_role"] = (
        df8["active_screen_role"].astype(str).str.strip()
    )
    df8["direction"] = pd.to_numeric(df8["direction"], errors="coerce") % 360

    unit_rows = []
    direction_rows = []

    # One speed only: do not include speed in the grouping key.
    for (unit_id, screen_role), group in df8.groupby(
        ["unit_id", "active_screen_role"],
        dropna=False,
    ):
        diff = group["moving_minus_baseline"].to_numpy(dtype=float)
        desc = descriptive_diff_stats(diff)

        _, p_two, p_greater, p_less = paired_sign_flip_test(
            diff,
            seed=stable_seed("8d-unit", unit_id, screen_role),
        )

        direction_f, p_direction = direction_permutation_test(
            values=group["moving_minus_baseline"],
            directions=group["direction"],
            seed=stable_seed("8d-direction-anova", unit_id, screen_role),
        )

        unit_rows.append(
            {
                "unit_id": unit_id,
                "active_screen_role": screen_role,
                **metadata_from_group(group),
                "n_trials": group["trial_id"].nunique(),
                "n_directions": group["direction"].nunique(),
                **desc,
                "p_motion_baseline_two_sided": p_two,
                "p_motion_baseline_responsive": p_greater,
                "p_motion_baseline_suppressed": p_less,
                "direction_f_stat_motion_baseline": direction_f,
                "p_direction_tuning_motion_baseline": p_direction,
            }
        )

        for direction, group_dir in group.groupby("direction", dropna=False):
            dir_diff = group_dir["moving_minus_baseline"].to_numpy(dtype=float)
            dir_desc = descriptive_diff_stats(dir_diff)

            _, p_dir_two, p_dir_greater, p_dir_less = paired_sign_flip_test(
                dir_diff,
                seed=stable_seed(
                    "8d-direction-response",
                    unit_id,
                    screen_role,
                    direction,
                ),
            )

            ttest = run_ttest(dir_diff)

            direction_rows.append(
                {
                    "unit_id": unit_id,
                    "active_screen_role": screen_role,
                    "direction": direction,
                    **metadata_from_group(group_dir),
                    "n_trials": group_dir["trial_id"].nunique(),
                    **dir_desc,
                    "p_motion_baseline_two_sided": p_dir_two,
                    "p_motion_baseline_responsive": p_dir_greater,
                    "p_motion_baseline_suppressed": p_dir_less,
                    **ttest,
                }
            )

    unit_sig = pd.DataFrame(unit_rows)
    direction_sig = pd.DataFrame(direction_rows)

    # FDR family 1: all unit x screen rows.
    unit_sig["q_motion_baseline"] = bh_fdr(
        unit_sig["p_motion_baseline_two_sided"]
    )
    unit_sig["q_direction_tuning_motion_baseline"] = bh_fdr(
        unit_sig["p_direction_tuning_motion_baseline"]
    )

    unit_sig["is_motion_baseline_responsive"] = (
        (unit_sig["q_motion_baseline"] < ALPHA)
        & (unit_sig["mean_moving_minus_baseline"] > 0)
    )
    unit_sig["is_motion_baseline_suppressed"] = (
        (unit_sig["q_motion_baseline"] < ALPHA)
        & (unit_sig["mean_moving_minus_baseline"] < 0)
    )
    unit_sig["is_direction_tuned_motion_baseline"] = (
        unit_sig["q_direction_tuning_motion_baseline"] < ALPHA
    )

    # FDR family 2: all unit x screen x direction rows.
    # Keep separate q-values for the permutation and paired t-test results.
    direction_sig["q_motion_baseline_direction"] = bh_fdr(
        direction_sig["p_motion_baseline_two_sided"]
    )
    direction_sig["q_ttest_direction"] = bh_fdr(
        direction_sig["p_ttest_two_sided"]
    )

    direction_sig["is_direction_response_significant"] = (
        direction_sig["q_motion_baseline_direction"] < ALPHA
    )
    direction_sig["is_direction_excited"] = (
        direction_sig["is_direction_response_significant"]
        & (direction_sig["mean_moving_minus_baseline"] > 0)
    )
    direction_sig["is_direction_suppressed"] = (
        direction_sig["is_direction_response_significant"]
        & (direction_sig["mean_moving_minus_baseline"] < 0)
    )

    direction_sig["is_direction_ttest_significant"] = (
        direction_sig["q_ttest_direction"] < ALPHA
    )
    direction_sig["is_direction_ttest_excited"] = (
        direction_sig["is_direction_ttest_significant"]
        & (direction_sig["mean_moving_minus_baseline"] > 0)
    )
    direction_sig["is_direction_ttest_suppressed"] = (
        direction_sig["is_direction_ttest_significant"]
        & (direction_sig["mean_moving_minus_baseline"] < 0)
    )

    return unit_sig, direction_sig


# ============================================================================
# 12-pattern analysis
# ============================================================================


def analyze_12patterns(unit_trial: pd.DataFrame) -> pd.DataFrame:
    df12 = unit_trial.loc[
        unit_trial["trial_kind"] == TRIAL_KIND_12P
    ].copy()

    if df12.empty:
        print("No 12-pattern trials found.")
        return pd.DataFrame()

    require_columns(
        df12,
        [
            "unit_id",
            "trial_id",
            "pattern",
            "moving_minus_baseline",
        ],
        "12-pattern unit_trial_summary",
    )

    rows = []

    # One speed only: group by unit and pattern only.
    for (unit_id, pattern), group in df12.groupby(
        ["unit_id", "pattern"],
        dropna=False,
    ):
        diff = group["moving_minus_baseline"].to_numpy(dtype=float)
        desc = descriptive_diff_stats(diff)

        _, p_two, p_greater, p_less = paired_sign_flip_test(
            diff,
            seed=stable_seed("12p-pattern", unit_id, pattern),
        )
        ttest = run_ttest(diff)

        row = {
            "unit_id": unit_id,
            "pattern": pattern,
            **metadata_from_group(group),
            "pattern_baseline_pool": first_nonnull(
                group["pattern_baseline_pool"]
            ) if "pattern_baseline_pool" in group.columns else np.nan,
            "biological_label": first_nonnull(
                group["biological_label"]
            ) if "biological_label" in group.columns else np.nan,
            "left_movement": first_nonnull(
                group["left_movement"]
            ) if "left_movement" in group.columns else np.nan,
            "front_movement": first_nonnull(
                group["front_movement"]
            ) if "front_movement" in group.columns else np.nan,
            "right_movement": first_nonnull(
                group["right_movement"]
            ) if "right_movement" in group.columns else np.nan,
            "n_trials": group["trial_id"].nunique(),
            **desc,
            "p_motion_specific_two_sided": p_two,
            "p_motion_responsive": p_greater,
            "p_motion_suppressed": p_less,
            **ttest,
        }
        rows.append(row)

    sig = pd.DataFrame(rows)

    # Global FDR across every unit x pattern test. Two-sided tests only.
    sig["q_motion_specific_global"] = bh_fdr(
        sig["p_motion_specific_two_sided"]
    )
    sig["q_ttest_two_sided_global"] = bh_fdr(
        sig["p_ttest_two_sided"]
    )

    # New: FDR independently within each pattern across units.
    sig = add_groupwise_fdr(
        sig,
        p_col="p_motion_specific_two_sided",
        q_col="q_motion_specific_within_pattern",
        group_cols=["pattern"],
    )
    sig = add_groupwise_fdr(
        sig,
        p_col="p_ttest_two_sided",
        q_col="q_ttest_two_sided_within_pattern",
        group_cols=["pattern"],
    )

    # Only two-sided gates are used for labels.
    sig["is_motion_specific_global"] = (
        sig["q_motion_specific_global"] < ALPHA
    )
    sig["is_responsive_global_two_sided_gate"] = (
        sig["is_motion_specific_global"]
        & (sig["mean_moving_minus_baseline"] > 0)
    )
    sig["is_suppressed_global_two_sided_gate"] = (
        sig["is_motion_specific_global"]
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    sig["is_ttest_specific_global"] = (
        sig["q_ttest_two_sided_global"] < ALPHA
    )
    sig["is_ttest_responsive_global_two_sided_gate"] = (
        sig["is_ttest_specific_global"]
        & (sig["mean_moving_minus_baseline"] > 0)
    )
    sig["is_ttest_suppressed_global_two_sided_gate"] = (
        sig["is_ttest_specific_global"]
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    sig["is_motion_specific_within_pattern"] = (
        sig["q_motion_specific_within_pattern"] < ALPHA
    )
    sig["is_responsive_within_pattern_two_sided_gate"] = (
        sig["is_motion_specific_within_pattern"]
        & (sig["mean_moving_minus_baseline"] > 0)
    )
    sig["is_suppressed_within_pattern_two_sided_gate"] = (
        sig["is_motion_specific_within_pattern"]
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    sig["is_ttest_specific_within_pattern"] = (
        sig["q_ttest_two_sided_within_pattern"] < ALPHA
    )
    sig["is_ttest_responsive_within_pattern_two_sided_gate"] = (
        sig["is_ttest_specific_within_pattern"]
        & (sig["mean_moving_minus_baseline"] > 0)
    )
    sig["is_ttest_suppressed_within_pattern_two_sided_gate"] = (
        sig["is_ttest_specific_within_pattern"]
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    return sig


# ============================================================================
# Main / merges / output
# ============================================================================


def main() -> None:
    print("===== NP unified significance analysis =====")
    print("Single-speed analysis: speed is metadata, not a grouping key.")

    trial_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    unit_trial = pd.read_csv(trial_path)

    require_columns(
        unit_trial,
        [
            "unit_id",
            "trial_id",
            "trial_kind",
            "moving_minus_baseline",
            "baseline_pooling_mode",
        ],
        "unit_trial_summary.csv",
    )

    print(f"Rows: {len(unit_trial)}")
    print(f"Units: {unit_trial['unit_id'].nunique()}")
    print("Trial kinds:")
    print(unit_trial["trial_kind"].value_counts(dropna=False))

    speed_values = unit_trial["speed"].dropna().unique().tolist() \
        if "speed" in unit_trial.columns else []
    print(f"Observed speed values: {speed_values}")
    if len(speed_values) > 1:
        raise ValueError(
            "This NP SB06 version is configured for one speed, but multiple "
            f"speed values were found: {speed_values}"
        )

    unit_sig, direction_sig = analyze_8direction(unit_trial)
    pattern_sig = analyze_12patterns(unit_trial)

    # Merge SB05 summaries where present.
    unit_sig = merge_optional(
        unit_sig,
        ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv",
        keys=["unit_id", "active_screen_role"],
        suffix="_tuning",
    )

    direction_sig = merge_optional(
        direction_sig,
        ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv",
        keys=["unit_id", "active_screen_role", "direction"],
        preferred_columns=[
            "n_trials",
            "baseline_fr",
            "baseline_fr_sem",
            "static_fr",
            "moving_fr",
            "moving_minus_baseline",
            "moving_minus_static",
            "pooled_baseline_fr",
            "baseline_pooling_mode",
        ],
        suffix="_condition",
    )

    pattern_sig = merge_optional(
        pattern_sig,
        ANALYSIS_OUTPUT_DIR / "unit_pattern_summary.csv",
        keys=["unit_id", "pattern"],
        preferred_columns=[
            "n_trials",
            "baseline_fr_mean",
            "baseline_fr_sem",
            "raw_static_fr_mean",
            "raw_static_fr_sem",
            "moving_fr_mean",
            "moving_fr_sem",
            "moving_minus_baseline_mean",
            "moving_minus_baseline_sem",
            "early_minus_baseline_mean",
            "early_minus_baseline_sem",
            "sustained_minus_baseline_mean",
            "sustained_minus_baseline_sem",
        ],
        suffix="_pattern_summary",
    )

    # Merge curated unit metadata into the two tables that do not already get it
    # through unit_tuning_summary.
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"
    if units_path.exists():
        units = pd.read_csv(units_path)
        require_columns(units, ["unit_id"], "curated_units.csv")
        units = units.drop_duplicates(subset=["unit_id"])

        if not unit_sig.empty:
            missing_unit_cols = [
                c for c in units.columns
                if c != "unit_id" and c not in unit_sig.columns
            ]
            unit_sig = unit_sig.merge(
                units[["unit_id"] + missing_unit_cols],
                on="unit_id",
                how="left",
            )

        if not direction_sig.empty:
            direction_sig = direction_sig.merge(
                units,
                on="unit_id",
                how="left",
                suffixes=("", "_unit"),
            )

        if not pattern_sig.empty:
            pattern_sig = pattern_sig.merge(
                units,
                on="unit_id",
                how="left",
                suffixes=("", "_unit"),
            )

    # Stable sort order.
    if not unit_sig.empty:
        unit_sig = unit_sig.sort_values(
            ["unit_id", "active_screen_role"]
        ).reset_index(drop=True)

    if not direction_sig.empty:
        direction_sig = direction_sig.sort_values(
            ["unit_id", "active_screen_role", "direction"]
        ).reset_index(drop=True)

    if not pattern_sig.empty:
        pattern_sig = pattern_sig.sort_values(
            ["unit_id", "pattern"]
        ).reset_index(drop=True)

    unit_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    direction_path = ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv"
    pattern_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_significance.csv"

    unit_sig.to_csv(unit_path, index=False)
    direction_sig.to_csv(direction_path, index=False)
    pattern_sig.to_csv(pattern_path, index=False)

    print("\n===== Saved =====")
    print(unit_path)
    print(direction_path)
    print(pattern_path)

    print("\n8-direction unit x screen rows:", len(unit_sig))
    if not unit_sig.empty:
        print("Motion responsive:", int(unit_sig["is_motion_baseline_responsive"].sum()))
        print("Motion suppressed:", int(unit_sig["is_motion_baseline_suppressed"].sum()))
        print("Direction tuned:", int(unit_sig["is_direction_tuned_motion_baseline"].sum()))

    print("\n8-direction unit x screen x direction rows:", len(direction_sig))
    if not direction_sig.empty:
        print(
            "Permutation significant directions:",
            int(direction_sig["is_direction_response_significant"].sum()),
        )
        print(
            "Paired t-test significant directions:",
            int(direction_sig["is_direction_ttest_significant"].sum()),
        )

    print("\n12-pattern unit x pattern rows:", len(pattern_sig))
    if not pattern_sig.empty:
        print(
            "Global permutation two-sided significant:",
            int(pattern_sig["is_motion_specific_global"].sum()),
        )
        print(
            "Within-pattern permutation two-sided significant:",
            int(pattern_sig["is_motion_specific_within_pattern"].sum()),
        )
        print(
            "Global t-test two-sided significant:",
            int(pattern_sig["is_ttest_specific_global"].sum()),
        )
        print(
            "Within-pattern t-test two-sided significant:",
            int(pattern_sig["is_ttest_specific_within_pattern"].sum()),
        )


if __name__ == "__main__":
    main()
