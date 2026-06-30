# SB06_compute_significance_speed_split.py
#
# Speed-specific significance testing for single-screen 8-direction / 2-speed stimulus.
#
# Inputs from updated SB05:
#   unit_trial_summary.csv
#   unit_tuning_summary.csv      # unit × speed
#
# Outputs:
#   unit_significance_summary.csv        unit × speed
#   unit_direction_significance.csv      unit × speed × direction
#
# All tests use moving_minus_baseline and are performed separately for each speed.

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


N_PERMUTATIONS = 5000
RANDOM_SEED = 42
ALPHA = 0.05


# =====================
# Statistical helpers
# =====================


def paired_permutation_test(diff, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Paired sign-flip permutation test.

    Used for trial-level signed response:
        moving_fr - baseline_fr

    Returns:
        obs_mean, p_two_sided, p_greater, p_less
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    obs = np.mean(diff)

    signs = rng.choice([-1, 1], size=(n_perm, len(diff)), replace=True)
    null = np.mean(signs * diff, axis=1)

    p_two = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    p_greater = (np.sum(null >= obs) + 1) / (n_perm + 1)
    p_less = (np.sum(null <= obs) + 1) / (n_perm + 1)

    return obs, p_two, p_greater, p_less


def one_way_f_stat(values, groups):
    """Simple one-way ANOVA F statistic."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    ok = ~np.isnan(values) & ~pd.isna(groups)
    values = values[ok]
    groups = groups[ok]

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        return np.nan

    grand_mean = np.mean(values)
    ss_between = 0.0
    ss_within = 0.0

    for g in unique_groups:
        v = values[groups == g]
        if len(v) == 0:
            continue
        ss_between += len(v) * (np.mean(v) - grand_mean) ** 2
        ss_within += np.sum((v - np.mean(v)) ** 2)

    df_between = len(unique_groups) - 1
    df_within = len(values) - len(unique_groups)

    if df_within <= 0 or ss_within <= 0:
        return np.nan

    return (ss_between / df_between) / (ss_within / df_within)


def direction_permutation_test(values, directions, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Permutation test for direction tuning within one speed.

    Null:
        moving-baseline responses are exchangeable across direction labels.
    """
    values = np.asarray(values, dtype=float)
    directions = np.asarray(directions)

    ok = ~np.isnan(values) & ~pd.isna(directions)
    values = values[ok]
    directions = directions[ok]

    if len(np.unique(directions)) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    obs_f = one_way_f_stat(values, directions)

    if np.isnan(obs_f):
        return np.nan, np.nan

    null_f = []
    for _ in range(n_perm):
        shuffled_dirs = rng.permutation(directions)
        null_f.append(one_way_f_stat(values, shuffled_dirs))

    null_f = np.asarray(null_f)
    null_f = null_f[~np.isnan(null_f)]

    if len(null_f) == 0:
        return obs_f, np.nan

    p_value = (np.sum(null_f >= obs_f) + 1) / (len(null_f) + 1)
    return obs_f, p_value


def speed_permutation_test(values, speeds, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Test whether responses differ across speed labels.

    Use case:
        unit × direction level

    Example:
        For one unit and one direction, test whether moving_minus_baseline
        differs among speed_1, speed_4, speed_16, speed_64, speed_128, speed_256.

    Null:
        response values are exchangeable across speed labels.
    """
    values = np.asarray(values, dtype=float)
    speeds = np.asarray(speeds)

    ok = ~np.isnan(values) & ~pd.isna(speeds)
    values = values[ok]
    speeds = speeds[ok]

    if len(np.unique(speeds)) < 2:
        return np.nan, np.nan

    obs_f = one_way_f_stat(values, speeds)

    if np.isnan(obs_f):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    null_f = []

    for _ in range(n_perm):
        shuffled_speeds = rng.permutation(speeds)
        null_f.append(one_way_f_stat(values, shuffled_speeds))

    null_f = np.asarray(null_f)
    null_f = null_f[~np.isnan(null_f)]

    if len(null_f) == 0:
        return obs_f, np.nan

    p_value = (np.sum(null_f >= obs_f) + 1) / (len(null_f) + 1)

    return obs_f, p_value


def speed_permutation_test_blocked_by_direction(
    values,
    speeds,
    directions,
    n_perm=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    """
    Test whether responses differ across speeds while preserving direction structure.

    Use case:
        unit level

    Why blocked by direction:
        A neuron may have strong direction tuning. If we simply shuffle all speed labels
        across all trials, direction tuning can contaminate the speed test.
        Therefore, speed labels are shuffled only within each direction.

    Null:
        within each direction, response values are exchangeable across speed labels.
    """
    df = pd.DataFrame(
        {
            "value": values,
            "speed": speeds,
            "direction": directions,
        }
    ).dropna(subset=["value", "speed", "direction"])

    if df["speed"].nunique() < 2:
        return np.nan, np.nan

    obs_f = one_way_f_stat(df["value"].values, df["speed"].values)

    if np.isnan(obs_f):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    null_f = []

    for _ in range(n_perm):
        shuffled = df.copy()

        for direction, idx in shuffled.groupby("direction").groups.items():
            shuffled.loc[idx, "speed"] = rng.permutation(
                shuffled.loc[idx, "speed"].values
            )

        null_f.append(
            one_way_f_stat(
                shuffled["value"].values,
                shuffled["speed"].values,
            )
        )

    null_f = np.asarray(null_f)
    null_f = null_f[~np.isnan(null_f)]

    if len(null_f) == 0:
        return obs_f, np.nan

    p_value = (np.sum(null_f >= obs_f) + 1) / (len(null_f) + 1)

    return obs_f, p_value


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)

    valid = ~np.isnan(p_values)
    p = p_values[valid]

    if len(p) == 0:
        return q_values

    order = np.argsort(p)
    ranked_p = p[order]

    m = len(p)
    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty_like(ranked_q)
    q[order] = ranked_q

    q_values[valid] = q
    return q_values


# =====================
# Main
# =====================


def main():
    print("===== Compute speed-specific motion-baseline and direction significance =====")

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"

    unit_trial = pd.read_csv(trial_summary_path)
    unit_tuning = pd.read_csv(tuning_summary_path)

    if "speed" not in unit_trial.columns:
        raise ValueError("unit_trial_summary.csv must contain a 'speed' column for speed-specific analysis.")

    rows = []
    direction_rows = []
    speed_rows = []
    direction_speed_rows = []

    for (unit_id, speed), df in unit_trial.groupby(["unit_id", "speed"], dropna=False):
        motion_response = df["moving_minus_baseline"]

        mean_motion_response, p_motion_two, p_motion_greater, p_motion_less = paired_permutation_test(
            motion_response
        )

        direction_f, p_direction = direction_permutation_test(
            values=motion_response,
            directions=df["direction"],
        )

        rows.append(
            {
                "unit_id": unit_id,
                "speed": speed,
                "n_trials": df["trial_id"].nunique(),
                "mean_moving_minus_baseline": mean_motion_response,
                "p_motion_baseline_two_sided": p_motion_two,
                "p_motion_baseline_responsive": p_motion_greater,
                "p_motion_baseline_suppressed": p_motion_less,
                "direction_f_stat_motion_baseline": direction_f,
                "p_direction_tuning_motion_baseline": p_direction,
            }
        )

        for direction, df_dir in df.groupby("direction", dropna=False):
            dir_response = df_dir["moving_minus_baseline"]

            mean_dir_response, p_dir_two, p_dir_greater, p_dir_less = paired_permutation_test(
                dir_response
            )

            direction_rows.append(
                {
                    "unit_id": unit_id,
                    "speed": speed,
                    "direction": direction,
                    "n_trials": df_dir["trial_id"].nunique(),
                    "mean_moving_minus_baseline": mean_dir_response,
                    "p_motion_baseline_two_sided": p_dir_two,
                    "p_motion_baseline_responsive": p_dir_greater,
                    "p_motion_baseline_suppressed": p_dir_less,
                }
            )

        # -----------------------------
    
    # Speed-effect tests
    # -----------------------------
    # These tests ask whether moving_minus_baseline differs across speeds.
    #
    # 1. unit-level speed effect:
    #       unit_id only
    #       speed labels are shuffled within each direction
    #
    # 2. unit × direction speed effect:
    #       unit_id × direction
    #       speed labels are shuffled within that direction

    for unit_id, df_unit in unit_trial.groupby("unit_id", dropna=False):

        speed_f, p_speed = speed_permutation_test_blocked_by_direction(
            values=df_unit["moving_minus_baseline"],
            speeds=df_unit["speed"],
            directions=df_unit["direction"],
        )

        speed_rows.append(
            {
                "unit_id": unit_id,
                "n_trials": df_unit["trial_id"].nunique(),
                "n_speeds": df_unit["speed"].nunique(),
                "n_directions": df_unit["direction"].nunique(),
                "speed_f_stat_motion_baseline": speed_f,
                "p_speed_effect_motion_baseline": p_speed,
            }
        )

        for direction, df_dir in df_unit.groupby("direction", dropna=False):

            speed_f_dir, p_speed_dir = speed_permutation_test(
                values=df_dir["moving_minus_baseline"],
                speeds=df_dir["speed"],
            )

            direction_speed_rows.append(
                {
                    "unit_id": unit_id,
                    "direction": direction,
                    "n_trials": df_dir["trial_id"].nunique(),
                    "n_speeds": df_dir["speed"].nunique(),
                    "speed_f_stat_motion_baseline": speed_f_dir,
                    "p_speed_effect_motion_baseline": p_speed_dir,
                }
            )

    sig = pd.DataFrame(rows)
    dir_sig = pd.DataFrame(direction_rows)

    speed_sig = pd.DataFrame(speed_rows)
    dir_speed_sig = pd.DataFrame(direction_speed_rows)

    # -----------------------------
    # FDR correction
    # -----------------------------
    # Unit × speed level. Correction is across all unit-speed rows.
    sig["q_motion_baseline"] = bh_fdr(sig["p_motion_baseline_two_sided"])
    sig["q_direction_tuning_motion_baseline"] = bh_fdr(sig["p_direction_tuning_motion_baseline"])

    sig["is_motion_baseline_responsive"] = (
        (sig["q_motion_baseline"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] > 0)
    )
    sig["is_motion_baseline_suppressed"] = (
        (sig["q_motion_baseline"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] < 0)
    )
    sig["is_direction_tuned_motion_baseline"] = (
        sig["q_direction_tuning_motion_baseline"] < ALPHA
    )

    # Unit × speed × direction level. Correction is across all rows.
    dir_sig["q_motion_baseline_direction"] = bh_fdr(
        dir_sig["p_motion_baseline_two_sided"]
    )
    dir_sig["is_direction_response_significant"] = (
        dir_sig["q_motion_baseline_direction"] < ALPHA
    )
    dir_sig["is_direction_excited"] = (
        dir_sig["is_direction_response_significant"]
        & (dir_sig["mean_moving_minus_baseline"] > 0)
    )
    dir_sig["is_direction_suppressed"] = (
        dir_sig["is_direction_response_significant"]
        & (dir_sig["mean_moving_minus_baseline"] < 0)
    )

    # -----------------------------
    # FDR correction for speed-effect tests
    # -----------------------------
    # Unit-level speed effect:
    # correction is across units.
    speed_sig["q_speed_effect_motion_baseline"] = bh_fdr(
        speed_sig["p_speed_effect_motion_baseline"]
    )

    speed_sig["is_speed_modulated_motion_baseline"] = (
        speed_sig["q_speed_effect_motion_baseline"] < ALPHA
    )

    # Unit × direction speed effect:
    # correction is across all unit-direction rows.
    dir_speed_sig["q_speed_effect_motion_baseline"] = bh_fdr(
        dir_speed_sig["p_speed_effect_motion_baseline"]
    )

    dir_speed_sig["is_speed_modulated_direction"] = (
        dir_speed_sig["q_speed_effect_motion_baseline"] < ALPHA
    )

    # Merge speed-specific tuning metrics from SB05.
    merge_cols = ["unit_id", "speed"]
    if not all(c in unit_tuning.columns for c in merge_cols):
        raise ValueError("unit_tuning_summary.csv must contain unit_id and speed.")

    sig = sig.merge(unit_tuning, on=merge_cols, how="left")

    # -----------------------------
    # Save
    # -----------------------------

    unit_sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    dir_sig_path = ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv"

    speed_sig_path = ANALYSIS_OUTPUT_DIR / "unit_speed_effect_summary.csv"
    dir_speed_sig_path = ANALYSIS_OUTPUT_DIR / "unit_direction_speed_effect.csv"

    sig.to_csv(unit_sig_path, index=False)
    dir_sig.to_csv(dir_sig_path, index=False)

    speed_sig.to_csv(speed_sig_path, index=False)
    dir_speed_sig.to_csv(dir_speed_sig_path, index=False)

    print("\n===== Saved =====")
    print(unit_sig_path)
    print(dir_sig_path)

    print("\nUnit × speed summary counts:")
    print("Motion-baseline responsive:", int(sig["is_motion_baseline_responsive"].sum()))
    print("Motion-baseline suppressed:", int(sig["is_motion_baseline_suppressed"].sum()))
    print("Direction tuned:", int(sig["is_direction_tuned_motion_baseline"].sum()))

    print("\nDirection-level counts:")
    print("Significant direction responses:", int(dir_sig["is_direction_response_significant"].sum()))
    print("Excited direction responses:", int(dir_sig["is_direction_excited"].sum()))
    print("Suppressed direction responses:", int(dir_sig["is_direction_suppressed"].sum()))

    print("\nFirst few unit × speed rows:")
    print(sig.head())

    print("\nFirst few unit × speed × direction rows:")
    print(dir_sig.head())


if __name__ == "__main__":
    main()
