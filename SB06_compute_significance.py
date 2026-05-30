# 05b_compute_significance.py

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


N_PERMUTATIONS = 5000
RANDOM_SEED = 42
ALPHA = 0.05


def paired_permutation_test(diff, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Paired sign-flip permutation test.

    Used for:
        moving_fr - static_fr
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    obs = np.mean(diff)

    signs = rng.choice(
        [-1, 1],
        size=(n_perm, len(diff)),
        replace=True,
    )

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
    Permutation test for direction tuning.

    Null:
        motion-specific responses are exchangeable across direction labels.

    Here values should be:
        moving_fr - static_fr
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


def main():
    print("===== Compute motion-specific and direction tuning significance =====")

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"

    unit_trial = pd.read_csv(trial_summary_path)
    unit_tuning = pd.read_csv(tuning_summary_path)

    rows = []

    for unit_id, df in unit_trial.groupby("unit_id"):

        motion_response = df["moving_fr"] - df["static_fr"]

        mean_motion_response, p_motion_two, p_motion_greater, p_motion_less = (
            paired_permutation_test(motion_response)
        )

        direction_f, p_direction = direction_permutation_test(
            values=motion_response,
            directions=df["direction"],
        )

        rows.append(
            {
                "unit_id": unit_id,
                "n_trials": df["trial_id"].nunique(),

                "mean_moving_minus_static": mean_motion_response,
                "p_motion_specific_two_sided": p_motion_two,
                "p_motion_responsive": p_motion_greater,
                "p_motion_suppressed": p_motion_less,

                "direction_f_stat": direction_f,
                "p_direction_tuning": p_direction,
            }
        )

    sig = pd.DataFrame(rows)

    # FDR correction across units
    sig["q_motion_specific"] = bh_fdr(sig["p_motion_specific_two_sided"])
    sig["q_direction_tuning"] = bh_fdr(sig["p_direction_tuning"])

    # Labels for plotting / summary
    sig["is_motion_responsive"] = (
        (sig["q_motion_specific"] < ALPHA)
        & (sig["mean_moving_minus_static"] > 0)
    )

    sig["is_motion_suppressed"] = (
        (sig["q_motion_specific"] < ALPHA)
        & (sig["mean_moving_minus_static"] < 0)
    )

    sig["is_direction_tuned"] = sig["q_direction_tuning"] < ALPHA

    # Merge tuning metrics from 05 if available
    sig = sig.merge(
        unit_tuning,
        on="unit_id",
        how="left",
    )

    out_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    sig.to_csv(out_path, index=False)

    print("\n===== Saved =====")
    print(out_path)

    print("\nSummary counts:")
    print("Motion responsive:", int(sig["is_motion_responsive"].sum()))
    print("Motion suppressed:", int(sig["is_motion_suppressed"].sum()))
    print("Direction tuned:", int(sig["is_direction_tuned"].sum()))

    print("\nFirst few rows:")
    print(sig.head())


if __name__ == "__main__":
    main()