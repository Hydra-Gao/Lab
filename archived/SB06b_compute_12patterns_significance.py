# SB05d_compute_vbc_pattern_significance.py

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


N_PERMUTATIONS = 5000
RANDOM_SEED = 42
ALPHA = 0.05


def paired_permutation_test(diff, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Paired sign-flip permutation test.

    Input:
        diff = moving_fr - baseline_fr for repeated trials
               within one unit × pattern × speed condition.

    Tests:
        two-sided: whether mean(diff) != 0
        greater:   whether mean(diff) > 0, responsive
        less:      whether mean(diff) < 0, suppressed
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
    print("===== Compute VbC pattern-specific motion significance =====")

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    pattern_summary_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_summary.csv"

    unit_trial = pd.read_csv(trial_summary_path)
    unit_pattern = pd.read_csv(pattern_summary_path)

    required_cols = [
        "unit_id",
        "trial_id",
        "pattern",
        "speed_deg_per_sec",
        "moving_minus_baseline",
    ]

    missing = [c for c in required_cols if c not in unit_trial.columns]
    if missing:
        raise ValueError(
            f"unit_trial_summary.csv is missing required columns: {missing}\n"
            f"Available columns: {unit_trial.columns.tolist()}"
        )

    group_cols = [
        "unit_id",
        "pattern",
        "speed_deg_per_sec",
    ]

    optional_metadata_cols = [
        "biological_label",
        "left_movement",
        "front_movement",
        "right_movement",
        "tf_hz",
        "sf_cpd",
    ]

    for col in optional_metadata_cols:
        if col in unit_trial.columns:
            group_cols.append(col)

    rows = []

    for keys, df in unit_trial.groupby(group_cols, dropna=False):

        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))

        diff = df["moving_minus_baseline"].to_numpy(dtype=float)

        mean_diff, p_two, p_greater, p_less = paired_permutation_test(diff)

        rows.append(
            {
                **key_dict,

                "n_trials": df["trial_id"].nunique(),

                "mean_moving_minus_baseline": mean_diff,

                "p_motion_specific_two_sided": p_two,
                "p_motion_responsive": p_greater,
                "p_motion_suppressed": p_less,
            }
        )

    sig = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # FDR correction
    # ------------------------------------------------------------
    # This corrects across all unit × pattern × speed tests.
    # For one slow run:
    #   n_tests = n_units × 12 patterns
    #
    # q_motion_specific:
    #   correction for two-sided motion-specific response.
    #
    # q_motion_responsive / q_motion_suppressed:
    #   correction for one-sided responsive / suppressed labels.
    # ------------------------------------------------------------

    sig["q_motion_specific"] = bh_fdr(sig["p_motion_specific_two_sided"])
    sig["q_motion_responsive"] = bh_fdr(sig["p_motion_responsive"])
    sig["q_motion_suppressed"] = bh_fdr(sig["p_motion_suppressed"])

    sig["is_motion_specific"] = sig["q_motion_specific"] < ALPHA

    sig["is_motion_responsive"] = (
        (sig["q_motion_responsive"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] > 0)
    )

    sig["is_motion_suppressed"] = (
        (sig["q_motion_suppressed"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    # A stricter alternative:
    # require two-sided significance first, then direction by sign.
    sig["is_responsive_two_sided_gate"] = (
        (sig["q_motion_specific"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] > 0)
    )

    sig["is_suppressed_two_sided_gate"] = (
        (sig["q_motion_specific"] < ALPHA)
        & (sig["mean_moving_minus_baseline"] < 0)
    )

    # ------------------------------------------------------------
    # Merge mean / SEM from 05c if available
    # ------------------------------------------------------------

    merge_cols = [
        "unit_id",
        "pattern",
        "speed_deg_per_sec",
    ]

    summary_cols = [
        "unit_id",
        "pattern",
        "speed_deg_per_sec",
        "baseline_fr_mean",
        "baseline_fr_sem",
        "moving_fr_mean",
        "moving_fr_sem",
        "moving_minus_baseline_mean",
        "moving_minus_baseline_sem",
        "early_minus_baseline_mean",
        "early_minus_baseline_sem",
        "sustained_minus_baseline_mean",
        "sustained_minus_baseline_sem",
    ]

    summary_cols = [c for c in summary_cols if c in unit_pattern.columns]

    sig = sig.merge(
        unit_pattern[summary_cols],
        on=merge_cols,
        how="left",
    )

    # Sort for easier reading
    sig = sig.sort_values(
        ["unit_id", "speed_deg_per_sec", "pattern"]
    ).reset_index(drop=True)

    out_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_significance.csv"
    sig.to_csv(out_path, index=False)

    print("\n===== Saved =====")
    print(out_path)

    print("\nSummary counts:")
    print("Total unit × pattern tests:", len(sig))
    print("Motion-specific:", int(sig["is_motion_specific"].sum()))
    print("Responsive:", int(sig["is_motion_responsive"].sum()))
    print("Suppressed:", int(sig["is_motion_suppressed"].sum()))

    print("\nFirst few rows:")
    print(sig.head())


if __name__ == "__main__":
    main()