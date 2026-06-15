# Screen- and speed-specific significance testing for
# single-screen 8-direction / 2-speed stimulus.
#
# Inputs from updated SB05c:
#   unit_trial_summary.csv
#       unit × trial, with screen_role and speed columns
#
#   unit_tuning_summary.csv
#       unit × screen_role × speed
#
# Outputs:
#   unit_significance_summary.csv
#       unit × screen_role × speed
#
#   unit_direction_significance.csv
#       unit × screen_role × speed × direction
#
# Important:
#   All tests are performed separately for each:
#       unit_id × screen_role × speed
#
#   This script does NOT compare front vs left vs right.
#   It only computes the same within-screen analyses independently.

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
    Permutation test for direction tuning within one screen_role × speed.

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


def normalize_group_columns(df):
    """
    Normalize screen_role and speed columns to avoid grouping mismatches
    caused by spaces or mixed numeric/string types.
    """
    if "screen_role" not in df.columns:
        df["screen_role"] = "unknown"

    if "speed" not in df.columns:
        raise ValueError(
            "unit_trial_summary.csv must contain a 'speed' column "
            "for speed-specific analysis."
        )

    df["screen_role"] = df["screen_role"].astype(str).str.strip()
    df["speed"] = df["speed"].astype(str).str.strip()

    return df


# =====================
# Main
# =====================


def main():
    print("===== Compute screen- and speed-specific motion-baseline significance =====")

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"

    unit_trial = pd.read_csv(trial_summary_path)
    unit_tuning = pd.read_csv(tuning_summary_path)

    unit_trial = normalize_group_columns(unit_trial)

    if "screen_role" not in unit_tuning.columns:
        print(
            "\nWarning: unit_tuning_summary.csv has no screen_role column. "
            "Using screen_role='unknown'. Run updated SB05c first if you want "
            "front/left/right-specific tuning metrics.\n"
        )
        unit_tuning["screen_role"] = "unknown"

    if "speed" not in unit_tuning.columns:
        raise ValueError("unit_tuning_summary.csv must contain unit_id, screen_role, and speed.")

    unit_tuning["screen_role"] = unit_tuning["screen_role"].astype(str).str.strip()
    unit_tuning["speed"] = unit_tuning["speed"].astype(str).str.strip()

    print("\nRows in unit_trial_summary:")
    print(len(unit_trial))

    print("\nUnit-trial counts by screen_role and speed:")
    print(
        unit_trial
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    rows = []
    direction_rows = []

    group_cols = ["unit_id", "screen_role", "speed"]

    for (unit_id, screen_role, speed), df in unit_trial.groupby(group_cols, dropna=False):

        # -----------------------------
        # Unit × screen_role × speed level
        # -----------------------------

        motion_response = df["moving_minus_baseline"]

        (
            mean_motion_response,
            p_motion_two,
            p_motion_greater,
            p_motion_less,
        ) = paired_permutation_test(motion_response)

        direction_f, p_direction = direction_permutation_test(
            values=motion_response,
            directions=df["direction"],
        )

        rows.append(
            {
                "unit_id": unit_id,
                "screen_role": screen_role,
                "speed": speed,

                # Keep useful screen metadata if available.
                "original_segment_index": df["original_segment_index"].dropna().iloc[0]
                if "original_segment_index" in df.columns and df["original_segment_index"].notna().any()
                else np.nan,

                "active_monitor_config_index": df["active_monitor_config_index"].dropna().iloc[0]
                if "active_monitor_config_index" in df.columns and df["active_monitor_config_index"].notna().any()
                else np.nan,

                "active_monitor_label": df["active_monitor_label"].dropna().iloc[0]
                if "active_monitor_label" in df.columns and df["active_monitor_label"].notna().any()
                else np.nan,

                "active_screen_number": df["active_screen_number"].dropna().iloc[0]
                if "active_screen_number" in df.columns and df["active_screen_number"].notna().any()
                else np.nan,

                "n_trials": df["trial_id"].nunique(),

                "mean_moving_minus_baseline": mean_motion_response,

                "p_motion_baseline_two_sided": p_motion_two,
                "p_motion_baseline_responsive": p_motion_greater,
                "p_motion_baseline_suppressed": p_motion_less,

                "direction_f_stat_motion_baseline": direction_f,
                "p_direction_tuning_motion_baseline": p_direction,
            }
        )

        # -----------------------------
        # Unit × screen_role × speed × direction level
        # -----------------------------

        for direction, df_dir in df.groupby("direction", dropna=False):
            dir_response = df_dir["moving_minus_baseline"]

            (
                mean_dir_response,
                p_dir_two,
                p_dir_greater,
                p_dir_less,
            ) = paired_permutation_test(dir_response)

            direction_rows.append(
                {
                    "unit_id": unit_id,
                    "screen_role": screen_role,
                    "speed": speed,
                    "direction": direction,

                    "original_segment_index": df_dir["original_segment_index"].dropna().iloc[0]
                    if "original_segment_index" in df_dir.columns and df_dir["original_segment_index"].notna().any()
                    else np.nan,

                    "active_monitor_config_index": df_dir["active_monitor_config_index"].dropna().iloc[0]
                    if "active_monitor_config_index" in df_dir.columns and df_dir["active_monitor_config_index"].notna().any()
                    else np.nan,

                    "active_monitor_label": df_dir["active_monitor_label"].dropna().iloc[0]
                    if "active_monitor_label" in df_dir.columns and df_dir["active_monitor_label"].notna().any()
                    else np.nan,

                    "active_screen_number": df_dir["active_screen_number"].dropna().iloc[0]
                    if "active_screen_number" in df_dir.columns and df_dir["active_screen_number"].notna().any()
                    else np.nan,

                    "n_trials": df_dir["trial_id"].nunique(),

                    "mean_moving_minus_baseline": mean_dir_response,

                    "p_motion_baseline_two_sided": p_dir_two,
                    "p_motion_baseline_responsive": p_dir_greater,
                    "p_motion_baseline_suppressed": p_dir_less,
                }
            )

    sig = pd.DataFrame(rows)
    dir_sig = pd.DataFrame(direction_rows)

    # -----------------------------
    # FDR correction
    # -----------------------------
    # Same logic as old SB06c:
    # correction across all rows in the relevant table.
    #
    # Now rows are:
    #   unit × screen_role × speed
    # and:
    #   unit × screen_role × speed × direction
    # -----------------------------

    sig["q_motion_baseline"] = bh_fdr(
        sig["p_motion_baseline_two_sided"]
    )

    sig["q_direction_tuning_motion_baseline"] = bh_fdr(
        sig["p_direction_tuning_motion_baseline"]
    )

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
    # Merge screen-speed-specific tuning metrics from SB05c
    # -----------------------------

    merge_cols = ["unit_id", "screen_role", "speed"]

    missing_merge_cols = [
        c for c in merge_cols
        if c not in unit_tuning.columns
    ]

    if missing_merge_cols:
        raise ValueError(
            f"unit_tuning_summary.csv is missing merge columns: {missing_merge_cols}. "
            "Run updated SB05c first."
        )

    sig = sig.merge(
        unit_tuning,
        on=merge_cols,
        how="left",
        suffixes=("", "_tuning"),
    )

    # -----------------------------
    # Save
    # -----------------------------

    unit_sig_path = ANALYSIS_OUTPUT_DIR / "unit_significance_summary.csv"
    dir_sig_path = ANALYSIS_OUTPUT_DIR / "unit_direction_significance.csv"

    sig.to_csv(unit_sig_path, index=False)
    dir_sig.to_csv(dir_sig_path, index=False)

    print("\n===== Saved =====")
    print(unit_sig_path)
    print(dir_sig_path)

    print("\nUnit × screen_role × speed summary counts:")
    print(
        sig
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    print("\nDirection-level counts by screen_role and speed:")
    print(
        dir_sig
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    print("\nSignificant unit × screen_role × speed counts:")
    print("Motion-baseline responsive:", int(sig["is_motion_baseline_responsive"].sum()))
    print("Motion-baseline suppressed:", int(sig["is_motion_baseline_suppressed"].sum()))
    print("Direction tuned:", int(sig["is_direction_tuned_motion_baseline"].sum()))

    print("\nDirection-level significant counts:")
    print(
        "Significant direction responses:",
        int(dir_sig["is_direction_response_significant"].sum()),
    )
    print("Excited direction responses:", int(dir_sig["is_direction_excited"].sum()))
    print("Suppressed direction responses:", int(dir_sig["is_direction_suppressed"].sum()))

    print("\nFirst few unit × screen_role × speed rows:")
    print(sig.head())

    print("\nFirst few unit × screen_role × speed × direction rows:")
    print(dir_sig.head())


if __name__ == "__main__":
    main()