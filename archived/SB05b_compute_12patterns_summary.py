# SB05c_compute_vbc_pattern_summary.py

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    BASELINE_WINDOW,
    EARLY_WINDOW,
    SUSTAINED_RESPONSE_WINDOW,
    MOVING_WINDOW,
)


def count_spikes_in_window(labeled, unit_id, trial_id, t0, t1):
    """Count spikes for one unit/trial in a window relative to moving onset."""
    sp = labeled[
        (labeled["unit_id"] == unit_id)
        & (labeled["trial_id"] == trial_id)
        & (labeled["time_from_moving_onset"] >= t0)
        & (labeled["time_from_moving_onset"] < t1)
    ]
    return len(sp)


def firing_rate(count, window):
    """Convert spike count to firing rate."""
    duration = window[1] - window[0]
    if duration <= 0:
        return np.nan
    return count / duration


def sem(x):
    """Standard error of the mean."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def main():
    print("===== Compute VbC three-screen pattern summary =====")

    labeled_path = ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv"
    trial_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    labeled = pd.read_csv(labeled_path)
    trials = pd.read_csv(trial_path)
    units = pd.read_csv(units_path)

    unit_ids = units["unit_id"].tolist()

    print(f"Units: {len(unit_ids)}")
    print(f"Trials: {len(trials)}")
    print(f"Labeled spikes: {len(labeled)}")

    # -----------------------------
    # Unit × trial response table
    # -----------------------------

    rows = []

    for unit_id in unit_ids:
        for _, tr in trials.iterrows():

            trial_id = tr["trial_id"]

            baseline_count = count_spikes_in_window(
                labeled,
                unit_id,
                trial_id,
                BASELINE_WINDOW[0],
                BASELINE_WINDOW[1],
            )

            early_count = count_spikes_in_window(
                labeled,
                unit_id,
                trial_id,
                EARLY_WINDOW[0],
                EARLY_WINDOW[1],
            )

            sustained_count = count_spikes_in_window(
                labeled,
                unit_id,
                trial_id,
                SUSTAINED_RESPONSE_WINDOW[0],
                SUSTAINED_RESPONSE_WINDOW[1],
            )

            moving_count = count_spikes_in_window(
                labeled,
                unit_id,
                trial_id,
                MOVING_WINDOW[0],
                MOVING_WINDOW[1],
            )

            baseline_fr = firing_rate(baseline_count, BASELINE_WINDOW)
            early_fr = firing_rate(early_count, EARLY_WINDOW)
            sustained_fr = firing_rate(
                sustained_count,
                SUSTAINED_RESPONSE_WINDOW,
            )
            moving_fr = firing_rate(moving_count, MOVING_WINDOW)

            rows.append(
                {
                    "unit_id": unit_id,
                    "trial_id": trial_id,

                    "trial_number_randomized": tr["trial_number_randomized"],
                    "replicate": tr["replicate"],

                    "pattern": tr["pattern"],
                    "biological_label": tr["biological_label"],

                    "left_movement": tr["left_movement"],
                    "front_movement": tr["front_movement"],
                    "right_movement": tr["right_movement"],

                    "tf_hz": tr["tf_hz"],
                    "sf_cpd": tr["sf_cpd"],
                    "phase_step": tr["phase_step"],
                    "speed_deg_per_sec": tr["speed_deg_per_sec"],

                    "baseline_count": baseline_count,
                    "early_count": early_count,
                    "sustained_count": sustained_count,
                    "moving_count": moving_count,

                    "baseline_fr": baseline_fr,
                    "early_fr": early_fr,
                    "sustained_fr": sustained_fr,
                    "moving_fr": moving_fr,

                    "moving_minus_baseline": moving_fr - baseline_fr,
                    "early_minus_baseline": early_fr - baseline_fr,
                    "sustained_minus_baseline": sustained_fr - baseline_fr,
                }
            )

    unit_trial_summary = pd.DataFrame(rows)

    # -----------------------------
    # Unit × pattern × speed summary
    # -----------------------------

    group_cols = [
        "unit_id",
        "pattern",
        "biological_label",
        "left_movement",
        "front_movement",
        "right_movement",
        "speed_deg_per_sec",
        "tf_hz",
        "sf_cpd",
    ]

    unit_pattern_summary = (
        unit_trial_summary
        .groupby(group_cols, dropna=False)
        .agg(
            n_trials=("trial_id", "nunique"),

            baseline_fr_mean=("baseline_fr", "mean"),
            baseline_fr_sem=("baseline_fr", sem),

            early_fr_mean=("early_fr", "mean"),
            early_fr_sem=("early_fr", sem),

            sustained_fr_mean=("sustained_fr", "mean"),
            sustained_fr_sem=("sustained_fr", sem),

            moving_fr_mean=("moving_fr", "mean"),
            moving_fr_sem=("moving_fr", sem),

            moving_minus_baseline_mean=("moving_minus_baseline", "mean"),
            moving_minus_baseline_sem=("moving_minus_baseline", sem),

            early_minus_baseline_mean=("early_minus_baseline", "mean"),
            early_minus_baseline_sem=("early_minus_baseline", sem),

            sustained_minus_baseline_mean=("sustained_minus_baseline", "mean"),
            sustained_minus_baseline_sem=("sustained_minus_baseline", sem),
        )
        .reset_index()
    )

    # -----------------------------
    # Add simple unit metadata
    # -----------------------------

    unit_pattern_summary = unit_pattern_summary.merge(
        units,
        on="unit_id",
        how="left",
    )

    # -----------------------------
    # Save
    # -----------------------------

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    pattern_summary_path = ANALYSIS_OUTPUT_DIR / "unit_pattern_summary.csv"

    unit_trial_summary.to_csv(trial_summary_path, index=False)
    unit_pattern_summary.to_csv(pattern_summary_path, index=False)

    print("\n===== Saved =====")
    print(trial_summary_path)
    print(pattern_summary_path)

    print("\nFirst few trial summary rows:")
    print(unit_trial_summary.head())

    print("\nFirst few pattern summary rows:")
    print(unit_pattern_summary.head())


if __name__ == "__main__":
    main()