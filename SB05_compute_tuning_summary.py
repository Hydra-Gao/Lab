# 05_compute_tuning_summary.py

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


def compute_vector_strength(direction_response):
    """
    Compute vector-sum preferred direction and vector strength.

    direction_response:
        DataFrame with columns direction, response
    """
    df = direction_response.dropna(subset=["direction", "response"]).copy()

    if df.empty or df["response"].sum() <= 0:
        return np.nan, np.nan

    angles = np.deg2rad(df["direction"].astype(float).values)
    responses = df["response"].astype(float).values

    vec = np.sum(responses * np.exp(1j * angles))
    total = np.sum(responses)

    preferred_angle = np.rad2deg(np.angle(vec)) % 360
    vector_strength = np.abs(vec) / total

    return preferred_angle, vector_strength


def compute_dsi(direction_response):
    """
    Direction Selectivity Index:
    DSI = (R_pref - R_opp) / (R_pref + R_opp)

    Opposite direction is preferred + 180 degrees.
    """
    df = direction_response.dropna(subset=["direction", "response"]).copy()

    if df.empty:
        return np.nan

    df["direction"] = df["direction"].astype(float) % 360

    pref_row = df.loc[df["response"].idxmax()]
    pref_dir = pref_row["direction"]
    r_pref = pref_row["response"]

    opp_dir = (pref_dir + 180) % 360

    # Find the measured direction closest to the opposite direction
    df["opp_distance"] = np.abs(((df["direction"] - opp_dir + 180) % 360) - 180)
    r_opp = df.loc[df["opp_distance"].idxmin(), "response"]

    if (r_pref + r_opp) <= 0:
        return np.nan

    return (r_pref - r_opp) / (r_pref + r_opp)


def main():
    print("===== Compute tuning summary =====")

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
                labeled, unit_id, trial_id, BASELINE_WINDOW[0], BASELINE_WINDOW[1]
            )
            early_count = count_spikes_in_window(
                labeled, unit_id, trial_id, EARLY_WINDOW[0], EARLY_WINDOW[1]
            )
            sustained_count = count_spikes_in_window(
                labeled, unit_id, trial_id,
                SUSTAINED_RESPONSE_WINDOW[0],
                SUSTAINED_RESPONSE_WINDOW[1],
            )
            moving_count = count_spikes_in_window(
                labeled, unit_id, trial_id, MOVING_WINDOW[0], MOVING_WINDOW[1]
            )

            # Static window is directly from trial_table, relative to moving onset.
            static_window = (
                tr["static_start_sec"] - tr["moving_start_sec"],
                tr["static_end_sec"] - tr["moving_start_sec"],
            )
            static_count = count_spikes_in_window(
                labeled, unit_id, trial_id, static_window[0], static_window[1]
            )

            baseline_fr = firing_rate(baseline_count, BASELINE_WINDOW)
            early_fr = firing_rate(early_count, EARLY_WINDOW)
            sustained_fr = firing_rate(sustained_count, SUSTAINED_RESPONSE_WINDOW)
            moving_fr = firing_rate(moving_count, MOVING_WINDOW)
            static_fr = firing_rate(static_count, static_window)

            rows.append(
                {
                    "unit_id": unit_id,
                    "trial_id": trial_id,
                    "direction": tr["direction"],
                    "orientation": tr["orientation"],
                    "pattern": tr["pattern"],
                    "speed": tr["speed"],

                    "baseline_count": baseline_count,
                    "static_count": static_count,
                    "early_count": early_count,
                    "sustained_count": sustained_count,
                    "moving_count": moving_count,

                    "baseline_fr": baseline_fr,
                    "static_fr": static_fr,
                    "early_fr": early_fr,
                    "sustained_fr": sustained_fr,
                    "moving_fr": moving_fr,

                    "moving_minus_baseline": moving_fr - baseline_fr,
                    "moving_minus_static": moving_fr - static_fr,
                }
            )

    unit_trial_summary = pd.DataFrame(rows)

    # -----------------------------
    # Unit × condition summary
    # -----------------------------

    group_cols = ["unit_id", "direction", "orientation", "pattern", "speed"]

    unit_condition_summary = (
        unit_trial_summary
        .groupby(group_cols, dropna=False)
        .agg(
            n_trials=("trial_id", "nunique"),

            baseline_fr=("baseline_fr", "mean"),
            static_fr=("static_fr", "mean"),
            early_fr=("early_fr", "mean"),
            sustained_fr=("sustained_fr", "mean"),
            moving_fr=("moving_fr", "mean"),

            moving_minus_baseline=("moving_minus_baseline", "mean"),
            moving_minus_static=("moving_minus_static", "mean"),
        )
        .reset_index()
    )

    # -----------------------------
    # Unit tuning summary
    # -----------------------------

    tuning_rows = []

    for unit_id, df_unit in unit_condition_summary.groupby("unit_id"):

        # Use baseline-subtracted moving response for tuning.
        resp = (
            df_unit[["direction", "moving_minus_baseline"]]
            .rename(columns={"moving_minus_baseline": "response"})
            .copy()
        )

        # Negative responses make vector metrics hard to interpret.
        # First-pass choice: floor at 0.
        resp["response"] = resp["response"].clip(lower=0)

        if resp["response"].sum() > 0:
            pref_row = resp.loc[resp["response"].idxmax()]
            preferred_direction = pref_row["direction"]
            preferred_response = pref_row["response"]
        else:
            preferred_direction = np.nan
            preferred_response = np.nan

        vector_sum_direction, vector_strength = compute_vector_strength(resp)
        dsi = compute_dsi(resp)

        tuning_rows.append(
            {
                "unit_id": unit_id,

                "mean_baseline_fr": df_unit["baseline_fr"].mean(),
                "mean_static_fr": df_unit["static_fr"].mean(),
                "mean_moving_fr": df_unit["moving_fr"].mean(),

                "preferred_direction": preferred_direction,
                "preferred_response": preferred_response,
                "vector_sum_direction": vector_sum_direction,
                "vector_strength": vector_strength,
                "dsi": dsi,
            }
        )

    unit_tuning_summary = pd.DataFrame(tuning_rows)

    # Add simple unit metadata if available
    unit_tuning_summary = unit_tuning_summary.merge(
        units,
        on="unit_id",
        how="left",
    )

    # -----------------------------
    # Save
    # -----------------------------

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    condition_summary_path = ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv"
    tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"

    unit_trial_summary.to_csv(trial_summary_path, index=False)
    unit_condition_summary.to_csv(condition_summary_path, index=False)
    unit_tuning_summary.to_csv(tuning_summary_path, index=False)

    print("\n===== Saved =====")
    print(trial_summary_path)
    print(condition_summary_path)
    print(tuning_summary_path)

    print("\nFirst few condition summary rows:")
    print(unit_condition_summary.head())

    print("\nFirst few tuning summary rows:")
    print(unit_tuning_summary.head())


if __name__ == "__main__":
    main()