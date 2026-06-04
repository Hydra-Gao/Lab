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

EPS = 0.1  # spikes/s, can adjust later


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


# def compute_dsi(direction_response):
#     """
#     Direction Selectivity Index:
#     DSI = (R_pref - R_opp) / (R_pref + R_opp)

#     Opposite direction is preferred + 180 degrees.
#     """
#     df = direction_response.dropna(subset=["direction", "response"]).copy()

#     if df.empty:
#         return np.nan

#     df["direction"] = df["direction"].astype(float) % 360

#     pref_row = df.loc[df["response"].idxmax()]
#     pref_dir = pref_row["direction"]
#     r_pref = pref_row["response"]

#     opp_dir = (pref_dir + 180) % 360

#     # Find the measured direction closest to the opposite direction
#     df["opp_distance"] = np.abs(((df["direction"] - opp_dir + 180) % 360) - 180)
#     r_opp = df.loc[df["opp_distance"].idxmin(), "response"]

#     if (r_pref + r_opp) <= 0:
#         return np.nan

#     return (r_pref - r_opp) / (r_pref + r_opp)

def compute_dsi_details(direction_response):
    """
    Compute DSI and return preferred/opposite direction details.

    direction_response:
        columns: direction, response
        response must be non-negative for DSI interpretation.
    """
    df = direction_response.dropna(subset=["direction", "response"]).copy()

    if df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    df["direction"] = df["direction"].astype(float) % 360
    df["response"] = df["response"].astype(float)

    if df["response"].sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    pref_row = df.loc[df["response"].idxmax()]
    pref_dir = pref_row["direction"]
    r_pref = pref_row["response"]

    opp_dir = (pref_dir + 180) % 360
    df["opp_distance"] = np.abs(((df["direction"] - opp_dir + 180) % 360) - 180)

    opp_row = df.loc[df["opp_distance"].idxmin()]
    measured_opp_dir = opp_row["direction"]
    r_opp = opp_row["response"]

    if (r_pref + r_opp) <= 0:
        dsi = np.nan
    else:
        dsi = (r_pref - r_opp) / (r_pref + r_opp)

    return pref_dir, r_pref, measured_opp_dir, r_opp, dsi


def classify_signed_response(direction_summary, eps=EPS):
    """
    Classify unit based on direction-mean moving_minus_baseline.

    all_excited_or_nonnegative:
        no clearly negative directions, at least one positive direction

    all_suppressed_or_nonpositive:
        no clearly positive directions, at least one negative direction

    mixed_excited_and_suppressed:
        at least one positive and one negative direction

    flat_or_zero:
        all directions are close to zero
    """
    r = direction_summary["moving_minus_baseline"].dropna().to_numpy(dtype=float)

    if len(r) == 0:
        return "no_data", 0, 0

    n_pos = int(np.sum(r > eps))
    n_neg = int(np.sum(r < -eps))

    if n_pos > 0 and n_neg == 0:
        return "all_excited_or_nonnegative", n_pos, n_neg

    if n_neg > 0 and n_pos == 0:
        return "all_suppressed_or_nonpositive", n_pos, n_neg

    if n_pos > 0 and n_neg > 0:
        return "mixed_excited_and_suppressed", n_pos, n_neg

    return "flat_or_zero", n_pos, n_neg


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
    # Unit × direction summary
    # pooled across speed
    # -----------------------------

    unit_direction_summary = (
        unit_trial_summary
        .groupby(["unit_id", "direction"], dropna=False)
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

    # These two columns are for SB11 positive/negative polar plot.
    unit_direction_summary["motion_baseline_positive"] = (
        unit_direction_summary["moving_minus_baseline"].clip(lower=0)
    )

    unit_direction_summary["motion_baseline_negative_strength"] = (
        -unit_direction_summary["moving_minus_baseline"].clip(upper=0)
    )

    
    # -----------------------------
    # Unit tuning summary
    # -----------------------------

    tuning_rows = []

    for unit_id, df_unit in unit_direction_summary.groupby("unit_id"):

        df_unit = df_unit.sort_values("direction").copy()

        response_class, n_pos, n_neg = classify_signed_response(df_unit)

        # Signed motion-baseline response
        signed_resp = df_unit[["direction", "moving_minus_baseline"]].rename(
            columns={"moving_minus_baseline": "response"}
        ).copy()

        # Pure moving FR response
        moving_resp = df_unit[["direction", "moving_fr"]].rename(
            columns={"moving_fr": "response"}
        ).copy()

        # Excitatory component: max(moving-baseline, 0)
        excitatory_resp = df_unit[["direction", "motion_baseline_positive"]].rename(
            columns={"motion_baseline_positive": "response"}
        ).copy()

        # Suppressive component: max(-(moving-baseline), 0)
        suppressive_resp = df_unit[
            ["direction", "motion_baseline_negative_strength"]
        ].rename(
            columns={"motion_baseline_negative_strength": "response"}
        ).copy()

        # -------------------------
        # Decide primary PD / DSI
        # -------------------------

        if response_class == "all_excited_or_nonnegative":
            pd_method = "motion_minus_baseline_positive"
            primary_resp = excitatory_resp

            preferred_direction, preferred_response_used, opposite_direction, opposite_response_used, dsi = (
                compute_dsi_details(primary_resp)
            )

            # Signed response at PD is same sign as used response here.
            preferred_response_signed = preferred_response_used

            vector_sum_direction, vector_strength = compute_vector_strength(primary_resp)

        elif response_class == "all_suppressed_or_nonpositive":
            pd_method = "suppression_strength"
            primary_resp = suppressive_resp

            preferred_direction, preferred_response_used, opposite_direction, opposite_response_used, dsi = (
                compute_dsi_details(primary_resp)
            )

            # For suppressed units, the actual signed response at PD is negative.
            if not np.isnan(preferred_direction):
                signed_match = df_unit.loc[
                    df_unit["direction"].astype(float) % 360 == float(preferred_direction) % 360,
                    "moving_minus_baseline",
                ]
                preferred_response_signed = (
                    float(signed_match.iloc[0]) if len(signed_match) > 0 else np.nan
                )
            else:
                preferred_response_signed = np.nan

            vector_sum_direction, vector_strength = compute_vector_strength(primary_resp)

        elif response_class == "mixed_excited_and_suppressed":
            # For mixed units, primary PD/DSI comes from pure moving FR.
            # Positive and negative signed components are kept separately.
            pd_method = "pure_moving_fr_for_mixed"
            primary_resp = moving_resp

            preferred_direction, preferred_response_used, opposite_direction, opposite_response_used, dsi = (
                compute_dsi_details(primary_resp)
            )

            preferred_response_signed = np.nan
            vector_sum_direction, vector_strength = compute_vector_strength(primary_resp)

        else:
            pd_method = "none"
            preferred_direction = np.nan
            preferred_response_used = np.nan
            preferred_response_signed = np.nan
            opposite_direction = np.nan
            opposite_response_used = np.nan
            dsi = np.nan
            vector_sum_direction = np.nan
            vector_strength = np.nan

        # -------------------------
        # Always compute auxiliary metrics
        # -------------------------

        moving_pd, moving_pref_response, moving_opp_dir, moving_opp_response, moving_dsi = (
            compute_dsi_details(moving_resp)
        )
        moving_vector_direction, moving_vector_strength = compute_vector_strength(moving_resp)

        excitation_pd, excitation_pref_response, excitation_opp_dir, excitation_opp_response, excitation_dsi = (
            compute_dsi_details(excitatory_resp)
        )
        excitation_vector_direction, excitation_vector_strength = compute_vector_strength(excitatory_resp)

        suppression_pd, suppression_pref_response, suppression_opp_dir, suppression_opp_response, suppression_dsi = (
            compute_dsi_details(suppressive_resp)
        )
        suppression_vector_direction, suppression_vector_strength = compute_vector_strength(suppressive_resp)

        tuning_rows.append(
            {
                "unit_id": unit_id,

                "mean_baseline_fr": df_unit["baseline_fr"].mean(),
                "mean_static_fr": df_unit["static_fr"].mean(),
                "mean_moving_fr": df_unit["moving_fr"].mean(),
                "mean_moving_minus_baseline": df_unit["moving_minus_baseline"].mean(),
                "mean_moving_minus_static": df_unit["moving_minus_static"].mean(),

                "response_class": response_class,
                "n_positive_directions": n_pos,
                "n_negative_directions": n_neg,
                "classification_eps_spikes_per_sec": EPS,

                "pd_method": pd_method,
                "preferred_direction": preferred_direction,
                "preferred_response_used_for_dsi": preferred_response_used,
                "preferred_response_signed": preferred_response_signed,
                "opposite_direction": opposite_direction,
                "opposite_response_used_for_dsi": opposite_response_used,
                "dsi": dsi,
                "vector_sum_direction": vector_sum_direction,
                "vector_strength": vector_strength,

                # Pure moving FR tuning
                "moving_fr_preferred_direction": moving_pd,
                "moving_fr_preferred_response": moving_pref_response,
                "moving_fr_opposite_direction": moving_opp_dir,
                "moving_fr_opposite_response": moving_opp_response,
                "moving_fr_dsi": moving_dsi,
                "moving_fr_vector_direction": moving_vector_direction,
                "moving_fr_vector_strength": moving_vector_strength,

                # Excitatory signed component
                "excitation_preferred_direction": excitation_pd,
                "excitation_preferred_response": excitation_pref_response,
                "excitation_opposite_direction": excitation_opp_dir,
                "excitation_opposite_response": excitation_opp_response,
                "excitation_dsi": excitation_dsi,
                "excitation_vector_direction": excitation_vector_direction,
                "excitation_vector_strength": excitation_vector_strength,

                # Suppressive signed component
                "suppression_preferred_direction": suppression_pd,
                "suppression_preferred_response": suppression_pref_response,
                "suppression_opposite_direction": suppression_opp_dir,
                "suppression_opposite_response": suppression_opp_response,
                "suppression_dsi": suppression_dsi,
                "suppression_vector_direction": suppression_vector_direction,
                "suppression_vector_strength": suppression_vector_strength,
            }
        )

    unit_tuning_summary = pd.DataFrame(tuning_rows)


    # -----------------------------
    # Save
    # -----------------------------

    trial_summary_path = ANALYSIS_OUTPUT_DIR / "unit_trial_summary.csv"
    condition_summary_path = ANALYSIS_OUTPUT_DIR / "unit_condition_summary.csv"
    direction_summary_path = ANALYSIS_OUTPUT_DIR / "unit_direction_summary.csv"
    tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_tuning_summary.csv"

    unit_trial_summary.to_csv(trial_summary_path, index=False)
    unit_condition_summary.to_csv(condition_summary_path, index=False)
    unit_direction_summary.to_csv(direction_summary_path, index=False)
    unit_tuning_summary.to_csv(tuning_summary_path, index=False)

    print("\n===== Saved =====")
    print(trial_summary_path)
    print(condition_summary_path)
    print(direction_summary_path)
    print(tuning_summary_path)

    print("\nFirst few condition summary rows:")
    print(unit_condition_summary.head())

    print("\nFirst few tuning summary rows:")
    print(unit_tuning_summary.head())


if __name__ == "__main__":
    main()