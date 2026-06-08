# Screen- and speed-specific tuning summary for
# single-screen 8-direction / 2-speed stimulus.
#
# Main outputs:
#   unit_trial_summary.csv
#       unit × trial
#
#   unit_condition_summary.csv
#       unit × screen_role × speed × direction
#
#   unit_tuning_summary.csv
#       unit × screen_role × speed
#
#   unit_speed_tuning_summary.csv
#       same as unit_tuning_summary.csv, kept for backward compatibility
#
# Important:
#   All tuning/classification metrics are computed separately for each:
#       screen_role × speed

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    BASELINE_WINDOW,
    EARLY_WINDOW,
    SUSTAINED_RESPONSE_WINDOW,
    MOVING_WINDOW,
)

BASELINE_POOLING_MODE = "screen_global_static"
# BASELINE_POOLING_MODE = "screen_axis_static"
# BASELINE_POOLING_MODE = "trail_window_specific"


# Small tolerance to avoid classifying tiny numerical/noise-level deviations as
# true excitation or suppression. Adjust if needed.
EPS = 0  # spikes/s


# =====================
# Helpers
# =====================


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

    direction_response must have columns:
        direction, response

    response should be non-negative for interpretability.
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


def compute_dsi_details(direction_response):
    """
    Compute preferred direction, opposite direction, and DSI.

    DSI = (R_pref - R_opp) / (R_pref + R_opp)

    direction_response must have columns:
        direction, response

    response should be non-negative. For suppressed neurons, pass suppression
    strength = max(-(moving-baseline), 0), not the signed negative response.
    """
    df = direction_response.dropna(subset=["direction", "response"]).copy()

    if df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    df["direction"] = df["direction"].astype(float) % 360
    df["response"] = df["response"].astype(float)

    if df["response"].sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    pref_row = df.loc[df["response"].idxmax()]
    pref_dir = float(pref_row["direction"])
    r_pref = float(pref_row["response"])

    nominal_opp_dir = (pref_dir + 180) % 360
    df["opp_distance"] = np.abs(
        ((df["direction"] - nominal_opp_dir + 180) % 360) - 180
    )

    opp_row = df.loc[df["opp_distance"].idxmin()]
    measured_opp_dir = float(opp_row["direction"])
    r_opp = float(opp_row["response"])

    if (r_pref + r_opp) <= 0:
        dsi = np.nan
    else:
        dsi = (r_pref - r_opp) / (r_pref + r_opp)

    return pref_dir, r_pref, measured_opp_dir, r_opp, dsi


def classify_signed_response(df_condition_unit, eps=EPS):
    """
    Classify one unit within one screen_role × speed based on
    direction-mean moving-baseline.
    """
    r = df_condition_unit["moving_minus_baseline"].dropna().to_numpy(dtype=float)

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


def get_speed_value(trial_row):
    """Use speed_label when available; otherwise speed."""
    if "speed_label" in trial_row.index and not pd.isna(trial_row["speed_label"]):
        return trial_row["speed_label"]
    if "speed" in trial_row.index and not pd.isna(trial_row["speed"]):
        return trial_row["speed"]
    return "speed_unknown"


def get_screen_role_value(trial_row):
    """Use screen_role when available; otherwise unknown."""
    if "screen_role" in trial_row.index and not pd.isna(trial_row["screen_role"]):
        return str(trial_row["screen_role"]).strip()
    return "unknown"


def get_first_nonnull(series, default=np.nan):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return default
    return s.iloc[0]


# =====================
# Main
# =====================


def main():
    print("===== Compute screen- and speed-specific tuning summary =====")

    labeled_path = ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv"
    trial_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    labeled = pd.read_csv(labeled_path)
    trials = pd.read_csv(trial_path)
    units = pd.read_csv(units_path)

    if "screen_role" not in trials.columns:
        print(
            "\nWarning: trial_table.csv has no screen_role column. "
            "Using screen_role='unknown' for all trials.\n"
        )
        trials["screen_role"] = "unknown"

    if "screen_role" not in labeled.columns:
        print(
            "\nWarning: labeled_spikes.csv has no screen_role column. "
            "This is okay for counting by trial_id, but run updated SB04c "
            "if you want labeled_spikes.csv to carry screen metadata.\n"
        )
        labeled["screen_role"] = "unknown"

    unit_ids = units["unit_id"].tolist()

    print(f"Units: {len(unit_ids)}")
    print(f"Trials: {len(trials)}")
    print(f"Labeled spikes: {len(labeled)}")

    print("\nTrial counts by screen_role:")
    print(trials["screen_role"].value_counts(dropna=False))

    # Normalize common string columns to avoid mismatches like "front " vs "front".
    trials["screen_role"] = trials["screen_role"].astype(str).str.strip()
    if "speed" in trials.columns:
        trials["speed"] = trials["speed"].astype(str).str.strip()
    if "speed_label" in trials.columns:
        trials["speed_label"] = trials["speed_label"].astype(str).str.strip()

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

            static_window = (
                tr["static_start_sec"] - tr["moving_start_sec"],
                tr["static_end_sec"] - tr["moving_start_sec"],
            )

            static_count = count_spikes_in_window(
                labeled,
                unit_id,
                trial_id,
                static_window[0],
                static_window[1],
            )

            baseline_fr = firing_rate(baseline_count, BASELINE_WINDOW)
            early_fr = firing_rate(early_count, EARLY_WINDOW)
            sustained_fr = firing_rate(sustained_count, SUSTAINED_RESPONSE_WINDOW)
            moving_fr = firing_rate(moving_count, MOVING_WINDOW)
            static_fr = firing_rate(static_count, static_window)

            speed_value = get_speed_value(tr)
            screen_role_value = get_screen_role_value(tr)

            rows.append(
                {
                    "unit_id": unit_id,
                    "trial_id": trial_id,

                    # Screen / segment identity
                    "screen_role": screen_role_value,
                    "original_segment_index": tr.get("original_segment_index", np.nan),
                    "active_monitor_config_index": tr.get("active_monitor_config_index", np.nan),
                    "active_monitor_label": tr.get("active_monitor_label", np.nan),
                    "active_screen_number": tr.get("active_screen_number", np.nan),

                    # Stimulus identity
                    "direction": tr.get("direction", np.nan),
                    "orientation": tr.get("orientation", np.nan),
                    "pattern": tr.get("pattern", np.nan),
                    "speed": speed_value,
                    "speed_label": tr.get("speed_label", speed_value),
                    "speed_deg_per_sec": tr.get("speed_deg_per_sec", np.nan),
                    "tf_hz": tr.get("tf_hz", np.nan),
                    "sf_cpd": tr.get("sf_cpd", np.nan),

                    # Counts
                    "baseline_count": baseline_count,
                    "static_count": static_count,
                    "early_count": early_count,
                    "sustained_count": sustained_count,
                    "moving_count": moving_count,

                    # FRs
                    "baseline_fr": baseline_fr,
                    "static_fr": static_fr,
                    "early_fr": early_fr,
                    "sustained_fr": sustained_fr,
                    "moving_fr": moving_fr,

                    # Response metrics
                    "moving_minus_baseline": moving_fr - baseline_fr,
                    "moving_minus_static": moving_fr - static_fr,
                }
            )

    unit_trial_summary = pd.DataFrame(rows)

    # Normalize again after creation.
    unit_trial_summary["screen_role"] = (
        unit_trial_summary["screen_role"].astype(str).str.strip()
    )
    unit_trial_summary["speed"] = (
        unit_trial_summary["speed"].astype(str).str.strip()
    )
    unit_trial_summary["speed_label"] = (
        unit_trial_summary["speed_label"].astype(str).str.strip()
    )

    # Preserve original trial-level values
    unit_trial_summary["baseline_fr_original"] = unit_trial_summary["baseline_fr"]
    unit_trial_summary["static_fr_original"] = unit_trial_summary["static_fr"]
    unit_trial_summary["moving_minus_baseline_original"] = unit_trial_summary["moving_minus_baseline"]

    if BASELINE_POOLING_MODE == "screen_axis_static":
        unit_trial_summary["direction_float"] = (
            pd.to_numeric(unit_trial_summary["direction"], errors="coerce") % 360
        )

        # Opposite directions share the same static orientation/axis.
        # Examples: 45 and 225 -> 45; 90 and 270 -> 90.
        unit_trial_summary["direction_axis"] = (
            unit_trial_summary["direction_float"] % 180
        )

        pool_cols = ["unit_id", "screen_role", "direction_axis"]

        pooled = (
            unit_trial_summary
            .groupby(pool_cols, dropna=False)["static_fr"]
            .transform("mean")
        )

        unit_trial_summary["pooled_baseline_fr"] = pooled
        unit_trial_summary["baseline_pooling_mode"] = BASELINE_POOLING_MODE

        # Replace the response metric used downstream
        unit_trial_summary["baseline_fr"] = unit_trial_summary["pooled_baseline_fr"]
        unit_trial_summary["moving_minus_baseline"] = (
            unit_trial_summary["moving_fr"] - unit_trial_summary["pooled_baseline_fr"]
        )

    if BASELINE_POOLING_MODE == "screen_global_static":
        pool_cols = ["unit_id", "screen_role"]

        pooled = (
            unit_trial_summary
            .groupby(pool_cols, dropna=False)["static_fr"]
            .transform("mean")
        )

        unit_trial_summary["pooled_baseline_fr"] = pooled
        unit_trial_summary["baseline_pooling_mode"] = BASELINE_POOLING_MODE

        # Replace the response metric used downstream
        unit_trial_summary["baseline_fr"] = unit_trial_summary["pooled_baseline_fr"]
        unit_trial_summary["moving_minus_baseline"] = (
            unit_trial_summary["moving_fr"] - unit_trial_summary["pooled_baseline_fr"]
        )

    # -----------------------------
    # Unit × screen_role × speed × direction summary
    # -----------------------------

    group_cols = [
        "unit_id",
        "screen_role",
        "speed",
        "direction",
        "orientation",
        "pattern",
    ]

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

            # New: pooled static baseline used for analysis/plotting
            pooled_static_fr=("pooled_baseline_fr", "mean"),
            baseline_pooling_mode=("baseline_pooling_mode", "first"),

            moving_minus_baseline=("moving_minus_baseline", "mean"),
            moving_minus_static=("moving_minus_static", "mean"),

            speed_label=("speed_label", "first"),
            speed_deg_per_sec=("speed_deg_per_sec", "first"),
            tf_hz=("tf_hz", "first"),
            sf_cpd=("sf_cpd", "first"),

            original_segment_index=("original_segment_index", "first"),
            active_monitor_config_index=("active_monitor_config_index", "first"),
            active_monitor_label=("active_monitor_label", "first"),
            active_screen_number=("active_screen_number", "first"),
        )
        .reset_index()
    )

    unit_condition_summary["motion_baseline_positive"] = (
        unit_condition_summary["moving_minus_baseline"].clip(lower=0)
    )

    unit_condition_summary["motion_baseline_negative_strength"] = (
        -unit_condition_summary["moving_minus_baseline"].clip(upper=0)
    )

    # -----------------------------
    # Unit × screen_role × speed tuning summary
    # -----------------------------

    tuning_rows = []

    tuning_group_cols = ["unit_id", "screen_role", "speed"]

    for (unit_id, screen_role, speed), df_condition_unit in unit_condition_summary.groupby(
        tuning_group_cols,
        dropna=False,
    ):
        df_condition_unit = df_condition_unit.sort_values("direction").copy()

        response_class, n_pos, n_neg = classify_signed_response(df_condition_unit)

        moving_resp = (
            df_condition_unit[["direction", "moving_fr"]]
            .rename(columns={"moving_fr": "response"})
            .copy()
        )

        excitatory_resp = (
            df_condition_unit[["direction", "motion_baseline_positive"]]
            .rename(columns={"motion_baseline_positive": "response"})
            .copy()
        )

        suppressive_resp = (
            df_condition_unit[["direction", "motion_baseline_negative_strength"]]
            .rename(columns={"motion_baseline_negative_strength": "response"})
            .copy()
        )

        if response_class == "all_excited_or_nonnegative":
            pd_method = "motion_minus_baseline_positive"
            primary_resp = excitatory_resp

            (
                preferred_direction,
                preferred_response_used,
                opposite_direction,
                opposite_response_used,
                dsi,
            ) = compute_dsi_details(primary_resp)

            preferred_response_signed = preferred_response_used
            vector_sum_direction, vector_strength = compute_vector_strength(primary_resp)

        elif response_class == "all_suppressed_or_nonpositive":
            pd_method = "suppression_strength"
            primary_resp = suppressive_resp

            (
                preferred_direction,
                preferred_response_used,
                opposite_direction,
                opposite_response_used,
                dsi,
            ) = compute_dsi_details(primary_resp)

            if not np.isnan(preferred_direction):
                signed_match = df_condition_unit.loc[
                    df_condition_unit["direction"].astype(float) % 360
                    == float(preferred_direction) % 360,
                    "moving_minus_baseline",
                ]
                preferred_response_signed = (
                    float(signed_match.iloc[0]) if len(signed_match) > 0 else np.nan
                )
            else:
                preferred_response_signed = np.nan

            vector_sum_direction, vector_strength = compute_vector_strength(primary_resp)

        elif response_class == "mixed_excited_and_suppressed":
            pd_method = "pure_moving_fr_for_mixed"
            primary_resp = moving_resp

            (
                preferred_direction,
                preferred_response_used,
                opposite_direction,
                opposite_response_used,
                dsi,
            ) = compute_dsi_details(primary_resp)

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

        (
            moving_pd,
            moving_pref_response,
            moving_opp_dir,
            moving_opp_response,
            moving_dsi,
        ) = compute_dsi_details(moving_resp)

        moving_vector_direction, moving_vector_strength = compute_vector_strength(
            moving_resp
        )

        (
            excitation_pd,
            excitation_pref_response,
            excitation_opp_dir,
            excitation_opp_response,
            excitation_dsi,
        ) = compute_dsi_details(excitatory_resp)

        excitation_vector_direction, excitation_vector_strength = (
            compute_vector_strength(excitatory_resp)
        )

        (
            suppression_pd,
            suppression_pref_response,
            suppression_opp_dir,
            suppression_opp_response,
            suppression_dsi,
        ) = compute_dsi_details(suppressive_resp)

        suppression_vector_direction, suppression_vector_strength = (
            compute_vector_strength(suppressive_resp)
        )

        tuning_rows.append(
            {
                "unit_id": unit_id,

                # New grouping dimension
                "screen_role": screen_role,

                # Keep screen/segment metadata
                "original_segment_index": get_first_nonnull(
                    df_condition_unit["original_segment_index"]
                ),
                "active_monitor_config_index": get_first_nonnull(
                    df_condition_unit["active_monitor_config_index"]
                ),
                "active_monitor_label": get_first_nonnull(
                    df_condition_unit["active_monitor_label"]
                ),
                "active_screen_number": get_first_nonnull(
                    df_condition_unit["active_screen_number"]
                ),

                # Speed metadata
                "speed": speed,
                "speed_label": get_first_nonnull(
                    df_condition_unit["speed_label"],
                    default=speed,
                ),
                "speed_deg_per_sec": get_first_nonnull(
                    df_condition_unit["speed_deg_per_sec"]
                ),
                "tf_hz": get_first_nonnull(df_condition_unit["tf_hz"]),
                "sf_cpd": get_first_nonnull(df_condition_unit["sf_cpd"]),

                # Mean FR / response
                "mean_baseline_fr": df_condition_unit["baseline_fr"].mean(),
                "mean_static_fr": df_condition_unit["static_fr"].mean(),
                "mean_moving_fr": df_condition_unit["moving_fr"].mean(),
                "mean_moving_minus_baseline": df_condition_unit[
                    "moving_minus_baseline"
                ].mean(),
                "mean_moving_minus_static": df_condition_unit[
                    "moving_minus_static"
                ].mean(),

                # Classification
                "response_class": response_class,
                "n_positive_directions": n_pos,
                "n_negative_directions": n_neg,
                "classification_eps_spikes_per_sec": EPS,

                # Primary PD/DSI based on response class
                "pd_method": pd_method,
                "preferred_direction": preferred_direction,
                "preferred_response_used_for_dsi": preferred_response_used,
                "preferred_response_signed": preferred_response_signed,
                "opposite_direction": opposite_direction,
                "opposite_response_used_for_dsi": opposite_response_used,
                "dsi": dsi,
                "vector_sum_direction": vector_sum_direction,
                "vector_strength": vector_strength,

                # Pure moving FR metrics
                "moving_fr_preferred_direction": moving_pd,
                "moving_fr_preferred_response": moving_pref_response,
                "moving_fr_opposite_direction": moving_opp_dir,
                "moving_fr_opposite_response": moving_opp_response,
                "moving_fr_dsi": moving_dsi,
                "moving_fr_vector_direction": moving_vector_direction,
                "moving_fr_vector_strength": moving_vector_strength,

                # Excitation-only metrics
                "excitation_preferred_direction": excitation_pd,
                "excitation_preferred_response": excitation_pref_response,
                "excitation_opposite_direction": excitation_opp_dir,
                "excitation_opposite_response": excitation_opp_response,
                "excitation_dsi": excitation_dsi,
                "excitation_vector_direction": excitation_vector_direction,
                "excitation_vector_strength": excitation_vector_strength,

                # Suppression-only metrics
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

    # Add simple unit metadata if available. Because unit_tuning_summary is
    # unit × screen_role × speed, metadata is repeated once per screen-speed.
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
    speed_tuning_summary_path = ANALYSIS_OUTPUT_DIR / "unit_speed_tuning_summary.csv"

    unit_trial_summary.to_csv(trial_summary_path, index=False)
    unit_condition_summary.to_csv(condition_summary_path, index=False)
    unit_tuning_summary.to_csv(tuning_summary_path, index=False)
    unit_tuning_summary.to_csv(speed_tuning_summary_path, index=False)

    print("\n===== Saved =====")
    print(trial_summary_path)
    print(condition_summary_path)
    print(tuning_summary_path)
    print(speed_tuning_summary_path)

    print("\nUnit trial summary counts by screen_role:")
    print(unit_trial_summary.groupby("screen_role", dropna=False).size())

    print("\nCondition summary counts by screen_role and speed:")
    print(
        unit_condition_summary
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    print("\nTuning summary counts by screen_role and speed:")
    print(
        unit_tuning_summary
        .groupby(["screen_role", "speed"], dropna=False)
        .size()
    )

    print("\nFirst few condition summary rows:")
    print(unit_condition_summary.head())

    print("\nFirst few screen-speed-specific tuning rows:")
    print(unit_tuning_summary.head())


if __name__ == "__main__":
    main()