from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    STIMLOG_RUNS,
    MOTION_STATE,
    EXPECTED_MOTION_TTL_COUNT,

)


# =====================
# Screen role mapping
# =====================
# Based on your three-screen mapping:
#   ACTIVE_MONITOR_CONFIGS = [1, 0, 3]
#   left  = MONITOR_CONFIGS[1] = screen1
#   front = MONITOR_CONFIGS[0] = screen0
#   right = MONITOR_CONFIGS[3] = screen3
#
# So for single-screen runs:
#   Active_monitor_config_index 1 -> left
#   Active_monitor_config_index 0 -> front
#   Active_monitor_config_index 3 -> right

SCREEN_ROLE_BY_CONFIG_INDEX = {
    1: "left",
    0: "front",
    3: "right",
}

SCREEN_ROLE_BY_MONITOR_LABEL = {
    "screen1": "left",
    "screen0": "front",
    "screen3": "right",
}

SCREEN_ROLE_BY_SCREEN_NUMBER = {
    1: "left",
    0: "front",
    3: "right",
}

# If auto-detection fails, set this manually:
# MANUAL_SCREEN_ROLE = "front"
# MANUAL_SCREEN_ROLE = "left"
# MANUAL_SCREEN_ROLE = "right"
MANUAL_SCREEN_ROLE = None


def normalize_state(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def get_screen_role(row):
    """
    Infer screen_role for one stimlog row.

    Priority:
        1. MANUAL_SCREEN_ROLE
        2. Active_monitor_config_index
        3. Active_monitor_label
        4. Active_screen_number
    """
    if MANUAL_SCREEN_ROLE is not None:
        return MANUAL_SCREEN_ROLE

    if "Active_monitor_config_index" in row.index and not pd.isna(row["Active_monitor_config_index"]):
        try:
            idx = int(row["Active_monitor_config_index"])
            if idx in SCREEN_ROLE_BY_CONFIG_INDEX:
                return SCREEN_ROLE_BY_CONFIG_INDEX[idx]
        except Exception:
            pass

    if "Active_monitor_label" in row.index and not pd.isna(row["Active_monitor_label"]):
        label = str(row["Active_monitor_label"]).strip()
        if label in SCREEN_ROLE_BY_MONITOR_LABEL:
            return SCREEN_ROLE_BY_MONITOR_LABEL[label]

    if "Active_screen_number" in row.index and not pd.isna(row["Active_screen_number"]):
        try:
            screen_number = int(row["Active_screen_number"])
            if screen_number in SCREEN_ROLE_BY_SCREEN_NUMBER:
                return SCREEN_ROLE_BY_SCREEN_NUMBER[screen_number]
        except Exception:
            pass

    return "unknown"


def main():

    print("===== Building 8-direction single-screen trial table =====")

    # -----------------------------
    # Load files
    # -----------------------------

    stimlog = pd.read_csv(STIMLOG_PATH)

    ttl_path = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    ttl_df = pd.read_csv(ttl_path)

    print(f"Stimlog rows: {len(stimlog)}")
    print(f"TTL rows: {len(ttl_df)}")

    # -----------------------------
    # Keep motion rows only
    # -----------------------------

    motion_rows = (
        stimlog.loc[
            stimlog["Stimulus_state"].map(normalize_state)
            == normalize_state(MOTION_STATE)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Motion rows: {len(motion_rows)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

    # -----------------------------
    # Basic sanity check
    # -----------------------------

    if len(ttl_df) < len(motion_rows):
        raise ValueError(
            "Not enough TTL events for motion rows. "
            "Check segment selection or stimlog."
        )

    # -----------------------------
    # Match TTLs to motion rows
    # -----------------------------

    ttl_match = ttl_df.iloc[: len(motion_rows)].copy()

    # -----------------------------
    # Compute offset
    # -----------------------------

    first_motion_onset = motion_rows.loc[0, "Stimulus_start"]

    if "event_time_sec" in ttl_match.columns:
        first_ttl_time = ttl_match.loc[0, "event_time_sec"]
        ttl_time_col = "event_time_sec"
    elif "concat_time_sec" in ttl_match.columns:
        first_ttl_time = ttl_match.loc[0, "concat_time_sec"]
        ttl_time_col = "concat_time_sec"
    else:
        raise ValueError(
            "TTL file must contain either event_time_sec or concat_time_sec."
        )

    recording_offset = first_ttl_time - first_motion_onset

    print(f"\nRecording offset: {recording_offset:.6f} sec")
    print(f"Using TTL time column: {ttl_time_col}")

    # -----------------------------
    # Alignment QC
    # -----------------------------

    alignment_qc = pd.DataFrame(
        {
            "trial_id": np.arange(len(motion_rows)),
            "stimlog_motion_start_sec": motion_rows["Stimulus_start"].values,
            "ttl_time_sec": ttl_match[ttl_time_col].values,
        }
    )

    if "ttl_index_global" in ttl_match.columns:
        alignment_qc["ttl_index_global"] = ttl_match["ttl_index_global"].values

    if "original_segment_index" in ttl_match.columns:
        alignment_qc["original_segment_index"] = ttl_match["original_segment_index"].values

    alignment_qc["predicted_ttl_sec"] = (
        alignment_qc["stimlog_motion_start_sec"] + recording_offset
    )

    alignment_qc["residual_ms"] = (
        alignment_qc["ttl_time_sec"]
        - alignment_qc["predicted_ttl_sec"]
    ) * 1000

    print("\nAlignment residuals (ms):")
    print(alignment_qc["residual_ms"].describe())

    # -----------------------------
    # Shift stimlog into neural / concat time
    # -----------------------------

    updated_stimlog = stimlog.copy()

    updated_stimlog["Stimulus_start"] = (
        updated_stimlog["Stimulus_start"] + recording_offset
    )

    updated_stimlog["Stimulus_end"] = (
        updated_stimlog["Stimulus_end"] + recording_offset
    )

    updated_stimlog["screen_role"] = updated_stimlog.apply(get_screen_role, axis=1)

    # -----------------------------
    # Build trial table
    # -----------------------------
    # Single-screen stimulus structure:
    # Each trial:
    # static -> moving
    #
    # TTL is sent at moving onset, so ttl_match[trial_id]
    # corresponds to the moving row of each detected trial.

    trial_rows = []
    trial_id = 0

    for i in range(len(updated_stimlog) - 1):

        row_static = updated_stimlog.iloc[i]
        row_moving = updated_stimlog.iloc[i + 1]

        if (
            normalize_state(row_static["Stimulus_state"]) == "static"
            and normalize_state(row_moving["Stimulus_state"]) == "moving"
        ):

            ttl_row = ttl_match.iloc[trial_id]

            trial_rows.append(
                {
                    "trial_id": trial_id,

                    # Recording / segment identity if available
                    "original_segment_index": ttl_row.get("original_segment_index", np.nan),
                    "ttl_index_global": ttl_row.get("ttl_index_global", np.nan),

                    # Screen identity
                    "screen_role": row_moving.get("screen_role", "unknown"),
                    "active_monitor_config_index": row_moving.get("Active_monitor_config_index", np.nan),
                    "active_monitor_label": row_moving.get("Active_monitor_label", np.nan),
                    "active_screen_number": row_moving.get("Active_screen_number", np.nan),

                    # Core stimulus identity
                    "direction": row_moving["Direction_deg"],
                    "orientation": row_moving["Stimulus_orientation"],
                    "pattern": row_moving["Pattern"],

                    # Speed information
                    "speed": row_moving["Speed_label"],
                    "speed_label": row_moving["Speed_label"],
                    "speed_deg_per_sec": row_moving["Speed_deg_per_sec"],
                    "tf_hz": row_moving["GratingStim_TF_Hz"],
                    "sf_cpd": row_moving["GratingStim_SF_cpd"],

                    # State timing in neural-recording / concat time
                    "static_start_sec": row_static["Stimulus_start"],
                    "static_end_sec": row_static["Stimulus_end"],

                    "moving_start_sec": row_moving["Stimulus_start"],
                    "moving_end_sec": row_moving["Stimulus_end"],

                    # Matched Neuralynx TTL time
                    "ttl_time_sec": ttl_row[ttl_time_col],
                }
            )

            trial_id += 1

    if trial_id != len(motion_rows):
        print(
            f"Warning: detected trials ({trial_id}) != motion rows ({len(motion_rows)}). "
            "Check whether stimlog contains incomplete static/moving pairs."
        )

    trial_table = pd.DataFrame(trial_rows)

    # -----------------------------
    # Screen role QC
    # -----------------------------

    print("\nScreen role counts in trial table:")
    if "screen_role" in trial_table.columns:
        print(trial_table["screen_role"].value_counts(dropna=False))
    else:
        print("No screen_role column found.")

    if "unknown" in trial_table.get("screen_role", pd.Series(dtype=str)).astype(str).values:
        print(
            "\nWarning: some trials have screen_role='unknown'. "
            "Check Active_monitor_config_index / Active_monitor_label / Active_screen_number, "
            "or set MANUAL_SCREEN_ROLE."
        )

    # -----------------------------
    # Save outputs
    # -----------------------------

    alignment_qc_path = ANALYSIS_OUTPUT_DIR / "alignment_qc.csv"
    updated_stimlog_path = ANALYSIS_OUTPUT_DIR / "updated_stimlog.csv"
    trial_table_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"

    alignment_qc.to_csv(alignment_qc_path, index=False)
    updated_stimlog.to_csv(updated_stimlog_path, index=False)
    trial_table.to_csv(trial_table_path, index=False)

    print("\n===== Saved =====")
    print(alignment_qc_path)
    print(updated_stimlog_path)
    print(trial_table_path)

    print("\nFirst few trials:")
    print(trial_table.head())


if __name__ == "__main__":
    main()