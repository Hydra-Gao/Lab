# 02_build_trial_table.py

from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    STIMLOG_PATH,
    MOTION_STATE,
    EXPECTED_MOTION_TTL_COUNT,
)


def main():

    print("===== Building trial table =====")

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
        stimlog.loc[stimlog["Stimulus_state"] == MOTION_STATE]
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
    # First version:
    # assume the correct TTL block is the first N TTLs.
    # Later we can make this smarter if needed.

    ttl_match = ttl_df.iloc[: len(motion_rows)].copy()

    # -----------------------------
    # Compute offset
    # -----------------------------
    # TTL time = neural recording time
    # Stimulus_start = stimlog relative time
    #
    # recording_offset converts:
    # stimlog time --> neural recording time

    first_motion_onset = motion_rows.loc[0, "Stimulus_start"]
    first_ttl_time = ttl_match.loc[0, "event_time_sec"]

    recording_offset = first_ttl_time - first_motion_onset

    print(f"\nRecording offset: {recording_offset:.6f} sec")

    # -----------------------------
    # Alignment QC
    # -----------------------------

    alignment_qc = pd.DataFrame(
        {
            "trial_id": np.arange(len(motion_rows)),
            "stimlog_motion_start_sec": motion_rows["Stimulus_start"].values,
            "ttl_time_sec": ttl_match["event_time_sec"].values,
        }
    )

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
    # Shift stimlog into neural time
    # -----------------------------

    updated_stimlog = stimlog.copy()

    updated_stimlog["Stimulus_start"] = (
        updated_stimlog["Stimulus_start"] + recording_offset
    )

    updated_stimlog["Stimulus_end"] = (
        updated_stimlog["Stimulus_end"] + recording_offset
    )

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
            row_static["Stimulus_state"] == "static"
            and row_moving["Stimulus_state"] == "moving"
        ):

            trial_rows.append(
                {
                    "trial_id": trial_id,

                    # Core stimulus identity
                    "direction": row_moving["Direction_deg"],
                    "orientation": row_moving["Stimulus_orientation"],
                    "pattern": row_moving["Pattern"],

                    # Speed information
                    "speed": row_moving["Speed_label"],
                    "speed_label": row_moving["Speed_label"],
                    # Check stim_log about this
                    # "speed_deg_per_sec": row_moving["Speed_deg_per_sec"],
                    "speed_deg_per_sec": row_moving["Nominal_speed_deg_per_sec"],
                    "tf_hz": row_moving["GratingStim_TF_Hz"],
                    "sf_cpd": row_moving["GratingStim_SF_cpd"],

                    # Optional monitor metadata
                    "active_monitor_label": row_moving.get("Active_monitor_label", np.nan),
                    "active_screen_number": row_moving.get("Active_screen_number", np.nan),

                    # State timing in neural-recording time
                    "static_start_sec": row_static["Stimulus_start"],
                    "static_end_sec": row_static["Stimulus_end"],

                    "moving_start_sec": row_moving["Stimulus_start"],
                    "moving_end_sec": row_moving["Stimulus_end"],

                    # Matched Neuralynx TTL time
                    "ttl_time_sec": ttl_match.loc[trial_id, "event_time_sec"],
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