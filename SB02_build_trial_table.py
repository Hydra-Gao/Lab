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
    # Each trial:
    # blank -> static -> moving

    trial_rows = []

    trial_id = 0

    for i in range(len(updated_stimlog) - 2):

        row0 = updated_stimlog.iloc[i]
        row1 = updated_stimlog.iloc[i + 1]
        row2 = updated_stimlog.iloc[i + 2]

        if (
            row0["Stimulus_state"] == "blank"
            and row1["Stimulus_state"] == "static"
            and row2["Stimulus_state"] == "moving"
        ):

            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "direction": row2["Direction"],
                    "orientation": row2["Stimulus_orientation"],
                    "pattern": row2["Pattern"],
                    "speed": row2["Speed"],
                    "left_speed_factor": row2.get("Left_speed_factor", np.nan),
                    "right_speed_factor": row2.get("Right_speed_factor", np.nan),

                    "blank_start_sec": row0["Stimulus_start"],
                    "blank_end_sec": row0["Stimulus_end"],

                    "static_start_sec": row1["Stimulus_start"],
                    "static_end_sec": row1["Stimulus_end"],

                    "moving_start_sec": row2["Stimulus_start"],
                    "moving_end_sec": row2["Stimulus_end"],

                    "ttl_time_sec": ttl_match.loc[trial_id, "event_time_sec"],
                }
            )

            trial_id += 1

    trial_table = pd.DataFrame(trial_rows)

    print(f"\nDetected trials: {len(trial_table)}")

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