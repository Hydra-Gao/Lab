# SB02b_build_trial_table_vbc_3screen.py

from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    STIMLOG_PATH,
    MOTION_STATE,
    EXPECTED_MOTION_TTL_COUNT,
)


def require_columns(df, cols, name):
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def main():
    print("===== Building VbC three-screen trial table =====")

    # -----------------------------
    # Load files
    # -----------------------------

    stimlog = pd.read_csv(STIMLOG_PATH)

    ttl_path = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    ttl_df = pd.read_csv(ttl_path)

    print(f"Stimlog rows: {len(stimlog)}")
    print(f"TTL rows: {len(ttl_df)}")

    # -----------------------------
    # Required columns for this stimulus
    # -----------------------------

    required_stimlog_cols = [
        "Stimulus_state",
        "Stimulus_start",
        "Stimulus_end",

        "Trial_number_randomized",
        "Replicate",
        "Pattern",
        "Biological_label",
        "Left_movement",
        "Front_movement",
        "Right_movement",

        "GratingStim_TF_Hz",
        "GratingStim_SF_cpd",
        "GratingStim_phase_step",
        "Speed_deg_per_sec",
    ]

    required_ttl_cols = [
        "ttl_index",
        "event_time_sec",
    ]

    require_columns(stimlog, required_stimlog_cols, "stimlog")
    require_columns(ttl_df, required_ttl_cols, "TTL table")

    # -----------------------------
    # Remove non-stimulus rows
    # -----------------------------

    stimlog = stimlog[
        stimlog["Stimulus_state"].isin(["blank", "static", "moving"])
    ].copy()

    stimlog = stimlog.reset_index(drop=True)

    print("\nStimulus state counts:")
    print(stimlog["Stimulus_state"].value_counts())

    # -----------------------------
    # Keep moving rows only
    # -----------------------------
    # Each moving row should correspond to one TTL.
    # In this VbC three-screen stimulus:
    #   12 patterns × 6 replicates = 72 moving rows for one speed run.

    motion_rows = (
        stimlog.loc[stimlog["Stimulus_state"] == MOTION_STATE]
        .copy()
        .reset_index(drop=True)
    )

    print(f"\nMotion rows: {len(motion_rows)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

        if len(motion_rows) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: motion row count does not match "
                "EXPECTED_MOTION_TTL_COUNT."
            )

        if len(ttl_df) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: TTL count does not match "
                "EXPECTED_MOTION_TTL_COUNT."
            )

    if len(ttl_df) < len(motion_rows):
        raise ValueError(
            "Not enough TTL events for motion rows. "
            "Check RECORDING_SEGMENT_INDEX, stimlog path, or TTL extraction."
        )

    # -----------------------------
    # Match TTLs to moving rows
    # -----------------------------
    # First-pass rule:
    #   use the first N TTLs in this selected segment.
    #
    # This is OK if the segment contains only this stimulus run.
    # If a segment contains both slow and fast runs, later we should add:
    #   TTL_START_INDEX / TTL_END_INDEX
    # in config.

    ttl_match = ttl_df.iloc[: len(motion_rows)].copy().reset_index(drop=True)

    # -----------------------------
    # Compute recording offset
    # -----------------------------
    # stimlog time is relative to PsychoPy script start.
    # ttl event_time_sec is relative to Neuralynx selected segment start.
    #
    # recording_offset converts:
    #   stimlog time -> neural recording time

    first_motion_onset = float(motion_rows.loc[0, "Stimulus_start"])
    first_ttl_time = float(ttl_match.loc[0, "event_time_sec"])

    recording_offset = first_ttl_time - first_motion_onset

    print(f"\nRecording offset: {recording_offset:.6f} sec")

    # -----------------------------
    # Alignment QC
    # -----------------------------

    alignment_qc = pd.DataFrame(
        {
            "trial_id": np.arange(len(motion_rows), dtype=int),
            "trial_number_randomized": motion_rows[
                "Trial_number_randomized"
            ].values,
            "pattern": motion_rows["Pattern"].values,
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
    ) * 1000.0

    print("\nAlignment residuals in ms:")
    print(alignment_qc["residual_ms"].describe())

    # -----------------------------
    # Shift full stimlog into neural time
    # -----------------------------

    updated_stimlog = stimlog.copy()

    updated_stimlog["Stimulus_start"] = (
        updated_stimlog["Stimulus_start"].astype(float) + recording_offset
    )

    updated_stimlog["Stimulus_end"] = (
        updated_stimlog["Stimulus_end"].astype(float) + recording_offset
    )

    # -----------------------------
    # Build trial table
    # -----------------------------
    # The three-screen script logs:
    #   blank -> static -> moving
    #
    # blank duration is 0 in this script, but keeping it is still useful.
    # Response analysis should mainly use static as baseline.

    trial_rows = []

    trial_id = 0

    for i in range(len(updated_stimlog) - 2):

        row0 = updated_stimlog.iloc[i]
        row1 = updated_stimlog.iloc[i + 1]
        row2 = updated_stimlog.iloc[i + 2]

        is_trial = (
            row0["Stimulus_state"] == "blank"
            and row1["Stimulus_state"] == "static"
            and row2["Stimulus_state"] == "moving"
        )

        if not is_trial:
            continue

        # Safety check:
        # these three rows should belong to the same randomized trial.
        trial_numbers = {
            row0["Trial_number_randomized"],
            row1["Trial_number_randomized"],
            row2["Trial_number_randomized"],
        }

        if len(trial_numbers) != 1:
            raise ValueError(
                "blank/static/moving rows do not share the same "
                f"Trial_number_randomized near stimlog row {i}."
            )

        if trial_id >= len(ttl_match):
            raise ValueError(
                "More detected trials than matched TTLs. "
                "Check TTL matching."
            )

        trial_rows.append(
            {
                "trial_id": trial_id,

                "trial_number_randomized": int(row2["Trial_number_randomized"]),
                "replicate": int(row2["Replicate"]),

                # Main VbC condition labels
                "pattern": row2["Pattern"],
                "biological_label": row2["Biological_label"],

                # Three-screen movement labels
                "left_movement": row2["Left_movement"],
                "front_movement": row2["Front_movement"],
                "right_movement": row2["Right_movement"],

                # Speed / grating parameters
                "tf_hz": float(row2["GratingStim_TF_Hz"]),
                "sf_cpd": float(row2["GratingStim_SF_cpd"]),
                "phase_step": float(row2["GratingStim_phase_step"]),
                "speed_deg_per_sec": float(row2["Speed_deg_per_sec"]),

                # State timing in neural recording time
                "blank_start_sec": float(row0["Stimulus_start"]),
                "blank_end_sec": float(row0["Stimulus_end"]),

                "static_start_sec": float(row1["Stimulus_start"]),
                "static_end_sec": float(row1["Stimulus_end"]),

                "moving_start_sec": float(row2["Stimulus_start"]),
                "moving_end_sec": float(row2["Stimulus_end"]),

                # TTL timing in neural recording time
                "ttl_time_sec": float(ttl_match.loc[trial_id, "event_time_sec"]),
                "ttl_index": int(ttl_match.loc[trial_id, "ttl_index"]),
            }
        )

        trial_id += 1

    trial_table = pd.DataFrame(trial_rows)

    print(f"\nDetected trials: {len(trial_table)}")

    if len(trial_table) != len(motion_rows):
        print(
            "Warning: detected trial count does not match moving row count. "
            "This usually means the stimlog structure is not exactly "
            "blank -> static -> moving for every trial."
        )

    # -----------------------------
    # Extra QC
    # -----------------------------

    print("\nPattern counts:")
    print(
        trial_table
        .groupby(["speed_deg_per_sec", "pattern"])
        .size()
        .rename("n_trials")
        .reset_index()
        .sort_values(["speed_deg_per_sec", "pattern"])
    )

    print("\nTiming summary:")
    trial_table["static_duration_sec"] = (
        trial_table["static_end_sec"] - trial_table["static_start_sec"]
    )

    trial_table["moving_duration_sec"] = (
        trial_table["moving_end_sec"] - trial_table["moving_start_sec"]
    )

    print(
        trial_table[
            ["static_duration_sec", "moving_duration_sec"]
        ].describe()
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