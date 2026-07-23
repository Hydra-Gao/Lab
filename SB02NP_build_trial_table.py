# SB02_build_trial_table.py

from __future__ import annotations

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    STIMLOG_PATH,
    MOTION_STATE,
    EXPECTED_MOTION_TTL_COUNT,
)


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def main() -> None:
    print("===== Building trial table =====")

    stimlog = pd.read_csv(STIMLOG_PATH)
    ttl_path = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    ttl_df = pd.read_csv(ttl_path)

    require_columns(
        stimlog,
        [
            "Trial_number_overall",
            "Stimulus_state",
            "Stimulus_start",
            "Stimulus_end",
            "Direction_deg",
            "Pattern",
            "Speed_deg_per_sec",
        ],
        "Stimlog",
    )
    require_columns(ttl_df, ["event_time_sec"], "TTL table")

    print(f"Stimlog rows: {len(stimlog)}")
    print(f"TTL rows: {len(ttl_df)}")

    # Remove session bookkeeping rows such as INCEPTION and SESSION_END.
    state_rows = stimlog.loc[
        stimlog["Stimulus_state"].isin(["static", MOTION_STATE])
    ].copy()

    motion_rows = (
        state_rows.loc[state_rows["Stimulus_state"] == MOTION_STATE]
        .sort_values("Stimulus_start")
        .reset_index(drop=True)
    )

    print(f"Motion rows: {len(motion_rows)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

    if len(ttl_df) != len(motion_rows):
        raise ValueError(
            "TTL count does not equal the number of moving rows: "
            f"TTL={len(ttl_df)}, moving={len(motion_rows)}. "
            "Do not align by row order until duplicate/missing events are resolved."
        )

    ttl_match = ttl_df.sort_values("event_time_sec").reset_index(drop=True)

    # A single offset maps PsychoPy session time onto SpikeGLX recording time.
    first_motion_onset = float(motion_rows.loc[0, "Stimulus_start"])
    first_ttl_time = float(ttl_match.loc[0, "event_time_sec"])
    recording_offset = first_ttl_time - first_motion_onset

    print(f"\nRecording offset: {recording_offset:.9f} sec")

    alignment_qc = pd.DataFrame(
        {
            "trial_id": np.arange(len(motion_rows), dtype=int),
            "trial_number_overall": motion_rows["Trial_number_overall"].values,
            "stimlog_motion_start_sec": motion_rows["Stimulus_start"].values,
            "ttl_time_sec": ttl_match["event_time_sec"].values,
        }
    )
    alignment_qc["predicted_ttl_sec"] = (
        alignment_qc["stimlog_motion_start_sec"] + recording_offset
    )
    alignment_qc["residual_ms"] = (
        alignment_qc["ttl_time_sec"] - alignment_qc["predicted_ttl_sec"]
    ) * 1000.0

    # Diagnostic only: slope near 1 indicates negligible relative clock drift.
    slope, intercept = np.polyfit(
        alignment_qc["stimlog_motion_start_sec"].to_numpy(dtype=float),
        alignment_qc["ttl_time_sec"].to_numpy(dtype=float),
        deg=1,
    )
    alignment_qc["linear_fit_ttl_sec"] = (
        intercept + slope * alignment_qc["stimlog_motion_start_sec"]
    )
    alignment_qc["linear_fit_residual_ms"] = (
        alignment_qc["ttl_time_sec"] - alignment_qc["linear_fit_ttl_sec"]
    ) * 1000.0

    print("\nFixed-offset residuals (ms):")
    print(alignment_qc["residual_ms"].describe())
    print(f"Clock-fit slope: {slope:.10f}")
    print(f"Clock-fit intercept: {intercept:.9f} sec")

    updated_stimlog = stimlog.copy()
    valid_time = updated_stimlog["Stimulus_start"].notna()
    updated_stimlog.loc[valid_time, "Stimulus_start"] += recording_offset

    valid_time = updated_stimlog["Stimulus_end"].notna()
    updated_stimlog.loc[valid_time, "Stimulus_end"] += recording_offset

    # This TG963 log contains static -> moving, with no blank row.
    # Pair states by Trial_number_overall rather than assuming fixed row triplets.
    trial_rows = []

    grouped = state_rows.groupby("Trial_number_overall", sort=False, dropna=True)
    for trial_number, trial_df in grouped:
        static_df = trial_df.loc[trial_df["Stimulus_state"] == "static"]
        moving_df = trial_df.loc[trial_df["Stimulus_state"] == MOTION_STATE]

        if len(static_df) != 1 or len(moving_df) != 1:
            raise ValueError(
                f"Trial {trial_number} does not contain exactly one static and "
                f"one moving row: static={len(static_df)}, moving={len(moving_df)}"
            )

        static_row = static_df.iloc[0]
        moving_row = moving_df.iloc[0]

        trial_rows.append(
            {
                "trial_number_overall": int(trial_number),
                "replicate": moving_row.get("Replicate", np.nan),
                "condition_order": moving_row.get("Condition_order", np.nan),
                "condition_name": moving_row.get("Condition_name", np.nan),
                "trial_kind": moving_row.get("Trial_kind", np.nan),
                "trial_within_condition": moving_row.get(
                    "Trial_within_condition", np.nan
                ),
                "active_screen_role": moving_row.get("Active_screen_role", np.nan),
                "direction": moving_row.get("Direction_deg", np.nan),
                "orientation": moving_row.get(
                    "Stimulus_orientation_single_screen", np.nan
                ),
                "pattern": moving_row.get("Pattern", np.nan),
                "biological_label": moving_row.get("Biological_label", np.nan),
                "speed": moving_row.get("Speed_deg_per_sec", np.nan),
                "speed_label": moving_row.get("Speed_label", np.nan),
                "recording_site_side": moving_row.get("Recording_site_side", np.nan),
                "ipsilateral_screen_role": moving_row.get(
                    "Ipsilateral_screen_role", np.nan
                ),
                "contralateral_screen_role": moving_row.get(
                    "Contralateral_screen_role", np.nan
                ),
                "left_movement": moving_row.get("Left_movement", np.nan),
                "front_movement": moving_row.get("Front_movement", np.nan),
                "right_movement": moving_row.get("Right_movement", np.nan),
                # No blank state exists in this behavior log.
                "blank_start_sec": np.nan,
                "blank_end_sec": np.nan,
                "static_start_sec": float(static_row["Stimulus_start"] + recording_offset),
                "static_end_sec": float(static_row["Stimulus_end"] + recording_offset),
                "moving_start_sec": float(moving_row["Stimulus_start"] + recording_offset),
                "moving_end_sec": float(moving_row["Stimulus_end"] + recording_offset),
            }
        )

    trial_table = pd.DataFrame(trial_rows).sort_values(
        "trial_number_overall"
    ).reset_index(drop=True)
    trial_table.insert(0, "trial_id", np.arange(len(trial_table), dtype=int))

    # Because trial_table is sorted in the same chronological order as motion_rows,
    # attach the corresponding extracted TTL and residual diagnostics directly.
    trial_table["ttl_time_sec"] = ttl_match["event_time_sec"].to_numpy()
    trial_table["alignment_residual_ms"] = alignment_qc["residual_ms"].to_numpy()

    print(f"\nDetected trials: {len(trial_table)}")
    print("Trial kinds:")
    print(trial_table["trial_kind"].value_counts(dropna=False))

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
