# SB02_build_trial_table.py

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
# User settings
# =====================

# If you have only one stimlog, keep this as [STIMLOG_PATH].
# Later if you want to analyze multiple stimulus logs together, you can manually add:
# STIMLOG_PATHS = [
#     Path(r"...single_screen.csv"),
#     Path(r"...three_screen.csv"),
# ]
STIMLOG_PATHS = [STIMLOG_PATH]

# Optional labels. Must match STIMLOG_PATHS length.
# If None, file stems will be used.
STIMLOG_LABELS = None

# Optional manual TTL block start.
# Use None to auto-detect the best TTL block.
# Example:
#     TTL_BLOCK_START_INDICES = [0]
#     TTL_BLOCK_START_INDICES = [72, 168]
TTL_BLOCK_START_INDICES = [None]

# Alignment warning threshold.
# This does not stop the script; it only prints a warning.
ALIGNMENT_RMS_WARNING_MS = 20.0
ALIGNMENT_MAX_ABS_WARNING_MS = 50.0

# Input from SB01
TTL_GLOBAL_PATH = ANALYSIS_OUTPUT_DIR / "ttl_events_global.csv"

# Outputs
TRIAL_TABLE_PATH = ANALYSIS_OUTPUT_DIR / "trial_table.csv"
ALIGNMENT_QC_PATH = ANALYSIS_OUTPUT_DIR / "alignment_qc.csv"
UPDATED_STIMLOG_PATH = ANALYSIS_OUTPUT_DIR / "updated_stimlog_global.csv"


# =====================
# Helpers
# =====================

def get_stimlog_label(path: Path, index: int) -> str:
    if STIMLOG_LABELS is not None:
        return STIMLOG_LABELS[index]
    return path.stem


def normalize_stimulus_state(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def find_trial_number_column(stimlog: pd.DataFrame):
    candidates = [
        "Trial_number_randomized",
        "Trial_number",
        "trial_number",
        "trial_id",
    ]

    for c in candidates:
        if c in stimlog.columns:
            return c

    return None


def get_value(row, candidates, default=np.nan):
    for c in candidates:
        if c in row.index:
            return row[c]
    return default


def get_motion_rows(stimlog: pd.DataFrame) -> pd.DataFrame:
    if "Stimulus_state" not in stimlog.columns:
        raise ValueError("Stimlog must contain Stimulus_state column.")

    states = stimlog["Stimulus_state"].map(normalize_stimulus_state)
    motion_state = normalize_stimulus_state(MOTION_STATE)

    motion_rows = stimlog.loc[states == motion_state].copy()

    if motion_rows.empty:
        raise ValueError(
            f"No motion rows found using MOTION_STATE={MOTION_STATE}. "
            "Check SB0_config_analysis.py and stimlog Stimulus_state values."
        )

    return motion_rows.reset_index(drop=False).rename(columns={"index": "stimlog_row_index"})


def choose_best_ttl_block(
    motion_rows: pd.DataFrame,
    ttl_global: pd.DataFrame,
    manual_start_index=None,
):
    """
    Match stimlog motion rows to a consecutive TTL block.

    For each candidate TTL block:
        offset = first_ttl_concat_time - first_motion_stim_time
        predicted_ttl = stim_motion_start + offset
        residual = actual_ttl_concat_time - predicted_ttl

    Pick the candidate with the lowest RMS residual.
    """
    n_motion = len(motion_rows)
    n_ttl = len(ttl_global)

    if n_ttl < n_motion:
        raise ValueError(
            f"Not enough TTL events for motion rows. "
            f"TTL rows={n_ttl}, motion rows={n_motion}."
        )

    stim_starts = motion_rows["Stimulus_start"].to_numpy(dtype=float)

    if manual_start_index is not None:
        start = int(manual_start_index)

        if start < 0 or start + n_motion > n_ttl:
            raise ValueError(
                f"Manual TTL block start {start} is invalid for "
                f"{n_motion} motion rows and {n_ttl} TTLs."
            )

        ttl_block = ttl_global.iloc[start:start + n_motion].copy()
        ttl_times = ttl_block["concat_time_sec"].to_numpy(dtype=float)

        offset = ttl_times[0] - stim_starts[0]
        residual_ms = (ttl_times - (stim_starts + offset)) * 1000

        return {
            "start_index": start,
            "ttl_block": ttl_block,
            "recording_offset_sec": offset,
            "residual_ms": residual_ms,
            "rms_ms": float(np.sqrt(np.mean(residual_ms ** 2))),
            "max_abs_ms": float(np.max(np.abs(residual_ms))),
            "mode": "manual",
        }

    candidates = []

    for start in range(0, n_ttl - n_motion + 1):
        ttl_block = ttl_global.iloc[start:start + n_motion]
        ttl_times = ttl_block["concat_time_sec"].to_numpy(dtype=float)

        offset = ttl_times[0] - stim_starts[0]
        residual_ms = (ttl_times - (stim_starts + offset)) * 1000

        rms_ms = float(np.sqrt(np.mean(residual_ms ** 2)))
        max_abs_ms = float(np.max(np.abs(residual_ms)))

        candidates.append(
            {
                "start_index": start,
                "rms_ms": rms_ms,
                "max_abs_ms": max_abs_ms,
                "recording_offset_sec": offset,
            }
        )

    candidates_df = pd.DataFrame(candidates)

    best = candidates_df.sort_values(
        ["rms_ms", "max_abs_ms"],
        ascending=[True, True],
    ).iloc[0]

    best_start = int(best["start_index"])
    ttl_block = ttl_global.iloc[best_start:best_start + n_motion].copy()
    ttl_times = ttl_block["concat_time_sec"].to_numpy(dtype=float)

    offset = ttl_times[0] - stim_starts[0]
    residual_ms = (ttl_times - (stim_starts + offset)) * 1000

    return {
        "start_index": best_start,
        "ttl_block": ttl_block,
        "recording_offset_sec": offset,
        "residual_ms": residual_ms,
        "rms_ms": float(np.sqrt(np.mean(residual_ms ** 2))),
        "max_abs_ms": float(np.max(np.abs(residual_ms))),
        "mode": "auto",
        "candidate_summary": candidates_df.sort_values("rms_ms").head(10),
    }


def add_concat_times_to_stimlog(stimlog: pd.DataFrame, offset_sec: float) -> pd.DataFrame:
    updated = stimlog.copy()

    if "Stimulus_start" in updated.columns:
        updated["Stimulus_start_concat_sec"] = (
            pd.to_numeric(updated["Stimulus_start"], errors="coerce") + offset_sec
        )

    if "Stimulus_end" in updated.columns:
        updated["Stimulus_end_concat_sec"] = (
            pd.to_numeric(updated["Stimulus_end"], errors="coerce") + offset_sec
        )

    return updated


def build_trials_from_stimlog(
    updated_stimlog: pd.DataFrame,
    motion_rows: pd.DataFrame,
    ttl_block: pd.DataFrame,
    stimlog_path: Path,
    stimlog_label: str,
    global_trial_id_start: int,
):
    """
    Build trial table using concat-time columns.

    Supports:
    - blank -> static -> moving
    - static -> moving
    - motion-only, if no static row is found
    """
    trial_num_col = find_trial_number_column(updated_stimlog)

    trial_rows = []

    for local_trial_id, (_, motion_row_old) in enumerate(motion_rows.iterrows()):
        stimlog_row_index = int(motion_row_old["stimlog_row_index"])

        motion_row = updated_stimlog.loc[stimlog_row_index]

        ttl_row = ttl_block.iloc[local_trial_id]

        if trial_num_col is not None:
            trial_key = motion_row[trial_num_col]

            same_trial = updated_stimlog.loc[
                updated_stimlog[trial_num_col] == trial_key
            ].copy()
        else:
            # Fallback: use nearby rows before motion row
            trial_key = local_trial_id
            start_i = max(0, stimlog_row_index - 2)
            same_trial = updated_stimlog.iloc[start_i:stimlog_row_index + 1].copy()

        same_trial_states = same_trial["Stimulus_state"].map(normalize_stimulus_state)

        blank_rows = same_trial.loc[same_trial_states == "blank"]
        static_rows = same_trial.loc[same_trial_states == "static"]

        if not blank_rows.empty:
            blank_row = blank_rows.iloc[-1]
            blank_start = blank_row.get("Stimulus_start_concat_sec", np.nan)
            blank_end = blank_row.get("Stimulus_end_concat_sec", np.nan)
        else:
            blank_start = np.nan
            blank_end = np.nan

        if not static_rows.empty:
            static_row = static_rows.iloc[-1]
            static_start = static_row.get("Stimulus_start_concat_sec", np.nan)
            static_end = static_row.get("Stimulus_end_concat_sec", np.nan)
        else:
            static_start = np.nan
            static_end = np.nan

        moving_start = motion_row.get("Stimulus_start_concat_sec", np.nan)
        moving_end = motion_row.get("Stimulus_end_concat_sec", np.nan)

        row = {
            "trial_id": global_trial_id_start + local_trial_id,
            "local_trial_id": local_trial_id,
            "stimlog_file": str(stimlog_path),
            "stimlog_label": stimlog_label,
            "stimlog_row_index": stimlog_row_index,
            "stimlog_trial_key": trial_key,

            "original_segment_index": ttl_row.get("original_segment_index", np.nan),
            "ttl_index_global": ttl_row.get("ttl_index_global", np.nan),
            "ttl_timestamp_us": ttl_row.get("timestamp_us", np.nan),
            "ttl_concat_time_sec": ttl_row.get("concat_time_sec", np.nan),

            "blank_start_sec": blank_start,
            "blank_end_sec": blank_end,
            "static_start_sec": static_start,
            "static_end_sec": static_end,
            "moving_start_sec": moving_start,
            "moving_end_sec": moving_end,

            # Generic stimulus descriptors
            "direction": get_value(
                motion_row,
                ["Direction", "Direction_deg", "direction"],
            ),
            "orientation": get_value(
                motion_row,
                ["Stimulus_orientation", "Orientation", "orientation"],
            ),
            "pattern": get_value(
                motion_row,
                ["Pattern", "pattern"],
            ),
            "speed": get_value(
                motion_row,
                ["Speed", "Speed_label", "speed"],
            ),
            "speed_deg_per_sec": get_value(
                motion_row,
                ["Speed_deg_per_sec", "speed_deg_per_sec"],
            ),

            # Useful extra fields for your three-screen VbC stimulus
            "biological_label": get_value(
                motion_row,
                ["Biological_label", "biological_label"],
            ),
            "left_movement": get_value(
                motion_row,
                ["Left_movement"],
            ),
            "front_movement": get_value(
                motion_row,
                ["Front_movement"],
            ),
            "right_movement": get_value(
                motion_row,
                ["Right_movement"],
            ),

            "grating_tf_hz": get_value(
                motion_row,
                ["GratingStim_TF_Hz"],
            ),
            "grating_sf_cpd": get_value(
                motion_row,
                ["GratingStim_SF_cpd"],
            ),
            "grating_phase_step": get_value(
                motion_row,
                ["GratingStim_phase_step"],
            ),
        }

        trial_rows.append(row)

    return pd.DataFrame(trial_rows)


def process_one_stimlog(
    stimlog_path: Path,
    stimlog_label: str,
    ttl_global: pd.DataFrame,
    manual_ttl_start,
    global_trial_id_start: int,
):
    print("\n" + "=" * 80)
    print(f"Processing stimlog: {stimlog_path}")
    print(f"Stimlog label: {stimlog_label}")

    stimlog = pd.read_csv(stimlog_path)

    print(f"Stimlog rows: {len(stimlog)}")

    motion_rows = get_motion_rows(stimlog)

    print(f"Motion rows: {len(motion_rows)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

    match = choose_best_ttl_block(
        motion_rows=motion_rows,
        ttl_global=ttl_global,
        manual_start_index=manual_ttl_start,
    )

    ttl_block = match["ttl_block"]
    offset_sec = match["recording_offset_sec"]
    residual_ms = match["residual_ms"]

    print("\nTTL block matching:")
    print(f"Mode: {match['mode']}")
    print(f"TTL block start index: {match['start_index']}")
    print(f"Recording offset: {offset_sec:.6f} sec")
    print(f"Residual RMS: {match['rms_ms']:.3f} ms")
    print(f"Residual max abs: {match['max_abs_ms']:.3f} ms")

    if "candidate_summary" in match:
        print("\nTop candidate TTL blocks:")
        print(match["candidate_summary"])

    if match["rms_ms"] > ALIGNMENT_RMS_WARNING_MS:
        print(
            f"Warning: alignment RMS {match['rms_ms']:.3f} ms is larger than "
            f"{ALIGNMENT_RMS_WARNING_MS} ms."
        )

    if match["max_abs_ms"] > ALIGNMENT_MAX_ABS_WARNING_MS:
        print(
            f"Warning: alignment max abs residual {match['max_abs_ms']:.3f} ms "
            f"is larger than {ALIGNMENT_MAX_ABS_WARNING_MS} ms."
        )

    updated_stimlog = add_concat_times_to_stimlog(
        stimlog=stimlog,
        offset_sec=offset_sec,
    )

    updated_stimlog["stimlog_file"] = str(stimlog_path)
    updated_stimlog["stimlog_label"] = stimlog_label
    updated_stimlog["recording_offset_sec"] = offset_sec
    updated_stimlog["ttl_block_start_index"] = match["start_index"]

    alignment_qc = pd.DataFrame(
        {
            "stimlog_file": str(stimlog_path),
            "stimlog_label": stimlog_label,
            "local_trial_id": np.arange(len(motion_rows), dtype=int),
            "stimlog_row_index": motion_rows["stimlog_row_index"].values,
            "stimlog_motion_start_sec": motion_rows["Stimulus_start"].values,
            "ttl_index_global": ttl_block["ttl_index_global"].values,
            "ttl_concat_time_sec": ttl_block["concat_time_sec"].values,
        }
    )

    alignment_qc["recording_offset_sec"] = offset_sec
    alignment_qc["predicted_ttl_concat_sec"] = (
        alignment_qc["stimlog_motion_start_sec"] + offset_sec
    )
    alignment_qc["residual_ms"] = residual_ms

    trial_table = build_trials_from_stimlog(
        updated_stimlog=updated_stimlog,
        motion_rows=motion_rows,
        ttl_block=ttl_block,
        stimlog_path=stimlog_path,
        stimlog_label=stimlog_label,
        global_trial_id_start=global_trial_id_start,
    )

    return updated_stimlog, alignment_qc, trial_table


def main():
    print("===== SB02 Build global trial table using TTL concat time =====")

    if not TTL_GLOBAL_PATH.exists():
        raise FileNotFoundError(
            f"TTL global file not found:\n{TTL_GLOBAL_PATH}\n\n"
            "Run SB01_extract_events.py first."
        )

    ttl_global = pd.read_csv(TTL_GLOBAL_PATH)

    if "concat_time_sec" not in ttl_global.columns:
        raise ValueError(
            "ttl_events_global.csv must contain concat_time_sec. "
            "Check SB01_extract_events.py."
        )

    ttl_global = (
        ttl_global
        .sort_values("concat_time_sec")
        .reset_index(drop=True)
    )

    print(f"TTL global rows: {len(ttl_global)}")
    print("TTL by original segment:")
    print(ttl_global.groupby("original_segment_index").size())

    if TTL_BLOCK_START_INDICES is None:
        ttl_starts = [None] * len(STIMLOG_PATHS)
    else:
        ttl_starts = TTL_BLOCK_START_INDICES

    if len(ttl_starts) != len(STIMLOG_PATHS):
        raise ValueError(
            "TTL_BLOCK_START_INDICES must have the same length as STIMLOG_PATHS."
        )

    updated_stimlogs = []
    alignment_qcs = []
    trial_tables = []

    global_trial_id_start = 0

    for i, stimlog_path in enumerate(STIMLOG_PATHS):
        stimlog_path = Path(stimlog_path)
        stimlog_label = get_stimlog_label(stimlog_path, i)
        manual_start = ttl_starts[i]

        updated_stimlog, alignment_qc, trial_table = process_one_stimlog(
            stimlog_path=stimlog_path,
            stimlog_label=stimlog_label,
            ttl_global=ttl_global,
            manual_ttl_start=manual_start,
            global_trial_id_start=global_trial_id_start,
        )

        updated_stimlogs.append(updated_stimlog)
        alignment_qcs.append(alignment_qc)
        trial_tables.append(trial_table)

        global_trial_id_start += len(trial_table)

    updated_all = pd.concat(updated_stimlogs, ignore_index=True)
    alignment_all = pd.concat(alignment_qcs, ignore_index=True)
    trial_all = pd.concat(trial_tables, ignore_index=True)

    updated_all.to_csv(UPDATED_STIMLOG_PATH, index=False)
    alignment_all.to_csv(ALIGNMENT_QC_PATH, index=False)
    trial_all.to_csv(TRIAL_TABLE_PATH, index=False)

    print("\n===== Saved =====")
    print(f"Updated stimlog: {UPDATED_STIMLOG_PATH}")
    print(f"Alignment QC:    {ALIGNMENT_QC_PATH}")
    print(f"Trial table:     {TRIAL_TABLE_PATH}")

    print("\nAlignment residuals by stimlog:")
    print(
        alignment_all
        .groupby("stimlog_label")["residual_ms"]
        .describe()
    )

    print("\nDetected trials:")
    print(len(trial_all))

    print("\nFirst few trials:")
    print(trial_all.head())

    print("\nTrial counts by stimlog:")
    print(trial_all.groupby("stimlog_label").size())

    if "original_segment_index" in trial_all.columns:
        print("\nTrial counts by original segment:")
        print(trial_all.groupby("original_segment_index").size())


if __name__ == "__main__":
    main()