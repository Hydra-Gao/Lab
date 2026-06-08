from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    STIMLOG_RUNS,
    MOTION_STATE,
    EXPECTED_MOTION_TTL_COUNT,
)


TTL_GLOBAL_PATH = ANALYSIS_OUTPUT_DIR / "ttl_events_global.csv"

ALIGNMENT_RMS_WARNING_MS = 2.0
ALIGNMENT_MAX_ABS_WARNING_MS = 5.0


def normalize_state(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def get_run_value(run_cfg, key, default=None):
    if key in run_cfg:
        return run_cfg[key]
    return default


def get_ttl_time_col(ttl_df):
    if "concat_time_sec" in ttl_df.columns:
        return "concat_time_sec"
    if "event_time_sec" in ttl_df.columns:
        return "event_time_sec"
    raise ValueError("TTL file must contain either concat_time_sec or event_time_sec.")


def choose_ttl_block_for_run(motion_rows, ttl_segment, manual_start_index=None):
    """
    Match one stimlog's motion rows to a consecutive TTL block within one segment.

    If manual_start_index is None:
        try all possible blocks inside this segment and choose lowest RMS residual.
    If manual_start_index is given:
        use that local start index within ttl_segment.
    """
    n_motion = len(motion_rows)
    n_ttl = len(ttl_segment)

    if n_ttl < n_motion:
        raise ValueError(
            f"Not enough TTLs in this segment. TTL rows={n_ttl}, motion rows={n_motion}."
        )

    ttl_time_col = get_ttl_time_col(ttl_segment)
    stim_starts = motion_rows["Stimulus_start"].to_numpy(dtype=float)

    if manual_start_index is not None:
        start = int(manual_start_index)

        if start < 0 or start + n_motion > n_ttl:
            raise ValueError(
                f"Manual ttl_block_start_index={start} is invalid. "
                f"Segment has {n_ttl} TTLs and stimlog has {n_motion} motion rows."
            )

        ttl_match = ttl_segment.iloc[start:start + n_motion].copy()
        ttl_times = ttl_match[ttl_time_col].to_numpy(dtype=float)

        offset = ttl_times[0] - stim_starts[0]
        residual_ms = (ttl_times - (stim_starts + offset)) * 1000

        return {
            "mode": "manual",
            "ttl_time_col": ttl_time_col,
            "ttl_block_start_index": start,
            "ttl_match": ttl_match,
            "recording_offset_sec": offset,
            "residual_ms": residual_ms,
            "rms_ms": float(np.sqrt(np.mean(residual_ms ** 2))),
            "max_abs_ms": float(np.max(np.abs(residual_ms))),
        }

    candidates = []

    for start in range(0, n_ttl - n_motion + 1):
        ttl_block = ttl_segment.iloc[start:start + n_motion]
        ttl_times = ttl_block[ttl_time_col].to_numpy(dtype=float)

        offset = ttl_times[0] - stim_starts[0]
        residual_ms = (ttl_times - (stim_starts + offset)) * 1000

        candidates.append(
            {
                "ttl_block_start_index": start,
                "recording_offset_sec": offset,
                "rms_ms": float(np.sqrt(np.mean(residual_ms ** 2))),
                "max_abs_ms": float(np.max(np.abs(residual_ms))),
            }
        )

    candidates_df = pd.DataFrame(candidates)
    best = candidates_df.sort_values(["rms_ms", "max_abs_ms"]).iloc[0]

    start = int(best["ttl_block_start_index"])
    ttl_match = ttl_segment.iloc[start:start + n_motion].copy()
    ttl_times = ttl_match[ttl_time_col].to_numpy(dtype=float)

    offset = ttl_times[0] - stim_starts[0]
    residual_ms = (ttl_times - (stim_starts + offset)) * 1000

    return {
        "mode": "auto",
        "ttl_time_col": ttl_time_col,
        "ttl_block_start_index": start,
        "ttl_match": ttl_match,
        "recording_offset_sec": offset,
        "residual_ms": residual_ms,
        "rms_ms": float(np.sqrt(np.mean(residual_ms ** 2))),
        "max_abs_ms": float(np.max(np.abs(residual_ms))),
        "candidate_summary": candidates_df.sort_values("rms_ms").head(10),
    }


def build_trial_table_for_one_run(
    stimlog,
    ttl_match,
    ttl_time_col,
    recording_offset,
    run_cfg,
    run_index,
    global_trial_id_start,
):
    """
    Build trial rows for one single-screen run.

    Expected stimlog structure:
        static -> moving
    """
    original_segment_index = int(run_cfg["original_segment_index"])
    screen_role = str(run_cfg["screen_role"]).strip()
    stimlog_path = Path(run_cfg["stimlog_path"])
    stimlog_label = run_cfg.get("stimlog_label", stimlog_path.stem)

    updated_stimlog = stimlog.copy()

    updated_stimlog["Stimulus_start"] = (
        pd.to_numeric(updated_stimlog["Stimulus_start"], errors="coerce")
        + recording_offset
    )
    updated_stimlog["Stimulus_end"] = (
        pd.to_numeric(updated_stimlog["Stimulus_end"], errors="coerce")
        + recording_offset
    )

    updated_stimlog["run_index"] = run_index
    updated_stimlog["stimlog_file"] = str(stimlog_path)
    updated_stimlog["stimlog_label"] = stimlog_label
    updated_stimlog["screen_role"] = screen_role
    updated_stimlog["original_segment_index"] = original_segment_index
    updated_stimlog["recording_offset_sec"] = recording_offset

    trial_rows = []
    local_trial_id = 0

    for i in range(len(updated_stimlog) - 1):
        row_static = updated_stimlog.iloc[i]
        row_moving = updated_stimlog.iloc[i + 1]

        if (
            normalize_state(row_static["Stimulus_state"]) == "static"
            and normalize_state(row_moving["Stimulus_state"]) == "moving"
        ):
            if local_trial_id >= len(ttl_match):
                raise ValueError(
                    f"More detected static/moving trials than matched TTLs in run {stimlog_label}."
                )

            ttl_row = ttl_match.iloc[local_trial_id]

            trial_rows.append(
                {
                    "trial_id": global_trial_id_start + local_trial_id,
                    "local_trial_id": local_trial_id,

                    "run_index": run_index,
                    "stimlog_file": str(stimlog_path),
                    "stimlog_label": stimlog_label,

                    # Segment / TTL identity
                    "original_segment_index": original_segment_index,
                    "ttl_index_global": ttl_row.get("ttl_index_global", np.nan),
                    "ttl_timestamp_us": ttl_row.get("timestamp_us", np.nan),

                    # Screen identity from config
                    "screen_role": screen_role,

                    # Monitor metadata from stimlog if available
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

                    # State timing in sorter concat time
                    "static_start_sec": row_static["Stimulus_start"],
                    "static_end_sec": row_static["Stimulus_end"],
                    "moving_start_sec": row_moving["Stimulus_start"],
                    "moving_end_sec": row_moving["Stimulus_end"],

                    # Matched TTL in sorter concat time
                    "ttl_time_sec": ttl_row[ttl_time_col],
                    "ttl_concat_time_sec": ttl_row.get("concat_time_sec", ttl_row[ttl_time_col]),
                }
            )

            local_trial_id += 1

    return updated_stimlog, pd.DataFrame(trial_rows)


def process_one_run(run_cfg, run_index, ttl_global, global_trial_id_start):
    required = ["original_segment_index", "screen_role", "stimlog_path"]
    missing = [k for k in required if k not in run_cfg]
    if missing:
        raise ValueError(f"STIMLOG_RUNS[{run_index}] is missing keys: {missing}")

    original_segment_index = int(run_cfg["original_segment_index"])
    screen_role = str(run_cfg["screen_role"]).strip()
    stimlog_path = Path(run_cfg["stimlog_path"])
    stimlog_label = run_cfg.get("stimlog_label", stimlog_path.stem)
    manual_ttl_start = run_cfg.get("ttl_block_start_index", None)

    print("\n" + "=" * 80)
    print(f"Run {run_index}: {stimlog_label}")
    print(f"Stimlog: {stimlog_path}")
    print(f"Original segment index: {original_segment_index}")
    print(f"Screen role: {screen_role}")

    if not stimlog_path.exists():
        raise FileNotFoundError(f"Stimlog file not found: {stimlog_path}")

    stimlog = pd.read_csv(stimlog_path)

    motion_rows = (
        stimlog.loc[
            stimlog["Stimulus_state"].map(normalize_state)
            == normalize_state(MOTION_STATE)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Stimlog rows: {len(stimlog)}")
    print(f"Motion rows: {len(motion_rows)}")

    if len(motion_rows) == 0:
        raise ValueError(f"No motion rows found in {stimlog_path}")

    ttl_segment = (
        ttl_global.loc[
            ttl_global["original_segment_index"].astype(int)
            == original_segment_index
        ]
        .copy()
        .sort_values(get_ttl_time_col(ttl_global))
        .reset_index(drop=True)
    )

    print(f"TTL rows in selected segment: {len(ttl_segment)}")

    if len(ttl_segment) == 0:
        raise ValueError(
            f"No TTL events found for original_segment_index={original_segment_index}. "
            "Check SB01 output and STIMLOG_RUNS."
        )

    match = choose_ttl_block_for_run(
        motion_rows=motion_rows,
        ttl_segment=ttl_segment,
        manual_start_index=manual_ttl_start,
    )

    ttl_match = match["ttl_match"]
    ttl_time_col = match["ttl_time_col"]
    recording_offset = match["recording_offset_sec"]

    print("\nTTL block matching:")
    print(f"Mode: {match['mode']}")
    print(f"TTL local block start index: {match['ttl_block_start_index']}")
    print(f"Using TTL time column: {ttl_time_col}")
    print(f"Recording offset: {recording_offset:.6f} sec")
    print(f"Residual RMS: {match['rms_ms']:.3f} ms")
    print(f"Residual max abs: {match['max_abs_ms']:.3f} ms")

    if "candidate_summary" in match:
        print("\nTop TTL block candidates:")
        print(match["candidate_summary"])

    if match["rms_ms"] > ALIGNMENT_RMS_WARNING_MS:
        print(
            f"Warning: RMS residual {match['rms_ms']:.3f} ms > "
            f"{ALIGNMENT_RMS_WARNING_MS} ms."
        )

    if match["max_abs_ms"] > ALIGNMENT_MAX_ABS_WARNING_MS:
        print(
            f"Warning: max abs residual {match['max_abs_ms']:.3f} ms > "
            f"{ALIGNMENT_MAX_ABS_WARNING_MS} ms."
        )

    alignment_qc = pd.DataFrame(
        {
            "run_index": run_index,
            "stimlog_file": str(stimlog_path),
            "stimlog_label": stimlog_label,
            "screen_role": screen_role,
            "original_segment_index": original_segment_index,
            "local_trial_id": np.arange(len(motion_rows), dtype=int),
            "stimlog_motion_start_sec": motion_rows["Stimulus_start"].values,
            "ttl_time_sec": ttl_match[ttl_time_col].values,
        }
    )

    alignment_qc["recording_offset_sec"] = recording_offset
    alignment_qc["predicted_ttl_sec"] = (
        alignment_qc["stimlog_motion_start_sec"] + recording_offset
    )
    alignment_qc["residual_ms"] = match["residual_ms"]

    if "ttl_index_global" in ttl_match.columns:
        alignment_qc["ttl_index_global"] = ttl_match["ttl_index_global"].values

    if "timestamp_us" in ttl_match.columns:
        alignment_qc["ttl_timestamp_us"] = ttl_match["timestamp_us"].values

    updated_stimlog, trial_table = build_trial_table_for_one_run(
        stimlog=stimlog,
        ttl_match=ttl_match,
        ttl_time_col=ttl_time_col,
        recording_offset=recording_offset,
        run_cfg=run_cfg,
        run_index=run_index,
        global_trial_id_start=global_trial_id_start,
    )

    if len(trial_table) != len(motion_rows):
        print(
            f"Warning: detected trials ({len(trial_table)}) != motion rows ({len(motion_rows)}). "
            "Check whether stimlog contains incomplete static/moving pairs."
        )

    return updated_stimlog, alignment_qc, trial_table


def main():
    print("===== Building 8-direction single-screen trial table from multiple stimlogs =====")

    if len(STIMLOG_RUNS) == 0:
        raise ValueError("STIMLOG_RUNS is empty. Add runs in SB0_config_analysis.py.")

    if not TTL_GLOBAL_PATH.exists():
        raise FileNotFoundError(
            f"Global TTL file not found: {TTL_GLOBAL_PATH}\n"
            "Run updated SB01_extract_events.py first."
        )

    ttl_global = pd.read_csv(TTL_GLOBAL_PATH)

    if "original_segment_index" not in ttl_global.columns:
        raise ValueError(
            "ttl_events_global.csv must contain original_segment_index. "
            "Check updated SB01_extract_events.py."
        )

    ttl_time_col_global = get_ttl_time_col(ttl_global)

    ttl_global = (
        ttl_global
        .sort_values(ttl_time_col_global)
        .reset_index(drop=True)
    )

    print(f"TTL global rows: {len(ttl_global)}")
    print(f"Using global TTL time column: {ttl_time_col_global}")

    print("\nTTL counts by original_segment_index:")
    print(ttl_global.groupby("original_segment_index").size())

    updated_stimlogs = []
    alignment_qcs = []
    trial_tables = []

    global_trial_id_start = 0

    for run_index, run_cfg in enumerate(STIMLOG_RUNS):
        updated_stimlog, alignment_qc, trial_table = process_one_run(
            run_cfg=run_cfg,
            run_index=run_index,
            ttl_global=ttl_global,
            global_trial_id_start=global_trial_id_start,
        )

        updated_stimlogs.append(updated_stimlog)
        alignment_qcs.append(alignment_qc)
        trial_tables.append(trial_table)

        global_trial_id_start += len(trial_table)

    updated_all = pd.concat(updated_stimlogs, ignore_index=True)
    alignment_all = pd.concat(alignment_qcs, ignore_index=True)
    trial_all = pd.concat(trial_tables, ignore_index=True)

    # Optional global expected count check.
    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"\nExpected motion TTL count from config: {EXPECTED_MOTION_TTL_COUNT}")
        print(f"Total detected trials across STIMLOG_RUNS: {len(trial_all)}")
        if len(trial_all) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: total detected trials does not match EXPECTED_MOTION_TTL_COUNT. "
                "This may be fine if EXPECTED_MOTION_TTL_COUNT was set for one run only."
            )

    print("\nTrial counts by screen_role:")
    print(trial_all["screen_role"].value_counts(dropna=False))

    print("\nTrial counts by original_segment_index:")
    print(trial_all["original_segment_index"].value_counts(dropna=False).sort_index())

    print("\nAlignment residuals by run:")
    print(
        alignment_all
        .groupby(["screen_role", "original_segment_index"], dropna=False)["residual_ms"]
        .describe()
    )

    alignment_qc_path = ANALYSIS_OUTPUT_DIR / "alignment_qc.csv"
    updated_stimlog_path = ANALYSIS_OUTPUT_DIR / "updated_stimlog.csv"
    trial_table_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"

    alignment_all.to_csv(alignment_qc_path, index=False)
    updated_all.to_csv(updated_stimlog_path, index=False)
    trial_all.to_csv(trial_table_path, index=False)

    print("\n===== Saved =====")
    print(alignment_qc_path)
    print(updated_stimlog_path)
    print(trial_table_path)

    print("\nFirst few trials:")
    print(trial_all.head())


if __name__ == "__main__":
    main()