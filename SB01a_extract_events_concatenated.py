# SB01_extract_events.py

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    RAW_NLX_FOLDER,
    EVENTS_NEV_PATH,
    ANALYSIS_OUTPUT_DIR,
    EXPECTED_MOTION_TTL_COUNT,
)

from config_local import WORKING_DIR


# =====================
# User / pipeline settings
# =====================

# This file is created by SS2_Sorting.py / checked by SS4_Export_phy.py
SEGMENT_TIME_MAP_PATH = WORKING_DIR / "segment_time_map_M12.csv"

# Output files
TTL_GLOBAL_OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "ttl_events_global.csv"
SEGMENT_ABS_MAP_OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "segment_absolute_time_map.csv"

# Neuralynx constants
NLX_HEADER_SIZE = 16 * 1024
NEV_RECORD_SIZE = 184
NCS_RECORD_SIZE = 1044
NCS_SAMPLES_PER_RECORD = 512

# Neuralynx NEV record format
NEV_STRUCT = struct.Struct("<hhhQhhhhh8i128s")


# =====================
# Readers
# =====================

def read_nev(nev_path: Path) -> pd.DataFrame:
    """Read Neuralynx Events.nev into a DataFrame."""
    rows = []

    with open(nev_path, "rb") as f:
        f.seek(NLX_HEADER_SIZE)

        record_index = 0

        while True:
            chunk = f.read(NEV_RECORD_SIZE)

            if not chunk:
                break

            if len(chunk) != NEV_RECORD_SIZE:
                print(f"Warning: incomplete NEV record at index {record_index}, skipped.")
                break

            data = NEV_STRUCT.unpack(chunk)

            timestamp_us = data[3]
            event_id = data[4]
            ttl_value = data[5]
            event_string_raw = data[-1]

            event_string = (
                event_string_raw
                .split(b"\x00", 1)[0]
                .decode("latin-1", errors="replace")
                .strip()
            )

            rows.append(
                {
                    "record_index": record_index,
                    "timestamp_us": int(timestamp_us),
                    "event_id": int(event_id),
                    "ttl_value": int(ttl_value),
                    "event_string": event_string,
                }
            )

            record_index += 1

    if len(rows) == 0:
        raise ValueError(f"No events found in {nev_path}")

    return (
        pd.DataFrame(rows)
        .sort_values("timestamp_us")
        .reset_index(drop=True)
    )


def get_first_ncs_file(raw_folder: Path) -> Path:
    """Pick one NCS file to infer segment boundaries."""
    ncs_files = (
        list(raw_folder.glob("*.ncs"))
        + list(raw_folder.glob("*.Ncs"))
        + list(raw_folder.glob("*.NCS"))
    )

    if len(ncs_files) == 0:
        raise FileNotFoundError(f"No .ncs files found in {raw_folder}")

    return sorted(ncs_files)[0]


def read_ncs_timestamps(ncs_path: Path) -> np.ndarray:
    """
    Read NCS record start timestamps.

    Each NCS record:
        timestamp uint64
        channel int32
        sample_freq int32
        num_valid_samples int32
        512 int16 samples
    """
    timestamps = []

    with open(ncs_path, "rb") as f:
        f.seek(NLX_HEADER_SIZE)

        while True:
            ts_bytes = f.read(8)

            if not ts_bytes:
                break

            if len(ts_bytes) != 8:
                break

            timestamps.append(struct.unpack("<Q", ts_bytes)[0])

            # Skip rest of record
            f.seek(NCS_RECORD_SIZE - 8, 1)

    if len(timestamps) == 0:
        raise ValueError(f"No timestamps read from {ncs_path}")

    return np.asarray(timestamps, dtype=np.int64)


# =====================
# Segment mapping
# =====================

def infer_absolute_segments_from_ncs(raw_folder: Path) -> pd.DataFrame:
    """
    Infer original Neuralynx segment absolute timestamp boundaries from NCS gaps.

    Output uses Neuralynx absolute timestamps in microseconds.
    """
    ncs_path = get_first_ncs_file(raw_folder)
    timestamps = read_ncs_timestamps(ncs_path)

    diffs = np.diff(timestamps)

    normal_step_us = float(np.median(diffs))

    # Usually 512 samples at 32 kHz = 16000 us.
    # A much larger gap indicates a new Neuralynx recording segment.
    gap_threshold_us = max(normal_step_us * 10, 1_000_000)

    break_after = np.where(diffs > gap_threshold_us)[0]

    starts = np.r_[0, break_after + 1]
    ends = np.r_[break_after, len(timestamps) - 1]

    rows = []

    for segment_index, start_i, end_i in zip(
        np.arange(len(starts), dtype=int),
        starts,
        ends,
    ):
        segment_start_us = int(timestamps[start_i])

        # timestamps[end_i] is the start time of the last NCS record.
        # Add one normal NCS step to approximate the true segment end.
        segment_end_us_exclusive = int(timestamps[end_i] + normal_step_us)

        rows.append(
            {
                "original_segment_index": int(segment_index),
                "segment_start_timestamp_us": segment_start_us,
                "segment_end_timestamp_us_exclusive": segment_end_us_exclusive,
                "n_ncs_records": int(end_i - start_i + 1),
                "normal_step_us": normal_step_us,
                "duration_sec_from_timestamps": (
                    segment_end_us_exclusive - segment_start_us
                ) / 1_000_000.0,
            }
        )

    abs_segments = pd.DataFrame(rows)

    print(f"Using NCS file to infer absolute segments: {ncs_path.name}")
    print("\nInferred absolute Neuralynx segments:")
    print(abs_segments)

    return abs_segments


def load_concat_segment_time_map(path: Path) -> pd.DataFrame:
    """
    Load segment_time_map_M12.csv created by SS2.

    Required columns from SS2:
        original_segment_index
        concat_start_sec
        concat_end_sec
        n_samples
        duration_sec
    """
    if not path.exists():
        raise FileNotFoundError(
            f"segment_time_map file not found:\n{path}\n\n"
            "Run SS2_Sorting.py first after adding SEGMENTS_TO_SORT and "
            "save_segment_time_map()."
        )

    segment_map = pd.read_csv(path)

    required_cols = [
        "original_segment_index",
        "concat_start_sec",
        "concat_end_sec",
        "n_samples",
        "duration_sec",
    ]

    missing = [c for c in required_cols if c not in segment_map.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    segment_map["original_segment_index"] = (
        segment_map["original_segment_index"].astype(int)
    )

    print("\nLoaded concat segment time map:")
    print(segment_map)

    return segment_map


def build_full_segment_map(
    abs_segments: pd.DataFrame,
    concat_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge absolute Neuralynx segment boundaries with sorter concat time map.

    This is the bridge:

        Neuralynx timestamp_us
            -> time_within_original_segment_sec
            -> concat_time_sec
    """
    full_map = concat_map.merge(
        abs_segments,
        on="original_segment_index",
        how="left",
        validate="one_to_one",
    )

    if full_map["segment_start_timestamp_us"].isna().any():
        bad = full_map.loc[
            full_map["segment_start_timestamp_us"].isna(),
            "original_segment_index",
        ].tolist()

        raise ValueError(
            "Some selected segments from segment_time_map_M12.csv were not found "
            f"in the Neuralynx absolute segment map: {bad}"
        )

    # Add a small consistency check between preprocessed duration and NCS timestamp duration.
    full_map["duration_diff_sec"] = (
        full_map["duration_sec"]
        - full_map["duration_sec_from_timestamps"]
    )

    print("\nFull segment map:")
    print(full_map)

    return full_map


# =====================
# TTL conversion
# =====================


def deduplicate_ttl_events(ttl: pd.DataFrame, refractory_sec: float = 0.01) -> pd.DataFrame:
    """
    Collapse duplicate TTL events that belong to the same pulse.

    Neuralynx may record one serial/TTL write as multiple positive events
    separated by sub-millisecond intervals. Keep only the first event in each
    refractory window.
    """
    ttl = ttl.sort_values("timestamp_us").reset_index(drop=True).copy()

    dt_sec = ttl["timestamp_us"].diff() / 1_000_000.0

    keep = dt_sec.isna() | (dt_sec > refractory_sec)

    ttl_dedup = ttl.loc[keep].copy().reset_index(drop=True)

    print("\nTTL deduplication:")
    print(f"Before dedup: {len(ttl)}")
    print(f"After dedup:  {len(ttl_dedup)}")
    print(f"Removed:      {len(ttl) - len(ttl_dedup)}")
    print(f"Refractory:   {refractory_sec * 1000:.1f} ms")

    return ttl_dedup


def assign_events_to_concat_time(
    events: pd.DataFrame,
    full_segment_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each TTL event to one selected original segment and convert to concat time.
    """
    ttl = events.loc[events["ttl_value"] > 0].copy()
    ttl = deduplicate_ttl_events(ttl, refractory_sec=0.01)      

    assigned_parts = []

    for _, seg in full_segment_map.iterrows():
        original_segment_index = int(seg["original_segment_index"])
        seg_start_us = int(seg["segment_start_timestamp_us"])
        seg_end_us_exclusive = int(seg["segment_end_timestamp_us_exclusive"])
        concat_start_sec = float(seg["concat_start_sec"])

        mask = (
            (ttl["timestamp_us"] >= seg_start_us)
            & (ttl["timestamp_us"] < seg_end_us_exclusive)
        )

        ttl_seg = ttl.loc[mask].copy()

        if ttl_seg.empty:
            continue

        ttl_seg["original_segment_index"] = original_segment_index

        ttl_seg["time_within_original_segment_sec"] = (
            ttl_seg["timestamp_us"] - seg_start_us
        ) / 1_000_000.0

        ttl_seg["concat_time_sec"] = (
            concat_start_sec
            + ttl_seg["time_within_original_segment_sec"]
        )

        ttl_seg["concat_start_sec"] = concat_start_sec

        assigned_parts.append(ttl_seg)

    if len(assigned_parts) == 0:
        return pd.DataFrame()

    ttl_global = pd.concat(assigned_parts, ignore_index=True)

    ttl_global = (
        ttl_global
        .sort_values("concat_time_sec")
        .reset_index(drop=True)
    )

    ttl_global["ttl_index_global"] = np.arange(len(ttl_global), dtype=int)

    keep_cols = [
        "ttl_index_global",
        "original_segment_index",
        "record_index",
        "timestamp_us",
        "time_within_original_segment_sec",
        "concat_start_sec",
        "concat_time_sec",
        "ttl_value",
        "event_id",
        "event_string",
    ]

    ttl_global = ttl_global[keep_cols]

    return ttl_global


# =====================
# Main
# =====================

def main() -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== SB01 Extract TTL events in sorter concat time =====")
    print(f"Events file: {EVENTS_NEV_PATH}")
    print(f"Raw folder:  {RAW_NLX_FOLDER}")
    print(f"Segment time map: {SEGMENT_TIME_MAP_PATH}")

    # 1. Read all NEV events
    events = read_nev(EVENTS_NEV_PATH)

    print(f"\nTotal events in NEV: {len(events)}")
    print(f"Total TTL/high events in NEV: {int((events['ttl_value'] > 0).sum())}")

    # 2. Infer original Neuralynx absolute segments
    abs_segments = infer_absolute_segments_from_ncs(RAW_NLX_FOLDER)

    # 3. Load SS2 concat map
    concat_map = load_concat_segment_time_map(SEGMENT_TIME_MAP_PATH)

    # 4. Merge absolute segment time + concat time
    full_segment_map = build_full_segment_map(
        abs_segments=abs_segments,
        concat_map=concat_map,
    )

    # Save this for debugging / later SB02 use
    full_segment_map.to_csv(SEGMENT_ABS_MAP_OUTPUT_PATH, index=False)

    print("\nSaved full segment absolute time map:")
    print(SEGMENT_ABS_MAP_OUTPUT_PATH)

    # 5. Assign TTL events to selected segments and convert to concat time
    ttl_global = assign_events_to_concat_time(
        events=events,
        full_segment_map=full_segment_map,
    )

    if ttl_global.empty:
        raise ValueError(
            "No TTL events were found inside the selected segments. "
            "Check SEGMENTS_TO_SORT in SS2 and the Neuralynx Events.nev file."
        )

    ttl_global.to_csv(TTL_GLOBAL_OUTPUT_PATH, index=False)

    print("\n===== Output =====")
    print(f"Saved TTL global events: {TTL_GLOBAL_OUTPUT_PATH}")
    print(f"TTL events inside selected concatenated segments: {len(ttl_global)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

        if len(ttl_global) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: TTL count does not match EXPECTED_MOTION_TTL_COUNT.\n"
                "This may be okay if the selected concatenated segments contain "
                "multiple stimulus runs, extra TTLs, or only part of the behavior file.\n"
                "SB02 should match the correct TTL block to each stimlog."
            )

    print("\nTTL events by original segment:")
    print(ttl_global.groupby("original_segment_index").size())

    print("\nFirst few TTL global events:")
    print(ttl_global.head())

    print("\nLast few TTL global events:")
    print(ttl_global.tail())


if __name__ == "__main__":
    main()