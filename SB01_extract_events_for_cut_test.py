# SB01_extract_events_dedup_12patterns.py
# Minimal-change replacement for SB01_extract_events.py when one stimulus onset
# creates multiple very-close TTL rising events.
#
# Output names are intentionally kept the same for downstream scripts:
#   events_ttl_rising_segment.csv
# so SB02b and later scripts can stay almost unchanged.

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    RAW_NLX_FOLDER,
    EVENTS_NEV_PATH,
    ANALYSIS_OUTPUT_DIR,
    RECORDING_SEGMENT_INDEX,
    SEGMENT_START_TIMESTAMP_US,
    SEGMENT_END_TIMESTAMP_US,
    EXPECTED_MOTION_TTL_COUNT,
    ORIGINAL_RECORDING_SEGMENT_INDEX,
    TEST_START_SEC_WITHIN_ORIGINAL_SEGMENT,
    TEST_END_SEC_WITHIN_ORIGINAL_SEGMENT,
)


NLX_HEADER_SIZE = 16 * 1024
NEV_RECORD_SIZE = 184
NEV_STRUCT = struct.Struct("<hhhQhhhhh8i128s")

# TTLs within this interval are treated as duplicate pulses from the same moving onset.
# Your uploaded CSV has triplet intervals around 0.0002-0.0005 s and real inter-trial
# intervals around 8-9 s, so 0.5 s is safely between those scales.
TTL_DEBOUNCE_WINDOW_SEC = 0.5


def read_nev(nev_path: Path) -> pd.DataFrame:
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
            event_string = (
                data[-1]
                .split(b"\x00", 1)[0]
                .decode("latin-1", errors="replace")
                .strip()
            )
            rows.append(
                {
                    "record_index": record_index,
                    "timestamp_us": int(data[3]),
                    "event_id": int(data[4]),
                    "ttl_value": int(data[5]),
                    "event_string": event_string,
                }
            )
            record_index += 1
    if not rows:
        raise ValueError(f"No events found in {nev_path}")
    return pd.DataFrame(rows).sort_values("timestamp_us").reset_index(drop=True)


def get_first_ncs_file(raw_folder: Path) -> Path:
    ncs_files = (
        list(raw_folder.glob("*.ncs"))
        + list(raw_folder.glob("*.Ncs"))
        + list(raw_folder.glob("*.NCS"))
    )
    if not ncs_files:
        raise FileNotFoundError(f"No .ncs files found in {raw_folder}")
    return sorted(ncs_files)[0]


def read_ncs_timestamps(ncs_path: Path) -> np.ndarray:
    NCS_RECORD_SIZE = 1044
    timestamps = []
    with open(ncs_path, "rb") as f:
        f.seek(NLX_HEADER_SIZE)
        while True:
            ts_bytes = f.read(8)
            if not ts_bytes or len(ts_bytes) != 8:
                break
            timestamps.append(struct.unpack("<Q", ts_bytes)[0])
            f.seek(NCS_RECORD_SIZE - 8, 1)
    if not timestamps:
        raise ValueError(f"No timestamps read from {ncs_path}")
    return np.asarray(timestamps, dtype=np.int64)


def infer_segments_from_ncs(raw_folder: Path) -> pd.DataFrame:
    ncs_path = get_first_ncs_file(raw_folder)
    timestamps = read_ncs_timestamps(ncs_path)
    diffs = np.diff(timestamps)
    normal_step = np.median(diffs)
    gap_threshold = max(normal_step * 10, 1_000_000)
    break_after = np.where(diffs > gap_threshold)[0]
    starts = np.r_[0, break_after + 1]
    ends = np.r_[break_after, len(timestamps) - 1]
    segments = pd.DataFrame(
        {
            "segment_index": np.arange(len(starts), dtype=int),
            "segment_start_timestamp_us": timestamps[starts],
            "segment_end_timestamp_us": timestamps[ends],
            "n_ncs_records": ends - starts + 1,
        }
    )
    segments["duration_sec_approx"] = (
        segments["segment_end_timestamp_us"] - segments["segment_start_timestamp_us"]
    ) / 1_000_000.0
    print(f"Using NCS file to infer segments: {ncs_path.name}")
    print("\nInferred segments:")
    print(segments)
    return segments

def get_selected_segment_bounds(raw_folder: Path) -> tuple[int, int]:
    if SEGMENT_START_TIMESTAMP_US is not None and SEGMENT_END_TIMESTAMP_US is not None:
        return int(SEGMENT_START_TIMESTAMP_US), int(SEGMENT_END_TIMESTAMP_US)

    segments = infer_segments_from_ncs(raw_folder)

    match = segments.loc[
        segments["segment_index"] == ORIGINAL_RECORDING_SEGMENT_INDEX
    ]

    if match.empty:
        raise ValueError(
            f"ORIGINAL_RECORDING_SEGMENT_INDEX={ORIGINAL_RECORDING_SEGMENT_INDEX} not found. "
            f"Available segment indices: {segments['segment_index'].tolist()}"
        )

    row = match.iloc[0]

    return (
        int(row["segment_start_timestamp_us"]),
        int(row["segment_end_timestamp_us"]),
    )


def debounce_ttl_rising(ttl_raw: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """
    Collapse TTL bursts into one event per burst.

    Keeps the FIRST TTL in each burst because that is closest to the true moving onset.
    Adds columns documenting burst structure.
    """
    if ttl_raw.empty:
        return ttl_raw.copy()

    ttl = ttl_raw.sort_values("event_time_sec").reset_index(drop=True).copy()
    dt = ttl["event_time_sec"].diff()
    ttl["ttl_burst_id"] = (dt.isna() | (dt > window_sec)).cumsum() - 1

    burst_stats = (
        ttl.groupby("ttl_burst_id", dropna=False)
        .agg(
            ttl_burst_n_pulses=("ttl_index_raw", "size"),
            ttl_burst_first_time_sec=("event_time_sec", "first"),
            ttl_burst_last_time_sec=("event_time_sec", "last"),
        )
        .reset_index()
    )
    burst_stats["ttl_burst_duration_ms"] = (
        burst_stats["ttl_burst_last_time_sec"]
        - burst_stats["ttl_burst_first_time_sec"]
    ) * 1000.0

    dedup = ttl.groupby("ttl_burst_id", as_index=False).first()
    dedup = dedup.merge(burst_stats, on="ttl_burst_id", how="left")

    dedup = dedup.reset_index(drop=True)
    dedup["ttl_index"] = np.arange(len(dedup), dtype=int)

    preferred_cols = [
        "ttl_index",
        "ttl_index_raw",
        "ttl_burst_id",
        "ttl_burst_n_pulses",
        "ttl_burst_duration_ms",
        "segment_index",
        "record_index",
        "timestamp_us",
        "segment_start_timestamp_us",
        "event_time_sec",
        "ttl_value",
        "event_id",
        "event_string",
    ]
    preferred_cols = [c for c in preferred_cols if c in dedup.columns]
    other_cols = [c for c in dedup.columns if c not in preferred_cols]
    return dedup[preferred_cols + other_cols]


def main() -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== Extract TTL rising events for selected segment, then debounce bursts =====")
    print(f"Events file: {EVENTS_NEV_PATH}")
    print(f"Raw folder:  {RAW_NLX_FOLDER}")
    print(f"Segment index: {RECORDING_SEGMENT_INDEX}")
    print(f"TTL_DEBOUNCE_WINDOW_SEC: {TTL_DEBOUNCE_WINDOW_SEC}")

    events = read_nev(EVENTS_NEV_PATH)
    seg_start_us, seg_end_us = get_selected_segment_bounds(RAW_NLX_FOLDER)

    print("\nSelected segment bounds:")
    print(f"Start: {seg_start_us}")
    print(f"End:   {seg_end_us}")
    print(f"Approx duration: {(seg_end_us - seg_start_us) / 1_000_000:.2f} sec")

    test_start_us = seg_start_us + int(round(TEST_START_SEC_WITHIN_ORIGINAL_SEGMENT * 1_000_000))

    if TEST_END_SEC_WITHIN_ORIGINAL_SEGMENT is None:
        test_end_us = seg_end_us
    else:
        test_end_us = seg_start_us + int(round(TEST_END_SEC_WITHIN_ORIGINAL_SEGMENT * 1_000_000))

    events_seg = events.loc[
        (events["timestamp_us"] >= test_start_us)
        & (events["timestamp_us"] <= test_end_us)
    ].copy()

    ttl_raw = events_seg.loc[events_seg["ttl_value"] > 0].copy()
    ttl_raw = ttl_raw.sort_values("timestamp_us").reset_index(drop=True)
    ttl_raw["ttl_index_raw"] = np.arange(len(ttl_raw), dtype=int)
    ttl_raw["segment_index"] = RECORDING_SEGMENT_INDEX
    ttl_raw["segment_start_timestamp_us"] = test_start_us
    ttl_raw["original_segment_start_timestamp_us"] = seg_start_us
    ttl_raw["test_start_timestamp_us"] = test_start_us
    ttl_raw["test_end_timestamp_us"] = test_end_us
    ttl_raw["event_time_sec"] = (
        ttl_raw["timestamp_us"] - test_start_us
    ) / 1_000_000.0

    keep_cols_raw = [
        "ttl_index_raw",
        "segment_index",
        "record_index",
        "timestamp_us",
        "segment_start_timestamp_us",
        "original_segment_start_timestamp_us",
        "test_start_timestamp_us",
        "test_end_timestamp_us",
        "event_time_sec",
        "ttl_value",
        "event_id",
        "event_string",
    ]
    ttl_raw = ttl_raw[keep_cols_raw]

    ttl_dedup = debounce_ttl_rising(ttl_raw, TTL_DEBOUNCE_WINDOW_SEC)

    raw_out = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment_raw.csv"
    dedup_out = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    ttl_raw.to_csv(raw_out, index=False)
    ttl_dedup.to_csv(dedup_out, index=False)

    print("\n===== Output =====")
    print(f"Saved raw TTLs:       {raw_out}")
    print(f"Saved debounced TTLs: {dedup_out}")
    print(f"Raw TTL count:        {len(ttl_raw)}")
    print(f"Debounced TTL count:  {len(ttl_dedup)}")

    if len(ttl_dedup) > 0 and "ttl_burst_n_pulses" in ttl_dedup.columns:
        print("\nTTL pulses per burst:")
        print(ttl_dedup["ttl_burst_n_pulses"].value_counts().sort_index())
        print("\nTTL burst duration ms:")
        print(ttl_dedup["ttl_burst_duration_ms"].describe())

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"\nExpected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")
        if len(ttl_dedup) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: debounced TTL count still does not match "
                "EXPECTED_MOTION_TTL_COUNT. Check selected segment, stimlog, or config."
            )

    print("\nFirst few debounced TTLs:")
    print(ttl_dedup.head())


if __name__ == "__main__":
    main()
