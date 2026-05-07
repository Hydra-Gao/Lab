# 01_extract_events.py

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
)


NLX_HEADER_SIZE = 16 * 1024
NEV_RECORD_SIZE = 184

# Neuralynx NEV record format
NEV_STRUCT = struct.Struct("<hhhQhhhhh8i128s")


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

    return pd.DataFrame(rows).sort_values("timestamp_us").reset_index(drop=True)


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
    Read timestamps from a Neuralynx .ncs file.

    Each NCS record is 1044 bytes:
    timestamp uint64 + channel int32 + sample_freq int32 +
    num_valid_samples int32 + 512 int16 samples.
    """
    NCS_RECORD_SIZE = 1044
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


def infer_segments_from_ncs(raw_folder: Path) -> pd.DataFrame:
    """
    Infer recording segments from gaps in NCS timestamps.

    For continuous 32 kHz data, each NCS record has 512 samples,
    so the normal timestamp step is about 16000 us.
    Large gaps indicate a new recording segment.
    """
    ncs_path = get_first_ncs_file(raw_folder)
    timestamps = read_ncs_timestamps(ncs_path)

    diffs = np.diff(timestamps)

    # Robust estimate of normal NCS record step
    normal_step = np.median(diffs)

    # Anything much larger than normal is treated as a segment break.
    # This threshold is intentionally conservative.
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
        segments["segment_end_timestamp_us"]
        - segments["segment_start_timestamp_us"]
    ) / 1_000_000.0

    print(f"Using NCS file to infer segments: {ncs_path.name}")
    print("\nInferred segments:")
    print(segments)

    return segments


def get_selected_segment_bounds(raw_folder: Path) -> tuple[int, int]:
    """Get start/end timestamp for the selected recording segment."""
    if SEGMENT_START_TIMESTAMP_US is not None and SEGMENT_END_TIMESTAMP_US is not None:
        return int(SEGMENT_START_TIMESTAMP_US), int(SEGMENT_END_TIMESTAMP_US)

    segments = infer_segments_from_ncs(raw_folder)

    match = segments.loc[segments["segment_index"] == RECORDING_SEGMENT_INDEX]

    if match.empty:
        raise ValueError(
            f"RECORDING_SEGMENT_INDEX={RECORDING_SEGMENT_INDEX} not found. "
            f"Available segment indices: {segments['segment_index'].tolist()}"
        )

    row = match.iloc[0]

    return (
        int(row["segment_start_timestamp_us"]),
        int(row["segment_end_timestamp_us"]),
    )


def main() -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== Extract TTL rising events for selected segment =====")
    print(f"Events file: {EVENTS_NEV_PATH}")
    print(f"Raw folder:  {RAW_NLX_FOLDER}")
    print(f"Segment index: {RECORDING_SEGMENT_INDEX}")

    events = read_nev(EVENTS_NEV_PATH)

    # Optional debug: print all TTL rising/high events in the whole Events.nev
    """ all_ttl_rising = events.loc[events["ttl_value"] > 0].copy()
    all_ttl_rising = all_ttl_rising.sort_values("timestamp_us").reset_index(drop=True)
    all_ttl_rising["ttl_index_all"] = np.arange(len(all_ttl_rising), dtype=int)

    print("\n===== All TTL rising events in the whole Events.nev =====")
    print(f"Total TTL rising events in Events.nev: {len(all_ttl_rising)}")
    print(all_ttl_rising[[
        "ttl_index_all",
        "record_index",
        "timestamp_us",
        "ttl_value",
        "event_id",
        "event_string",
    ]].to_string(index=False)) """

    seg_start_us, seg_end_us = get_selected_segment_bounds(RAW_NLX_FOLDER)

    print("\nSelected segment bounds:")
    print(f"Start: {seg_start_us}")
    print(f"End:   {seg_end_us}")
    print(f"Approx duration: {(seg_end_us - seg_start_us) / 1_000_000:.2f} sec")

    # Keep events inside selected recording segment
    events_seg = events.loc[
        (events["timestamp_us"] >= seg_start_us)
        & (events["timestamp_us"] <= seg_end_us)
    ].copy()

    # Keep TTL rising/high events.
    # In this setup, the stimulus script writes TTL=1 during moving onset.
    ttl_rising = events_seg.loc[events_seg["ttl_value"] > 0].copy()

    ttl_rising = ttl_rising.sort_values("timestamp_us").reset_index(drop=True)
    ttl_rising["ttl_index"] = np.arange(len(ttl_rising), dtype=int)
    ttl_rising["segment_index"] = RECORDING_SEGMENT_INDEX
    ttl_rising["segment_start_timestamp_us"] = seg_start_us
    ttl_rising["event_time_sec"] = (
        ttl_rising["timestamp_us"] - seg_start_us
    ) / 1_000_000.0

    # Put useful columns first
    keep_cols = [
        "ttl_index",
        "segment_index",
        "record_index",
        "timestamp_us",
        "segment_start_timestamp_us",
        "event_time_sec",
        "ttl_value",
        "event_id",
        "event_string",
    ]

    ttl_rising = ttl_rising[keep_cols]

    out_path = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    ttl_rising.to_csv(out_path, index=False)

    print("\n===== Output =====")
    print(f"Saved: {out_path}")
    print(f"TTL rising events in selected segment: {len(ttl_rising)}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")

        if len(ttl_rising) != EXPECTED_MOTION_TTL_COUNT:
            print(
                "Warning: TTL count does not match EXPECTED_MOTION_TTL_COUNT. "
                "This may mean the selected segment contains multiple stimulus runs, "
                "or the stimlog only covers part of this segment."
            )

    print("\nFirst few TTLs:")
    print(ttl_rising.head())


if __name__ == "__main__":
    main()