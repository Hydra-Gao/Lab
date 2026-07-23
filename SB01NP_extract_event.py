# SB01_extract_events_spikeglx.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import SB0_config_analysis as cfg


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Required in SB0_config_analysis.py:
#   SPIKEGLX_BIN_PATH = Path(r"...\recording.ap.bin")  # or nidq.bin
#   ANALYSIS_OUTPUT_DIR = Path(r"...\analysis_output")
#
# Optional:
#   SPIKEGLX_META_PATH = None        # inferred from .bin when omitted
#   DIGITAL_WORD_CHANNEL_INDEX = -1 # saved-channel index containing the word
#   EVENT_BIT = 6                    # zero-based bit number; bit 6 => 1 << 6
#   MIN_EVENT_INTERVAL_SEC = 0.001   # debounce; set 0 to disable
#   EXPECTED_MOTION_TTL_COUNT = None

SPIKEGLX_BIN_PATH = Path(cfg.SPIKEGLX_BIN_PATH)
SPIKEGLX_META_PATH = getattr(cfg, "SPIKEGLX_META_PATH", None)
if SPIKEGLX_META_PATH is not None:
    SPIKEGLX_META_PATH = Path(SPIKEGLX_META_PATH)

ANALYSIS_OUTPUT_DIR = Path(cfg.ANALYSIS_OUTPUT_DIR)
DIGITAL_WORD_CHANNEL_INDEX = int(getattr(cfg, "DIGITAL_WORD_CHANNEL_INDEX", -1))
EVENT_BIT = int(getattr(cfg, "EVENT_BIT", 6))
MIN_EVENT_INTERVAL_SEC = float(getattr(cfg, "MIN_EVENT_INTERVAL_SEC", 0.001))
EXPECTED_MOTION_TTL_COUNT = getattr(cfg, "EXPECTED_MOTION_TTL_COUNT", None)

# Number of samples processed at once. The file itself remains memory-mapped.
CHUNK_SAMPLES = int(getattr(cfg, "EVENT_EXTRACTION_CHUNK_SAMPLES", 5_000_000))


def infer_meta_path(bin_path: Path) -> Path:
    """Infer the matching SpikeGLX .meta path from a .bin path."""
    if bin_path.suffix.lower() != ".bin":
        raise ValueError(f"Expected a .bin file, got: {bin_path}")

    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Matching .meta file was not found: {meta_path}\n"
            "Set SPIKEGLX_META_PATH explicitly if it has a different name."
        )
    return meta_path


def read_spikeglx_meta(meta_path: Path) -> dict[str, str]:
    """Read key=value fields from a SpikeGLX .meta file."""
    metadata: dict[str, str] = {}

    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            metadata[key.lstrip("~")] = value

    if not metadata:
        raise ValueError(f"No metadata fields could be read from {meta_path}")

    return metadata


def get_sampling_frequency(meta: dict[str, str]) -> float:
    """Get sample rate from imec or NI metadata."""
    for key in ("imSampRate", "niSampRate"):
        if key in meta:
            fs = float(meta[key])
            if fs <= 0:
                raise ValueError(f"Invalid {key}={fs}")
            return fs

    raise KeyError("Metadata contains neither imSampRate nor niSampRate.")


def get_num_saved_channels(meta: dict[str, str]) -> int:
    """Get the number of interleaved int16 channels in the .bin file."""
    if "nSavedChans" not in meta:
        raise KeyError("Metadata does not contain nSavedChans.")

    n_channels = int(meta["nSavedChans"])
    if n_channels <= 0:
        raise ValueError(f"Invalid nSavedChans={n_channels}")
    return n_channels


def normalize_channel_index(index: int, n_channels: int) -> int:
    """Convert a possibly negative channel index to a validated positive index."""
    normalized = index if index >= 0 else n_channels + index
    if not 0 <= normalized < n_channels:
        raise IndexError(
            f"DIGITAL_WORD_CHANNEL_INDEX={index} resolves to {normalized}, but "
            f"the file has {n_channels} saved channels."
        )
    return normalized


def find_rising_edges(
    bin_path: Path,
    n_channels: int,
    digital_channel_index: int,
    bit_number: int,
    chunk_samples: int,
) -> tuple[np.ndarray, int]:
    """
    Return sample indices where the selected bit changes from 0 to 1.

    SpikeGLX binary data are interleaved int16 values. The selected digital word
    is reinterpreted as uint16 before masking so signed int16 values do not affect
    bit extraction.
    """
    if bit_number < 0 or bit_number > 15:
        raise ValueError("EVENT_BIT must be between 0 and 15 for one uint16 word.")

    file_size_bytes = bin_path.stat().st_size
    bytes_per_sample = n_channels * np.dtype(np.int16).itemsize

    if file_size_bytes % bytes_per_sample != 0:
        raise ValueError(
            f"File size {file_size_bytes} is not divisible by "
            f"nSavedChans × 2 = {bytes_per_sample}. Check the .bin/.meta pairing."
        )

    n_samples = file_size_bytes // bytes_per_sample
    raw = np.memmap(
        bin_path,
        dtype=np.int16,
        mode="r",
        shape=(n_samples, n_channels),
        order="C",
    )

    mask = np.uint16(1 << bit_number)
    rising_parts: list[np.ndarray] = []
    previous_state = np.uint8(0)

    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)

        # Copy only one saved channel for this chunk, not the full recording.
        words = np.asarray(raw[start:stop, digital_channel_index], dtype=np.int16)
        words_u16 = words.view(np.uint16)
        states = ((words_u16 & mask) != 0).astype(np.uint8)

        if states.size == 0:
            continue

        # Include the previous chunk's final state so an edge at this chunk's
        # first sample is detected correctly.
        with_previous = np.empty(states.size + 1, dtype=np.uint8)
        with_previous[0] = previous_state
        with_previous[1:] = states

        local_edges = np.flatnonzero(np.diff(with_previous.astype(np.int8)) == 1)
        if local_edges.size:
            rising_parts.append(local_edges.astype(np.int64) + start)

        previous_state = states[-1]

    if rising_parts:
        rising_samples = np.concatenate(rising_parts)
    else:
        rising_samples = np.array([], dtype=np.int64)

    return rising_samples, int(n_samples)


def debounce_edges(
    rising_samples: np.ndarray,
    sampling_frequency: float,
    minimum_interval_sec: float,
) -> np.ndarray:
    """Keep the first edge in each group separated by less than the threshold."""
    if rising_samples.size <= 1 or minimum_interval_sec <= 0:
        return rising_samples

    minimum_samples = max(1, int(round(minimum_interval_sec * sampling_frequency)))
    keep = np.r_[True, np.diff(rising_samples) >= minimum_samples]
    return rising_samples[keep]


def main() -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bin_path = SPIKEGLX_BIN_PATH
    meta_path = SPIKEGLX_META_PATH or infer_meta_path(bin_path)

    if not bin_path.exists():
        raise FileNotFoundError(f"SpikeGLX binary file not found: {bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"SpikeGLX metadata file not found: {meta_path}")

    meta = read_spikeglx_meta(meta_path)
    sampling_frequency = get_sampling_frequency(meta)
    n_channels = get_num_saved_channels(meta)
    digital_channel_index = normalize_channel_index(
        DIGITAL_WORD_CHANNEL_INDEX,
        n_channels,
    )

    print("===== Extract SpikeGLX bit rising events =====")
    print(f"Binary file: {bin_path}")
    print(f"Meta file:   {meta_path}")
    print(f"Sampling frequency: {sampling_frequency:.6f} Hz")
    print(f"Saved channels: {n_channels}")
    print(f"Digital word saved-channel index: {digital_channel_index}")
    print(f"Event bit: {EVENT_BIT} (mask = {1 << EVENT_BIT})")

    rising_samples_raw, n_samples = find_rising_edges(
        bin_path=bin_path,
        n_channels=n_channels,
        digital_channel_index=digital_channel_index,
        bit_number=EVENT_BIT,
        chunk_samples=CHUNK_SAMPLES,
    )

    rising_samples = debounce_edges(
        rising_samples_raw,
        sampling_frequency=sampling_frequency,
        minimum_interval_sec=MIN_EVENT_INTERVAL_SEC,
    )

    recording_duration_sec = n_samples / sampling_frequency
    event_times_sec = rising_samples.astype(np.float64) / sampling_frequency

    events = pd.DataFrame(
        {
            "ttl_index": np.arange(len(rising_samples), dtype=int),
            "segment_index": 0,
            "event_sample": rising_samples,
            "event_time_sec": event_times_sec,
            "digital_word_channel_index": digital_channel_index,
            "event_bit": EVENT_BIT,
            "ttl_value": 1,
        }
    )

    # Keep the original filename and event_time_sec field expected by SB02.
    out_path = ANALYSIS_OUTPUT_DIR / "events_ttl_rising_segment.csv"
    events.to_csv(out_path, index=False)

    print("\n===== Output =====")
    print(f"Recording samples: {n_samples}")
    print(f"Recording duration: {recording_duration_sec:.3f} sec")
    print(f"Raw rising edges: {len(rising_samples_raw)}")
    print(
        f"Rising edges after {MIN_EVENT_INTERVAL_SEC * 1000:.3f} ms debounce: "
        f"{len(rising_samples)}"
    )
    print(f"Saved: {out_path}")

    if EXPECTED_MOTION_TTL_COUNT is not None:
        print(f"Expected motion TTL count: {EXPECTED_MOTION_TTL_COUNT}")
        if len(events) != int(EXPECTED_MOTION_TTL_COUNT):
            print(
                "Warning: extracted event count does not match "
                "EXPECTED_MOTION_TTL_COUNT. Check the digital word channel, "
                "bit numbering, debounce threshold, and whether this file "
                "contains more than one stimulus run."
            )

    print("\nFirst few events:")
    print(events.head().to_string(index=False))


if __name__ == "__main__":
    main()
