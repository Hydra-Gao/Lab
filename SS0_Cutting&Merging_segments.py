
from pathlib import Path
import json
import multiprocessing

import spikeinterface as si
import spikeinterface.extractors as se

RAW_FOLDER = r"G:\Lab\Raw_data\TG964\2026-06-24_13-48-39"
OUT_ROOT = r"G:\Lab\Raw_data\TG964\split_experiments"

EXP1_SEGMENTS = [0, 1]
EXP2_SEGMENT = 3
EXP3_SEGMENT = 4

EXP2_CUT_TIME_SEC = 1864.0
EXP3_CUT_TIME_SEC = 1768.0


def split_single_segment_by_time(recording, segment_index, cut_time_sec):
    fs = recording.get_sampling_frequency()

    mono = recording.select_segments([segment_index])
    n_frames = mono.get_num_frames(segment_index=0)

    cut_frame = int(round(cut_time_sec * fs))

    if cut_frame <= 0 or cut_frame >= n_frames:
        raise ValueError(
            f"Invalid cut_frame={cut_frame}. "
            f"Segment {segment_index} has {n_frames} frames "
            f"({n_frames / fs:.3f} sec)."
        )

    part1 = mono.frame_slice(start_frame=0, end_frame=cut_frame)
    part2 = mono.frame_slice(start_frame=cut_frame, end_frame=n_frames)

    return part1, part2


def print_recording_summary(name, recording):
    fs = recording.get_sampling_frequency()
    print(f"\n{name}")
    print(f"num segments: {recording.get_num_segments()}")
    print(f"num channels: {recording.get_num_channels()}")
    print(f"sampling frequency: {fs}")

    for seg_idx in range(recording.get_num_segments()):
        n = recording.get_num_frames(segment_index=seg_idx)
        print(f"  segment {seg_idx}: {n} frames, {n / fs:.3f} sec")


def main():
    raw_rec = se.read_neuralynx(RAW_FOLDER)

    print_recording_summary("Raw recording", raw_rec)

    # Experiment 1
    exp1_rec = si.append_recordings([
        raw_rec.select_segments([EXP1_SEGMENTS[0]]),
        raw_rec.select_segments([EXP1_SEGMENTS[1]]),
    ])

    # Experiment 2
    exp2_part1, exp2_part2 = split_single_segment_by_time(
        raw_rec,
        segment_index=EXP2_SEGMENT,
        cut_time_sec=EXP2_CUT_TIME_SEC,
    )

    exp2_rec = si.append_recordings([
        exp2_part1,
        exp2_part2,
    ])

    # Experiment 3
    exp3_part1, exp3_part2 = split_single_segment_by_time(
        raw_rec,
        segment_index=EXP3_SEGMENT,
        cut_time_sec=EXP3_CUT_TIME_SEC,
    )

    exp3_rec = si.append_recordings([
        exp3_part1,
        exp3_part2,
    ])

    print_recording_summary("Experiment 1", exp1_rec)
    print_recording_summary("Experiment 2", exp2_rec)
    print_recording_summary("Experiment 3", exp3_rec)

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    exp1_rec.save(
        folder=out_root / "experiment_1",
        format="binary",
        n_jobs=8,
        chunk_duration="1s",
        overwrite=True,
    )

    exp2_rec.save(
        folder=out_root / "experiment_2",
        format="binary",
        n_jobs=8,
        chunk_duration="1s",
        overwrite=True,
    )

    exp3_rec.save(
        folder=out_root / "experiment_3",
        format="binary",
        n_jobs=8,
        chunk_duration="1s",
        overwrite=True,
    )

    split_info = {
        "raw_folder": RAW_FOLDER,
        "exp1_original_segments": EXP1_SEGMENTS,
        "exp2_original_segment": EXP2_SEGMENT,
        "exp2_cut_time_sec_within_segment": EXP2_CUT_TIME_SEC,
        "exp3_original_segment": EXP3_SEGMENT,
        "exp3_cut_time_sec_within_segment": EXP3_CUT_TIME_SEC,
        "saved_as": "SpikeInterface binary folder",
    }

    with open(out_root / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=4)

    print("\nSaved all three experiments.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()