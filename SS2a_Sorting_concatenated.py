import spikeinterface.full as si
from spikeinterface.core import BinaryFolderRecording
from multiprocessing import freeze_support
import pandas as pd
import numpy as np

from config_local import WORKING_DIR, SEGMENT_INDEX_TO_USE


# =====================
# User settings
# =====================

PREPROCESSED_FOLDER = WORKING_DIR / "preprocessed_M12"

# [0, 1, 2] for all segments; change to [0] or [1] or [2] to sort only one segment; or [1, 2] to sort segments 1 and 2 together while skipping segment 0.
SEGMENTS_TO_SORT = SEGMENT_INDEX_TO_USE 

SORTER_NAME = "kilosort4"
SORTER_FOLDER = WORKING_DIR / "kilosort4_M12_test_output"

SEGMENT_TIME_MAP_PATH = WORKING_DIR / "segment_time_map_M12.csv"


# =====================
# Helper functions
# =====================

def print_recording_summary(recording, name="recording"):
    print(f"\n===== {name} =====")
    print(recording)
    print("Number of segments:", recording.get_num_segments())
    print("Channel IDs:", recording.get_channel_ids())
    print("Sampling frequency:", recording.get_sampling_frequency())
    print("Total duration:", recording.get_total_duration())

    try:
        print("Channel locations:")
        print(recording.get_channel_locations())
    except Exception as exc:
        print("Could not print channel locations:", exc)

    for seg_idx in range(recording.get_num_segments()):
        n_samples = recording.get_num_samples(segment_index=seg_idx)
        duration = n_samples / recording.get_sampling_frequency()
        print(
            f"Segment {seg_idx}: "
            f"n_samples={n_samples}, duration_sec={duration:.6f}"
        )


def make_concatenated_recording(recording_saved, segments_to_sort):
    """
    Select user-defined segments and concatenate them into one single-segment recording.

    Sorter time after this step is:
        concat_time_sec = frame / sampling_frequency

    Segment gaps from the original Neuralynx file are intentionally removed.
    TTL/stimlog alignment should later use segment_time_map_M12.csv.
    """
    n_segments_total = recording_saved.get_num_segments()

    for seg in segments_to_sort:
        if seg < 0 or seg >= n_segments_total:
            raise ValueError(
                f"SEGMENTS_TO_SORT contains invalid segment {seg}. "
                f"Available segments are 0 to {n_segments_total - 1}."
            )

    selected_recordings = []

    for seg in segments_to_sort:
        # select_segments([seg]) returns a one-segment recording
        rec_seg = recording_saved.select_segments([seg])

        # Optional but explicit: keep the full segment
        rec_seg = rec_seg.frame_slice(
            start_frame=0,
            end_frame=rec_seg.get_num_frames(segment_index=0),
        )

        selected_recordings.append(rec_seg)

    if len(selected_recordings) == 1:
        recording_concat = selected_recordings[0]
    else:
        recording_concat = si.concatenate_recordings(selected_recordings)

    return recording_concat


def save_segment_time_map(recording_saved, segments_to_sort, output_path):
    """
    Save mapping between original segment index and concatenated sorter time.

    This file is the bridge for later alignment:

        original segment-local time
            -> concat_start_sec + local time
            -> sorter spike time axis

    Important:
    This version maps based on sample counts/durations in the preprocessed recording.
    Later SB01 can add Neuralynx absolute timestamps from NCS/NEV if needed.
    """
    fs = recording_saved.get_sampling_frequency()

    rows = []
    concat_start_frame = 0

    for concat_order, original_seg_index in enumerate(segments_to_sort):
        n_samples = recording_saved.get_num_samples(segment_index=original_seg_index)
        duration_sec = n_samples / fs

        concat_end_frame = concat_start_frame + n_samples

        rows.append(
            {
                "concat_order": concat_order,
                "original_segment_index": original_seg_index,

                "sampling_frequency": fs,
                "n_samples": int(n_samples),
                "duration_sec": float(duration_sec),

                "concat_start_frame": int(concat_start_frame),
                "concat_end_frame": int(concat_end_frame),

                "concat_start_sec": float(concat_start_frame / fs),
                "concat_end_sec": float(concat_end_frame / fs),
            }
        )

        concat_start_frame = concat_end_frame

    segment_map = pd.DataFrame(rows)
    segment_map.to_csv(output_path, index=False)

    print("\n===== Saved segment time map =====")
    print(output_path)
    print(segment_map)

    return segment_map


def main():
    print("===== SS2 Sorting with selected segment concatenation =====")

    # 1. Load preprocessed multi-segment recording
    recording_saved = BinaryFolderRecording(PREPROCESSED_FOLDER)

    print_recording_summary(recording_saved, name="Loaded preprocessed recording")

    print("\nSegments selected for sorting:")
    print(SEGMENTS_TO_SORT)

    # 2. Save concat mapping for later TTL/stimlog alignment
    save_segment_time_map(
        recording_saved=recording_saved,
        segments_to_sort=SEGMENTS_TO_SORT,
        output_path=SEGMENT_TIME_MAP_PATH,
    )

    # 3. Make one single-segment recording for Kilosort4
    recording_test = make_concatenated_recording(
        recording_saved=recording_saved,
        segments_to_sort=SEGMENTS_TO_SORT,
    )

    print_recording_summary(recording_test, name="Recording sent to sorter")

    if recording_test.get_num_segments() != 1:
        raise RuntimeError(
            "Kilosort4 recording should be single-segment after concatenation."
        )

    # 4. Run sorter
    import spikeinterface.sorters as ss
    params = ss.get_default_sorter_params("kilosort4")

    params.update({
        "save_preprocessed_copy": True,
        "highpass_cutoff": 10,
        "do_CAR": False,
        "whitening_range": 5,
        "do_correction": False,
        # "duplicate_spike_ms": 0.1,
        # "skip_kilosort_preprocessing": True
    })

    sorting_test = si.run_sorter(
        sorter_name=SORTER_NAME,
        recording=recording_test,
        folder=SORTER_FOLDER,
        remove_existing_folder=True,
        docker_image=False,
        verbose=True,
        **params
    )

    print("\n===== Sorting finished =====")
    print(sorting_test)
    print("Spike counts:")
    print(sorting_test.count_num_spikes_per_unit())


if __name__ == "__main__":
    freeze_support()
    main()