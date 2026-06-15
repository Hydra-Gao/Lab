import spikeinterface.full as si
from spikeinterface.core import BinaryFolderRecording
from multiprocessing import freeze_support
import pandas as pd

from config_local import OUTPUT_DIR, WORKING_DIR, SEGMENT_INDEX_TO_USE


# =====================
# User settings
# =====================

PREPROCESSED_FOLDER = WORKING_DIR / "preprocessed_M12"

SEGMENTS_TO_SORT = SEGMENT_INDEX_TO_USE 

SEGMENT_TIME_MAP_PATH = WORKING_DIR / "segment_time_map_M12.csv"

CURATED_SORTING_FOLDERS = {
    "kilosort4": WORKING_DIR / "sorting_M12_kilosort4_curated",
    "mountainsort5": WORKING_DIR / "sorting_M12_mountainsort5_curated",
}

ANALYZER_FOLDERS = {
    name: WORKING_DIR / f"analyzer_M12_{name}_curated"
    for name in CURATED_SORTING_FOLDERS
}

PHY_FOLDERS = {
    name: OUTPUT_DIR / f"phy_M12_{name}_curated"
    for name in CURATED_SORTING_FOLDERS
}

QUALITY_METRIC_NAMES = [
    "snr",
    "isi_violation",
    "firing_rate",
    "presence_ratio",
]


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
    Rebuild the exact same concatenated recording used in SS2.

    Important:
    The sorting object has spike frames relative to this concatenated recording.
    Therefore analyzer / Phy export must use this same recording.
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
        rec_seg = recording_saved.select_segments([seg])

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


def save_or_check_segment_time_map(recording_saved, segments_to_sort, output_path):
    """
    Make sure SS4 is using the same segment mapping as SS2.

    If segment_time_map_M12.csv already exists, check that the selected
    segments match. If it does not exist, create it.
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

    current_map = pd.DataFrame(rows)

    if output_path.exists():
        old_map = pd.read_csv(output_path)

        old_segments = old_map["original_segment_index"].astype(int).tolist()
        new_segments = current_map["original_segment_index"].astype(int).tolist()

        if old_segments != new_segments:
            raise ValueError(
                "\nsegment_time_map_M12.csv exists but does not match SS4 "
                "SEGMENTS_TO_SORT.\n"
                f"Existing map segments: {old_segments}\n"
                f"Current SS4 segments:  {new_segments}\n\n"
                "This usually means SS2 and SS4 are using different segment lists. "
                "Make them identical before exporting to Phy."
            )

        print("\n===== Existing segment time map checked OK =====")
        print(output_path)
        print(old_map)

    else:
        current_map.to_csv(output_path, index=False)

        print("\n===== Saved new segment time map =====")
        print(output_path)
        print(current_map)

    return current_map


def export_one_sorter_to_phy(sorter_name, recording):
    curated_sorting_folder = CURATED_SORTING_FOLDERS[sorter_name]
    analyzer_folder = ANALYZER_FOLDERS[sorter_name]
    phy_folder = PHY_FOLDERS[sorter_name]

    print(f"\n===== Exporting {sorter_name} to Phy =====")

    if not curated_sorting_folder.exists():
        print(f"Skipping {sorter_name}: curated sorting folder not found")
        print(curated_sorting_folder)
        return

    # 1. Load curated sorting generated by SS3_Post_sorting_curation.py
    sorting = si.load(curated_sorting_folder)

    print("Curated sorting:")
    print(sorting)
    print("Curated spike counts:")
    print(sorting.count_num_spikes_per_unit())

    # 2. Sanity check: sorter output should be single segment
    if sorting.get_num_segments() != 1:
        raise RuntimeError(
            f"{sorter_name} sorting has {sorting.get_num_segments()} segments. "
            "Expected 1 segment because SS2 should have used concatenated recording."
        )

    if recording.get_num_segments() != 1:
        raise RuntimeError(
            f"Recording sent to analyzer has {recording.get_num_segments()} segments. "
            "Expected 1 segment after concatenation."
        )

    # 3. Create analyzer using the SAME concatenated recording as SS2
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=analyzer_folder,
        overwrite=True,
    )

    # 4. Compute extensions needed for waveform inspection, metrics, and Phy export
    analyzer.compute("random_spikes")
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("noise_levels")

    analyzer.compute(
        "quality_metrics",
        metric_names=QUALITY_METRIC_NAMES,
    )

    metrics = analyzer.get_extension("quality_metrics").get_data()

    print("\nQuality metrics:")
    print(metrics)

    metrics_path = WORKING_DIR / f"quality_metrics_M12_{sorter_name}_curated.csv"
    metrics.to_csv(metrics_path)

    print(f"Saved quality metrics: {metrics_path}")

    # 5. Export to Phy
    si.export_to_phy(
        analyzer,
        output_folder=phy_folder,
        remove_if_exists=True,
    )

    print(f"Exported Phy folder: {phy_folder}")


def main():
    print("===== SS4 Export Phy with selected segment concatenation =====")

    # 1. Load preprocessed multi-segment recording
    recording_saved = BinaryFolderRecording(PREPROCESSED_FOLDER)

    print_recording_summary(
        recording_saved,
        name="Loaded preprocessed recording",
    )

    print("\nSegments selected for analyzer / Phy export:")
    print(SEGMENTS_TO_SORT)

    # 2. Check / save segment time map
    save_or_check_segment_time_map(
        recording_saved=recording_saved,
        segments_to_sort=SEGMENTS_TO_SORT,
        output_path=SEGMENT_TIME_MAP_PATH,
    )

    # 3. Rebuild the same concatenated recording used in SS2
    recording_test = make_concatenated_recording(
        recording_saved=recording_saved,
        segments_to_sort=SEGMENTS_TO_SORT,
    )

    print_recording_summary(
        recording_test,
        name="Recording sent to analyzer / Phy",
    )

    if recording_test.get_num_segments() != 1:
        raise RuntimeError(
            "Analyzer recording should be single-segment after concatenation."
        )

    # 4. Export each curated sorter
    for sorter_name in CURATED_SORTING_FOLDERS:
        export_one_sorter_to_phy(sorter_name, recording_test)


if __name__ == "__main__":
    freeze_support()
    main()