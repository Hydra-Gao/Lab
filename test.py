import spikeinterface.full as si
from config_local import WORKING_DIR, SEGMENT_INDEX_TO_USE

# 这里按你的实际 sorter 改
SORTER_NAME = "kilosort4"

recording = si.load(WORKING_DIR / "preprocessed_M12")

# 和你 sorting 时保持一致：你 SS2 里用了 select_segments([1])
recording_seg = recording.select_segments([SEGMENT_INDEX_TO_USE])

sorting = si.load(WORKING_DIR / f"{SORTER_NAME}_M12_test_output")

n_frames = recording_seg.get_num_frames(segment_index=0)

print("Recording frames:", n_frames)
print("Recording duration sec:", recording_seg.get_total_duration())

total_excess = 0

for unit_id in sorting.get_unit_ids():
    spikes = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)

    excess_mask = spikes >= n_frames
    n_excess = excess_mask.sum()

    if n_excess > 0:
        total_excess += n_excess
        print(
            f"Unit {unit_id}: {n_excess} excess spikes "
            f"/ {len(spikes)} total, "
            f"max_frame={spikes.max()}"
        )

print("\nTotal excess spikes:", total_excess)