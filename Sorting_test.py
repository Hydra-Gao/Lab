import spikeinterface.full as si
from spikeinterface.core import BinaryFolderRecording

recording_saved = BinaryFolderRecording(r"E:\Lab\preprocessed_M12")

print(recording_saved)
print("Channel IDs:", recording_saved.get_channel_ids())
print("Sampling frequency:", recording_saved.get_sampling_frequency())
print("Duration:", recording_saved.get_total_duration())
print("Channel locations:")
print(recording_saved.get_channel_locations())

""" import matplotlib.pyplot as plt

si.plot_traces(
    recording_saved,
    segment_index=0,
    time_range=(0, 50),
    backend="matplotlib"
)

plt.show() """

from spikeinterface.sorters import Kilosort4Sorter

from spikeinterface.core import select_segment_recording

recording_seg0 = recording_saved.select_segments([0])

recording_test = recording_seg0.frame_slice(
    start_frame=0,
    end_frame=recording_seg0.get_num_frames(segment_index=0)
)

from config_local import WORKING_DIR

sorter_folder = WORKING_DIR / "kilosort4_M12_test_output"

sorting_test = si.run_sorter(
    sorter_name="kilosort4",
    recording=recording_test,
    folder=sorter_folder,
    remove_existing_folder=True
)