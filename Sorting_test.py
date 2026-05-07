import spikeinterface.full as si
from spikeinterface.core import BinaryFolderRecording

from config_local import WORKING_DIR
preprocessed_folder = WORKING_DIR / "preprocessed_M12"
recording_saved = BinaryFolderRecording(preprocessed_folder)

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

recording_seg = recording_saved.select_segments([0])

recording_test = recording_seg.frame_slice(
    start_frame=0,
    end_frame=recording_seg.get_num_frames(segment_index=0)
)

from config_local import WORKING_DIR

sorter_folder_kilosort4 = WORKING_DIR / "kilosort4_M12_test_output"
sorter_folder_mountainsort5 = WORKING_DIR / "mountainsort5_M12_test_output"

""" sorting_test = si.run_sorter(
    sorter_name="kilosort4",
    recording=recording_test,
    folder=sorter_folder_kilosort4,
    remove_existing_folder=True,
    docker_image=False,
    verbose=True
) """

sorting_test = si.run_sorter(
    sorter_name="mountainsort5",
    recording=recording_test,
    folder=sorter_folder_mountainsort5,
    remove_existing_folder=True,
    docker_image=True,
    verbose=True
)