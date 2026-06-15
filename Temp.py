# import spikeinterface.full as si
# from pathlib import Path

# preprocessed_folder = Path(r"F:\Lab\Processing\Working_dir_processing\preprocessed_M12")

# recording = si.load(preprocessed_folder)

# print(recording)
# print(recording.get_channel_ids())
# print(recording.get_sampling_frequency())
# print(recording.get_num_channels())
# print(recording.get_num_segments())

# import matplotlib.pyplot as plt

# si.plot_traces(
#     recording,
#     segment_index=1,   # if you want the same segment as Phy
#     time_range=(315, 320),
#     backend="matplotlib"
# )

# plt.show()
import spikeinterface.full as si
from pathlib import Path
import matplotlib.pyplot as plt

preprocessed_folder = Path(r"F:\Lab\Processing\Working_dir_processing\preprocessed_M12")
recording = si.load(preprocessed_folder)

fs = recording.get_sampling_frequency()

seg = 1
t0 = 315
t1 = 320

start_frame = int(t0 * fs)
end_frame = int(t1 * fs)

# First select the second segment only
recording_seg1 = recording.select_segments([seg])

# Then slice 315-320 s within that selected segment
recording_slice = recording_seg1.frame_slice(
    start_frame=start_frame,
    end_frame=end_frame
)

print(recording_slice)
print("slice duration:", recording_slice.get_total_duration())
print("slice shape:", recording_slice.get_traces(segment_index=0).shape)

times = recording_slice.get_times(segment_index=0)

print(times[0])
print(times[-1])
print(len(times))

# Now plot from 0 to 5 s because the sliced recording starts at 0
si.plot_traces(
    recording_slice,
    segment_index=0,
    time_range=(times[0], times[-1]),
    backend="matplotlib"
)

plt.show()
