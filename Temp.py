# # import spikeinterface.full as si
# # from pathlib import Path

# # preprocessed_folder = Path(r"F:\Lab\Processing\Working_dir_processing\preprocessed_M12")

# # recording = si.load(preprocessed_folder)

# # print(recording)
# # print(recording.get_channel_ids())
# # print(recording.get_sampling_frequency())
# # print(recording.get_num_channels())
# # print(recording.get_num_segments())

# # import matplotlib.pyplot as plt

# # si.plot_traces(
# #     recording,
# #     segment_index=1,   # if you want the same segment as Phy
# #     time_range=(315, 320),
# #     backend="matplotlib"
# # )

# # plt.show()
# import spikeinterface.full as si
# from pathlib import Path
# import matplotlib.pyplot as plt

# preprocessed_folder = Path(r"F:\Lab\Processing\Working_dir_processing\preprocessed_M12")
# recording = si.load(preprocessed_folder)

# fs = recording.get_sampling_frequency()

# seg = 1
# t0 = 315
# t1 = 320

# start_frame = int(t0 * fs)
# end_frame = int(t1 * fs)

# # First select the second segment only
# recording_seg1 = recording.select_segments([seg])

# # Then slice 315-320 s within that selected segment
# recording_slice = recording_seg1.frame_slice(
#     start_frame=start_frame,
#     end_frame=end_frame
# )

# print(recording_slice)
# print("slice duration:", recording_slice.get_total_duration())
# print("slice shape:", recording_slice.get_traces(segment_index=0).shape)

# times = recording_slice.get_times(segment_index=0)

# print(times[0])
# print(times[-1])
# print(len(times))

# # Now plot from 0 to 5 s because the sliced recording starts at 0
# si.plot_traces(
#     recording_slice,
#     segment_index=0,
#     time_range=(times[0], times[-1]),
#     backend="matplotlib"
# )

# plt.show()

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 改成你的 Phy folder =====
phy_folder = Path(r"F:\Lab\Processing\Output_dir_almost\phy_M12_kilosort4_curated")

# ===== 2. 读取 params.py 里的 dat_path / dtype / n_channels_dat / sample_rate =====
params_path = phy_folder / "params.py"

params = {}
with open(params_path, "r", encoding="utf-8") as f:
    exec(f.read(), {}, params)

dat_path = Path(params["dat_path"])
if not dat_path.is_absolute():
    dat_path = phy_folder / dat_path

n_channels = int(params["n_channels_dat"])
dtype = np.dtype(params["dtype"])
fs = float(params["sample_rate"])

print("dat_path:", dat_path)
print("n_channels:", n_channels)
print("dtype:", dtype)
print("fs:", fs)

# ===== 3. 切出 315–320 s =====
t0 = 315
t1 = 320

start_frame = int(t0 * fs)
end_frame = int(t1 * fs)
n_frames = end_frame - start_frame

# .dat 一般是 time x channels interleaved
data = np.memmap(dat_path, dtype=dtype, mode="r")
total_frames = data.size // n_channels
data = data.reshape(total_frames, n_channels)

print("total_frames:", total_frames)
print("total_duration_sec:", total_frames / fs)
print("slice frames:", start_frame, end_frame)

slice_data = np.asarray(data[start_frame:end_frame, :])

print("slice shape:", slice_data.shape)
print("slice duration:", slice_data.shape[0] / fs)

# ===== 4. 画所有 channel，加 vertical offset =====
time = np.arange(slice_data.shape[0]) / fs + t0

# 建议先减去每个 channel 的中位数，方便看形状
plot_data = slice_data.astype(float)
plot_data = plot_data - np.median(plot_data, axis=0, keepdims=True)

# 自动设置 offset
offset = np.nanpercentile(np.abs(plot_data), 99) * 4
if offset == 0 or np.isnan(offset):
    offset = 100

plt.figure(figsize=(14, 10))

for ch in range(n_channels):
    plt.plot(time, plot_data[:, ch] + ch * offset, linewidth=0.4)

plt.xlabel("Time (s)")
plt.ylabel("Channel index in Phy .dat order")
plt.title("Phy .dat trace view: 315–320 s")
plt.yticks(np.arange(n_channels) * offset, [str(i) for i in range(n_channels)])
plt.tight_layout()
plt.show()