import numpy as np
import spikeinterface as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

analyzer = si.load_sorting_analyzer(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\sorting_analyzer"
)
print(analyzer.sparsity.unit_id_to_channel_ids[527])
print(analyzer.get_extension("principal_components").params)
print(analyzer.get_extension("quality_metrics").params)


#########################################################################

# tsv_file = r"C:\Lab\Processing\TG963_site1-2\cluster_info.tsv"
# excel_file = r"C:\Lab\Processing\TG963_site1-2\cluster_info.csv"

# df = pd.read_csv(tsv_file, sep="\t")
# df.to_csv(excel_file, index=False)

####################################################################

# ks_folder = r"C:\Lab\Processing\TG963_site1-2"

# spike_times = np.load(
#     ks_folder + r"\spike_times.npy"
# ).squeeze()

# spike_clusters = np.load(
#     ks_folder + r"\spike_clusters.npy"
# ).squeeze()

# unit_id = 70

# unit_spike_times = spike_times[
#     spike_clusters == unit_id
# ]

# print("n spikes:", len(unit_spike_times))
# print("first spikes:", unit_spike_times[:10])

# temp = np.memmap(
#     r"C:\Lab\Processing\TG963_site1-2\temp_wh.dat",
#     dtype="int16",
#     mode="r"
# ).reshape(-1, 385)

# snippets = []

# for t in unit_spike_times[:100]:
#     t = int(t)

#     snippets.append(
#         temp[t-30:t+60, 24]
#     )

# snippets = np.asarray(snippets)

# mean_wf = snippets.mean(axis=0)

# plt.figure(figsize=(10, 4))

# for wf in snippets[:30]:
#     plt.plot(wf, alpha=0.15)

# plt.plot(
#     mean_wf,
#     linewidth=3
# )

# plt.axhline(0)
# plt.title("Unit 70, temp_wh.dat, binary column 24")
# plt.show()

############################################################################

# analyzer = si.load_sorting_analyzer(
#     r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened_\sorting_analyzer"
# )

# unit_id = 70

# templates = (
#     analyzer
#     .get_extension("templates")
#     .get_data(operator="average")
# )

# unit_ids = analyzer.sorting.unit_ids
# unit_index = list(unit_ids).index(unit_id)

# template = templates[unit_index]

# print("template shape:", template.shape)

# ptp = np.ptp(template, axis=0)

# best_ch_ind = np.argmax(ptp)

# print("PTP best channel:", best_ch_ind)
# print("KS channel:", 24)

# plt.figure(figsize=(10, 4))
# plt.plot(template[:, 24], marker=".")
# plt.axhline(0)
# plt.title("Unit 70 - analyzer channel 24")
# plt.show()

# import spikeinterface.preprocessing as spre

# sorting70 = analyzer.sorting.select_units([70])

# recording_hp = spre.highpass_filter(
#     analyzer.recording,
#     freq_min=150,
#     ignore_low_freq_error=True
# )

# test_sp = si.estimate_sparsity(
#     sorting=sorting70,
#     recording=recording_hp,
#     num_spikes_for_sparsity=1000,
#     method="radius",
#     radius_um=100,
#     peak_sign="neg",
# )

# print(test_sp.unit_id_to_channel_ids[70])

########################################################################
# import numpy as np
# import pandas as pd

# ks_folder = r"C:\Lab\Processing\TG963_site1-2"

# # channel_map = np.load(
# #     rf"{ks_folder}\channel_map.npy"
# # ).squeeze()

# # channel_positions = np.load(
# #     rf"{ks_folder}\channel_positions.npy"
# # )

# # print("channel_map shape:", channel_map.shape)
# # print("channel_positions shape:", channel_positions.shape)

# # df = pd.DataFrame({
# #     "phy_index": np.arange(len(channel_map)),
# #     "channel_map": channel_map,
# #     "x_um": channel_positions[:, 0],
# #     "y_um": channel_positions[:, 1],
# # })

# # print(df.to_string(index=False))
# channel_map = np.load(
#     rf"{ks_folder}\channel_map.npy"
# ).squeeze()

# print(channel_map[:20])
# print(channel_map[-20:])
# print(channel_map.shape)

import inspect
import spikeinterface.full as si

print(inspect.signature(si.load_sorting_analyzer))

#######################################################################

# import spikeinterface as si

# sorting70 = analyzer.sorting.select_units([70])
# recording = analyzer.recording

# test_sparsity = si.estimate_sparsity(
#     sorting=sorting70,
#     recording=recording,
#     num_spikes_for_sparsity=100,
#     method="radius",
#     radius_um=100,
#     peak_sign="both",
# )

# print(test_sparsity.unit_id_to_channel_ids[70])

# test_sparsity_1000 = si.estimate_sparsity(
#     sorting=sorting70,
#     recording=recording,
#     num_spikes_for_sparsity=10000,
#     method="radius",
#     radius_um=100,
#     peak_sign="both",
# )

# print(test_sparsity_1000.unit_id_to_channel_ids[70])

# test_sparsity_neg = si.estimate_sparsity(
#     sorting=sorting70,
#     recording=recording,
#     num_spikes_for_sparsity=5000,
#     method="radius",
#     radius_um=100,
#     peak_sign="neg",
# )

# print(test_sparsity_neg.unit_id_to_channel_ids[70])


# print(analyzer.sparsity)
# print(analyzer.sparsity.unit_id_to_channel_ids[70])
# print(analyzer)
# print(analyzer.sparsity.unit_id_to_channel_indices[70])

#######################################################################

# unit_id = 70
# templates = analyzer.get_extension("templates").get_data()
# unit_index = analyzer.sorting.id_to_index(unit_id)
# template = templates[unit_index]

# for ch_id in [
#     "imec0.ap#AP24",
#     "imec0.ap#AP278",
#     "imec0.ap#AP283",
# ]:
#     idx = analyzer.channel_ids.tolist().index(ch_id)
#     wf = template[:, idx]

#     print(
#         ch_id,
#         "min =", wf.min(),
#         "max =", wf.max(),
#         "PTP =", np.ptp(wf)
#     )

# plt.figure()

# for ch_id in [
#     "imec0.ap#AP24",
#     "imec0.ap#AP278",
#     "imec0.ap#AP283",
# ]:
#     idx = analyzer.channel_ids.tolist().index(ch_id)

#     plt.plot(
#         template[:, idx],
#         label=ch_id
#     )

# plt.legend()
# plt.show()

# wf_ext = analyzer.get_extension("waveforms")
# wfs = wf_ext.get_waveforms_one_unit(70)

# print("wfs shape:", wfs.shape)

# plt.figure()

# for ch_id in [
#     "imec0.ap#AP24",
#     "imec0.ap#AP278",
#     "imec0.ap#AP283",
# ]:
#     idx = analyzer.channel_ids.tolist().index(ch_id)

#     plt.plot(
#         wfs[:, :, idx].mean(axis=0),
#         label=ch_id
#     )

# plt.legend()
# plt.show()


#########################################################################

# from spikeinterface.core.template_tools import get_template_extremum_channel

# extremum_channels = get_template_extremum_channel(
#     analyzer,
#     peak_sign="neg"
# )

# print(extremum_channels[70])

###############################################################

# print("\n========================================")
# print("Computing random spikes")
# print("========================================")

# analyzer.compute(
#     "random_spikes",
#     method="uniform",
#     max_spikes_per_unit=500,
#     seed=42,
# )

# analyzer.compute(
#     "waveforms",
#     ms_before=1.0,
#     ms_after=2.0,
# )

# print("\n========================================")
# print("Computing templates")
# print("========================================")

# analyzer.compute(
#     "templates",
#     operators=[
#         "average",
#         "median",
#         "std",
#     ],
# )

######################################################################3

# unit_id = 70
# si_times = analyzer.sorting.get_unit_spike_train(unit_id)
# spike_frame = si_times[100]

# fs = analyzer.sampling_frequency

# before = int(0.001 * fs)
# after = int(0.002 * fs)

# traces = analyzer.recording.get_traces(
#     start_frame=spike_frame - before,
#     end_frame=spike_frame + after,
#     return_scaled=True
# )

# ch24_idx = 24
# ch278_idx = 278

# import matplotlib.pyplot as plt

# t_ms = (
#     np.arange(traces.shape[0]) - before
# ) / fs * 1000

# plt.plot(
#     t_ms,
#     traces[:, ch24_idx],
#     label="channel 24"
# )

# plt.plot(
#     t_ms,
#     traces[:, ch278_idx],
#     label="channel 278"
# )

# plt.xlabel("Time (ms)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()

# print(analyzer.get_loaded_extension_names())
# wf_ext = analyzer.get_extension("waveforms")
# wfs70 = wf_ext.get_waveforms_one_unit(70)

# print(wfs70.shape)
# ch24_index = analyzer.channel_ids.tolist().index("imec0.ap#AP24")
# import matplotlib.pyplot as plt

# plt.figure()

# for i in range(min(100, wfs70.shape[0])):
#     plt.plot(
#         wfs70[i, :, ch24_index],
#         alpha=0.15
#     )

# plt.title("Stored SI waveforms: unit 70, AP24")
# plt.show()


################################################################
# unit_id = 70

# # ----------------------------
# # SpikeInterface spike times
# # ----------------------------
# si_times = analyzer.sorting.get_unit_spike_train(unit_id)

# print("SI unit 70 n spikes:", len(si_times))
# print("SI first 20:")
# print(si_times[:20])


# # ----------------------------
# # Phy / Kilosort spike times
# # ----------------------------
# ks_folder = r"C:\Lab\Processing\TG963_site1-2"

# phy_times = np.load(
#     rf"{ks_folder}\spike_times.npy"
# ).squeeze()

# phy_clusters = np.load(
#     rf"{ks_folder}\spike_clusters.npy"
# ).squeeze()

# phy70_times = phy_times[phy_clusters == 70]

# print("\nPhy cluster 70 n spikes:", len(phy70_times))
# print("Phy first 20:")
# print(phy70_times[:20])


# # ----------------------------
# # Compare
# # ----------------------------
# print("\nSame spike count:",
#       len(si_times) == len(phy70_times))

# print(
#     "Exactly same spike times:",
#     np.array_equal(si_times, phy70_times)
# )
###################################################################

# locations = analyzer.get_channel_locations()

# for i, ch_id in enumerate(analyzer.channel_ids):
#     print(
#         i,
#         ch_id,
#         locations[i]
#     )
########################################################################

# best_channel_id = "imec0.ap#AP278"

# best_idx = analyzer.channel_ids.tolist().index(best_channel_id)

# print("SI channel index:", best_idx)
# print("SI channel id:", analyzer.channel_ids[best_idx])
# print("SI location:", analyzer.get_channel_locations()[best_idx])


# print(analyzer.unit_ids)
# print(analyzer.recording)
# print("sampling rate:", analyzer.sampling_frequency)
# print("num channels:", analyzer.get_num_channels())
# print("channel ids:", analyzer.channel_ids)