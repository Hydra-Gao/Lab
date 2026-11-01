from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si


ANALYZER_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened\sorting_analyzer"
)

UNIT_ID = 84   # 改成你要检查的 unit


# ============================================================
# Load saved SortingAnalyzer
# ============================================================

analyzer = si.load_sorting_analyzer(
    ANALYZER_FOLDER,
    load_extensions=True,
)

print(analyzer)


# ============================================================
# Get the exact average template stored by SI
# ============================================================

templates_ext = analyzer.get_extension("templates")

average_templates = templates_ext.get_data(operator="average")

unit_ids = np.asarray(analyzer.unit_ids)
unit_index = np.where(unit_ids == UNIT_ID)[0][0]

template = average_templates[unit_index]

print("unit_id:", UNIT_ID)
print("template shape:", template.shape)
# shape = (n_samples, n_channels)


# ============================================================
# Find channel with largest absolute extremum
# ============================================================

unit_mask = analyzer.sparsity.mask[unit_index]

sparse_channel_indices = np.where(unit_mask)[0]

sparse_channel_ids = np.asarray(
    analyzer.channel_ids
)[sparse_channel_indices]


# Only examine channels belonging to this unit's sparse footprint
template_sparse = template[:, unit_mask]

channel_ptp_sparse = np.ptp(
    template_sparse,
    axis=0
)

best_sparse_index = np.argmax(
    channel_ptp_sparse
)

best_global_index = sparse_channel_indices[
    best_sparse_index
]

best_channel_id = analyzer.channel_ids[
    best_global_index
]

waveform = template[:, best_global_index]


print("Number of sparse channels:", len(sparse_channel_indices))
print("Sparse channel IDs:", sparse_channel_ids)

print("Best index within sparse footprint:",
      best_sparse_index)

print("Best global channel index:",
      best_global_index)

print("Best physical channel ID:",
      best_channel_id)

print(
    "minimum:", waveform.min(),
    "at sample:", np.argmin(waveform)
)

print(
    "maximum:", waveform.max(),
    "at sample:", np.argmax(waveform)
)


# ============================================================
# Plot main channel waveform
# ============================================================

fs = analyzer.sampling_frequency

n_samples = waveform.size

# your original extraction was:
# ms_before = 1.0
# ms_after = 2.0
#
# make spike time = 0 ms
time_ms = (
    np.arange(n_samples) / fs * 1000
    - 1.0
)

plt.figure(figsize=(9, 5))
plt.plot(time_ms, waveform, marker="o", markersize=3)

plt.axhline(0, linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)

plt.scatter(
    time_ms[np.argmin(waveform)],
    waveform.min(),
    s=80,
    label="minimum"
)

plt.scatter(
    time_ms[np.argmax(waveform)],
    waveform.max(),
    s=80,
    label="maximum"
)

plt.xlabel("Time relative to spike (ms)")
plt.ylabel("Amplitude (temp_wh units)")
plt.title(
    f"Unit {UNIT_ID} — SI average template\n"
    f"channel {best_channel_id}"
)

plt.legend()
plt.tight_layout()
plt.show()