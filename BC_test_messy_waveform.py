import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from pathlib import Path
from spikeinterface.core import get_template_extremum_channel

unit_id = 300

OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_filtered_raw"
)

ANALYZER_FOLDER = OUTPUT_DIR / "sorting_analyzer"

analyzer = si.load_sorting_analyzer(
    ANALYZER_FOLDER,
    load_extensions=True,
)

sorting = analyzer.sorting
recording = analyzer.recording

fs = recording.get_sampling_frequency()

# spike time，单位是 sample
spike_times = sorting.get_unit_spike_train(unit_id)
print("n spikes:", len(spike_times))
print("fs:", fs)

template_extremum_channel = get_template_extremum_channel(
    analyzer,
    peak_sign="both",
    mode="extremum",
    outputs="id",
)

main_channel_id = template_extremum_channel[unit_id]
main_channel_index = recording.ids_to_indices([main_channel_id])[0]

print("main channel ID:", main_channel_id)
print("main channel index:", main_channel_index)

ms_before = 1.0
ms_after = 1.5

nbefore = int(ms_before * fs / 1000)
nafter  = int(ms_after  * fs / 1000)

# 为了先诊断，不需要把 30115 个全取出来
rng = np.random.default_rng(0)

n_sample = min(3000, len(spike_times))
selected_spikes = rng.choice(
    spike_times,
    size=n_sample,
    replace=False
)
selected_spikes = np.sort(selected_spikes)

waveforms = []

num_samples = recording.get_num_samples()

for st in selected_spikes:

    start = int(st - nbefore)
    end   = int(st + nafter)

    if start < 0 or end >= num_samples:
        continue

    trace = recording.get_traces(
        start_frame=start,
        end_frame=end,
        channel_ids=[main_channel_id],
    )[:, 0]

    waveforms.append(trace)

waveforms = np.asarray(waveforms)

print(waveforms.shape)

plt.figure(figsize=(7, 5))

for w in waveforms[:300]:
    plt.plot(w, alpha=0.05)

plt.plot(
    np.median(waveforms, axis=0),
    linewidth=2,
    label="median waveform"
)

plt.axvline(nbefore, linestyle="--", label="Kilosort spike time")

plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

search_ms = 0.3
search_half_width = int(search_ms * fs / 1000)

center = nbefore

lo = center - search_half_width
hi = center + search_half_width + 1

trough_samples = (
    np.argmin(waveforms[:, lo:hi], axis=1)
    + lo
)

values, counts = np.unique(
    trough_samples,
    return_counts=True
)

plt.figure(figsize=(7, 4))

plt.bar(
    values - center,
    counts / counts.sum()
)

plt.xlabel("Trough offset from spike time (samples)")
plt.ylabel("Fraction of spikes")
plt.show()

offsets = trough_samples - center

median_offset = np.median(offsets)

mad = np.median(
    np.abs(offsets - median_offset)
)

sd = np.std(offsets)

frac_outside_1 = np.mean(
    np.abs(offsets - median_offset) > 1
)

frac_outside_2 = np.mean(
    np.abs(offsets - median_offset) > 2
)

frac_outside_3 = np.mean(
    np.abs(offsets - median_offset) > 3
)

print("median offset:", median_offset)
print("MAD:", mad)
print("SD:", sd)
print(">1 sample:", frac_outside_1)
print(">2 samples:", frac_outside_2)
print(">3 samples:", frac_outside_3)

# 定义正常和异常 waveform
deviation = np.abs(offsets - median_offset)

good_mask = deviation <= 1
bad_mask = deviation > 2

print("good:", good_mask.sum(), good_mask.mean())
print("bad :", bad_mask.sum(), bad_mask.mean())

median_waveform = np.median(waveforms, axis=0)

# good waveforms
plt.figure(figsize=(8, 5))

for w in waveforms[good_mask][:200]:
    plt.plot(w, alpha=0.05)

plt.plot(median_waveform, linewidth=2, label="median")
plt.axvline(center, linestyle="--")
plt.title("Aligned waveforms: trough within ±1 sample")
plt.legend()
plt.show()


# bad waveforms
plt.figure(figsize=(8, 5))

for w in waveforms[bad_mask][:200]:
    plt.plot(w, alpha=0.15)

plt.plot(median_waveform, linewidth=2, label="median")
plt.axvline(center, linestyle="--")
plt.title("Outlier waveforms: trough >2 samples away")
plt.legend()
plt.show()

template = np.median(waveforms, axis=0)

# baseline subtract
template = template - np.median(template[:10])

MAX_SHIFT = 8

best_shifts = np.zeros(len(waveforms), dtype=int)
best_corrs = np.zeros(len(waveforms))

for i, w in enumerate(waveforms):

    w = w - np.median(w[:10])

    best_corr = -np.inf
    best_shift = 0

    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):

        if shift < 0:
            x = w[-shift:]
            y = template[:len(template) + shift]

        elif shift > 0:
            x = w[:-shift]
            y = template[shift:]

        else:
            x = w
            y = template

        # correlation
        if np.std(x) == 0 or np.std(y) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(x, y)[0, 1]

        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    best_shifts[i] = best_shift
    best_corrs[i] = best_corr

values, counts = np.unique(best_shifts, return_counts=True)

plt.figure(figsize=(7, 4))
plt.bar(values, counts / counts.sum())
plt.xlabel("Best alignment shift (samples)")
plt.ylabel("Fraction of spikes")
plt.show()

print("median best shift:", np.median(best_shifts))
print(
    "fraction |shift| > 1:",
    np.mean(np.abs(best_shifts) > 1)
)
print(
    "fraction |shift| > 2:",
    np.mean(np.abs(best_shifts) > 2)
)

print("median best correlation:", np.median(best_corrs))
print(
    "fraction corr < 0.8:",
    np.mean(best_corrs < 0.8)
)