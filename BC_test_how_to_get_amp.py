import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import spikeinterface.full as si
from pathlib import Path
from spikeinterface.core import get_template_extremum_channel

unit_id = 483

OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_filtered_raw"
)

ANALYZER_FOLDER = OUTPUT_DIR / "sorting_analyzer"

analyzer = si.load_sorting_analyzer(
    ANALYZER_FOLDER,
    load_extensions=True,
)

template = analyzer.get_extension("templates").get_data(
    operator="median"
)[unit_id]

max_pos = template.max()
max_neg = template.min()
ptp = template.max() - template.min()

print("max positive:", max_pos)
print("max negative:", max_neg)
print("peak-to-peak:", ptp)

best_ch = np.argmax(np.ptp(template, axis=0))

wave = template[:, best_ch]

print("best channel:", best_ch)
print("positive peak:", wave.max())
print("negative trough:", wave.min())
print("peak-to-peak:", np.ptp(wave))

amp_ext = analyzer.get_extension("spike_amplitudes")
amps = amp_ext.get_data()

spike_vector = analyzer.sorting.to_spike_vector()
unit_ids = analyzer.sorting.unit_ids

unit_index = list(unit_ids).index(unit_id)
unit_amps = amps[spike_vector["unit_index"] == unit_index]

print("n spikes:", len(unit_amps))
print("median:", np.median(unit_amps))
print("median abs:", np.median(np.abs(unit_amps)))
print("min:", np.min(unit_amps))
print("max:", np.max(unit_amps))
print("P10, P50, P90:", np.percentile(unit_amps, [10, 50, 90]))

print("median abs:", np.median(np.abs(unit_amps)))
import matplotlib.pyplot as plt

plt.hist(unit_amps, bins=100)
plt.axvline(np.median(unit_amps), linestyle="--")
plt.xlabel("Spike amplitude (uV)")
plt.ylabel("Count")
plt.title(f"Unit {unit_id}")
plt.show()

spike_vector = analyzer.sorting.to_spike_vector()

mask = spike_vector["unit_index"] == unit_index

spike_samples = spike_vector["sample_index"][mask]
spike_times = spike_samples / analyzer.sampling_frequency

plt.scatter(
    spike_times,
    unit_amps,
    s=2,
    alpha=0.3,
)

plt.xlabel("Time (s)")
plt.ylabel("Spike amplitude (uV)")
plt.title(f"Unit {unit_id}")
plt.show()

channel_ids = analyzer.channel_ids

print("channel IDs around 157:")
print(channel_ids)

# 找到 Phy channel 157 在 analyzer 数组里的位置
ch157_id = "imec0.ap#AP157"
ch157_index = list(channel_ids).index(ch157_id)

wave157 = template[:, ch157_index]

print("\nPhy best channel 157:")
print("positive peak:", wave157.max())
print("negative trough:", wave157.min())
print("peak-to-peak:", np.ptp(wave157))

from spikeinterface.core import get_template_extremum_channel

neg_ch = get_template_extremum_channel(
    analyzer,
    peak_sign="neg"
)

pos_ch = get_template_extremum_channel(
    analyzer,
    peak_sign="pos"
)

both_ch = get_template_extremum_channel(
    analyzer,
    peak_sign="both"
)

print("negative extremum channel:", neg_ch[unit_id])
print("positive extremum channel:", pos_ch[unit_id])
print("both extremum channel:", both_ch[unit_id])
print("Phy best channel: 157")

import numpy as np

unit_ids = list(analyzer.sorting.unit_ids)
unit_index = unit_ids.index(unit_id)

templates_ext = analyzer.get_extension("templates")
templates = templates_ext.get_data(operator="median")

print("unit_id:", unit_id)
print("unit_index:", unit_index)
print("templates shape:", templates.shape)

template_check = templates[unit_index]

print("single template shape:", template_check.shape)
print("analyzer n channels:", len(analyzer.channel_ids))

channel_ids = list(analyzer.channel_ids)

idx157 = channel_ids.index("imec0.ap#AP157")

w157 = template_check[:, idx157]

print("\nAP157")
print("positive:", np.max(w157))
print("negative:", np.min(w157))
print("PTP:", np.ptp(w157))
print("nonzero samples:", np.count_nonzero(w157))

idx186 = channel_ids.index("imec0.ap#AP186")

w186 = template_check[:, idx186]

print("\nAP186")
print("positive:", np.max(w186))
print("negative:", np.min(w186))
print("PTP:", np.ptp(w186))
print("nonzero samples:", np.count_nonzero(w186))

print("old template shape:", template.shape)
print("new template shape:", template_check.shape)
print("same?", np.array_equal(template, template_check))