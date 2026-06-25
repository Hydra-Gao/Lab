from pathlib import Path
import spikeinterface.full as si
import probeinterface as pi
import numpy as np
from config_local import RAW_DATA
from spikeinterface.core import BinaryFolderRecording

print("Loading...")

bird = "TG964"
date = "split_experiments/experiment_1"

folder = RAW_DATA / bird / date 

#folder = Path(r"C:\Users\15018\Desktop\Data\Neuralynx\2026-04-27_18-54-07") 

# recording = si.read_neuralynx(folder)
recording = BinaryFolderRecording(folder)

# If the reference was physical site 2, then site_to_ad says AD channel = 16.
REFERENCE_SITE = 10   # <-- CHANGE THIS to your actual reference site

info_lines = []

info_lines.append(str(recording))
info_lines.append("")

# full segment info
info_lines.append(f"Number of segments: {recording.get_num_segments()}")

for seg_idx in range(recording.get_num_segments()):

    n_samples = recording.get_num_samples(segment_index=seg_idx)

    duration_sec = (
        n_samples / recording.get_sampling_frequency()
    )

    info_lines.append(
        f"Segment {seg_idx}: "
        f"samples={n_samples}, "
        f"duration_sec={duration_sec:.2f}"
    )

info_lines.append("")
info_lines.append(f"Channel IDs: {recording.get_channel_ids()}")
info_lines.append(f"Sampling frequency: {recording.get_sampling_frequency()}")
info_lines.append(f"Total duration: {recording.get_total_duration():.2f} sec")

for line in info_lines:
    print(line)

from config_local import WORKING_DIR

info_path = WORKING_DIR / "recording_info_M12.txt"

with open(info_path, "w", encoding="utf-8") as f:
    for line in info_lines:
        f.write(line + "\n")

print(f"Saved recording info to: {info_path}")

site_to_ad = {
    1: 8,  2: 16, 3: 7,  4: 31,
    5: 9,  6: 17, 7: 6,  8: 30,
    9: 10, 10: 18, 11: 5, 12: 29,
    13: 11, 14: 19, 15: 4, 16: 28,
    17: 12, 18: 20, 19: 3, 20: 27,
    21: 13, 22: 21, 23: 2, 24: 26,
    25: 14, 26: 22, 27: 1, 28: 25,
    29: 15, 30: 23, 31: 0, 32: 24,
}

# site_order_tip_to_base = list(range(32, 0, -1))
# ad_order_tip_to_base = [site_to_ad[site] for site in site_order_tip_to_base]

# ============================================================
# Remove the hardware reference channel BEFORE filtering / CMR
# ============================================================

if REFERENCE_SITE not in site_to_ad:
    raise ValueError(f"REFERENCE_SITE {REFERENCE_SITE} is not in site_to_ad map")

REFERENCE_AD_CHANNEL = site_to_ad[REFERENCE_SITE]

print(f"Removing reference physical site: {REFERENCE_SITE}")
print(f"Removing reference AD channel: {REFERENCE_AD_CHANNEL}")

# Keep all physical sites except the reference site
site_order_tip_to_base = [
    site for site in range(32, 0, -1)
    if site != REFERENCE_SITE
]

ad_order_tip_to_base = [site_to_ad[site] for site in site_order_tip_to_base]

print("Site order tip to base, reference removed:", site_order_tip_to_base)
print("AD order tip to base, reference removed:", ad_order_tip_to_base)

# Check channel id type first
channel_ids = recording.get_channel_ids()
print("Original channel IDs:", channel_ids)
print("Channel ID type:", type(channel_ids[0]))

# Convert AD order to the same type as recording channel IDs
if isinstance(channel_ids[0], str):
    ad_order_tip_to_base_for_si = [str(ch) for ch in ad_order_tip_to_base]
else:
    ad_order_tip_to_base_for_si = ad_order_tip_to_base

# Safety check
missing_channels = set(ad_order_tip_to_base_for_si) - set(channel_ids)
if len(missing_channels) > 0:
    raise ValueError(f"These channels are missing from recording: {missing_channels}")

# Reorder channels according to physical probe order, excluding reference channel
recording_ordered = recording.select_channels(ad_order_tip_to_base_for_si)

print("Ordered channel IDs, reference removed:", recording_ordered.get_channel_ids())
print("Number of channels after removing reference:", recording_ordered.get_num_channels())

# Build probe geometry for only the remaining channels.
# This preserves the real physical spacing, including the gap where the reference site was removed.
contact_positions = np.array([
    [0, (32 - site) * 50]   # site 32 at tip position, then 50 um spacing
    for site in site_order_tip_to_base
], dtype=float)

probe = pi.Probe(ndim=2, si_units="um")
probe.set_contacts(
    positions=contact_positions,
    shapes="circle",
    shape_params={"radius": 6},
)

probe.set_device_channel_indices(range(len(site_order_tip_to_base)))

recording_ordered = recording_ordered.set_probe(probe)

print("Site order tip to base:", site_order_tip_to_base)
print("AD order tip to base:", ad_order_tip_to_base)

# # Check channel id type first
# channel_ids = recording.get_channel_ids()
# print("Original channel IDs:", channel_ids)
# print("Channel ID type:", type(channel_ids[0]))

# # Convert AD order to the same type as recording channel IDs
# if isinstance(channel_ids[0], str):
#     ad_order_tip_to_base_for_si = [str(ch) for ch in ad_order_tip_to_base]
# else:
#     ad_order_tip_to_base_for_si = ad_order_tip_to_base

# # Reorder channels according to physical probe order
# recording_ordered = recording.select_channels(ad_order_tip_to_base_for_si)

# print("Ordered channel IDs:", recording_ordered.get_channel_ids())

# probe = pi.generate_linear_probe(
#     num_elec=32,
#     ypitch=50
# )

# probe.set_device_channel_indices(range(32))

# recording_ordered = recording_ordered.set_probe(probe)

# #print("Channel locations:")
# #print(recording_ordered.get_channel_locations())

recording_f = si.bandpass_filter(
    recording_ordered,
    freq_min=10,
    freq_max=5000,
    ignore_low_freq_error=True
)

recording_cmr = si.common_reference(
    recording_f,
    reference="global",
    operator="median",
)

from config_local import WORKING_DIR

preprocessed_folder = WORKING_DIR / "preprocessed_M12"

recording_saved = recording_cmr.save(
    folder=preprocessed_folder,
    format="binary",
    overwrite=True,
    chunk_duration="20s",
)

