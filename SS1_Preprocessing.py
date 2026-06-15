from pathlib import Path
import spikeinterface.full as si
import probeinterface as pi
from config_local import RAW_DATA
import numpy as np

print("Loading...")

bird = "TG915"
date = "2026-05-27_21-19-20"

folder = RAW_DATA / bird / date

#folder = Path(r"C:\Users\15018\Desktop\Data\Neuralynx\2026-04-27_18-54-07") 

REFERENCE_CSC_CHANNEL = 15   # change this only

recording = si.read_neuralynx(folder)

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

if REFERENCE_CSC_CHANNEL not in site_to_ad:
    raise ValueError(
        f"REFERENCE_CSC_CHANNEL={REFERENCE_CSC_CHANNEL} is invalid. "
        f"Must be one of {list(site_to_ad.keys())}."
    )

REFERENCE_AD_CHANNEL = site_to_ad[REFERENCE_CSC_CHANNEL]

print(f"Reference CSC/site channel: {REFERENCE_CSC_CHANNEL}")
print(f"Converted reference AD channel: {REFERENCE_AD_CHANNEL}")

site_order_tip_to_base = list(range(32, 0, -1))
ad_order_tip_to_base = [site_to_ad[site] for site in site_order_tip_to_base]

print("Site order tip to base:", site_order_tip_to_base)
print("AD order tip to base:", ad_order_tip_to_base)

# Check channel id type first
channel_ids = recording.get_channel_ids()
print("Original channel IDs:", channel_ids)
print("Channel ID type:", type(channel_ids[0]))

# Convert AD order to the same type as recording channel IDs
if isinstance(channel_ids[0], str):
    ad_order_tip_to_base_for_si = [str(ch) for ch in ad_order_tip_to_base]
else:
    ad_order_tip_to_base_for_si = ad_order_tip_to_base

# # Reorder channels according to physical probe order
# recording_ordered = recording.select_channels(ad_order_tip_to_base_for_si)

# print("Ordered channel IDs:", recording_ordered.get_channel_ids())

# probe = pi.generate_linear_probe(
#     num_elec=32,
#     ypitch=50
# )

# probe.set_device_channel_indices(range(32))

# recording_ordered = recording_ordered.set_probe(probe)

# Reorder channels according to physical probe order: tip -> base
recording_ordered_32 = recording.select_channels(ad_order_tip_to_base_for_si)

ordered_ids_32 = list(recording_ordered_32.get_channel_ids())

print("Ordered 32-channel IDs:", ordered_ids_32)

# Convert reference channel id to the same type as SpikeInterface channel IDs
if isinstance(ordered_ids_32[0], str):
    reference_channel_id = str(REFERENCE_AD_CHANNEL)
else:
    reference_channel_id = REFERENCE_AD_CHANNEL

if reference_channel_id not in ordered_ids_32:
    raise ValueError(
        f"Reference channel {reference_channel_id} not found in ordered channel IDs. "
        f"Available IDs: {ordered_ids_32}"
    )

# Find where the reference channel sits in the ordered physical probe
reference_position = ordered_ids_32.index(reference_channel_id)

print(f"Reference channel ID: {reference_channel_id}")
print(f"Reference position in ordered probe: {reference_position}")

# Original 32-site geometry.
# Do NOT regenerate a compressed 31-channel probe after deleting reference.
locations_32 = np.column_stack([
    np.zeros(32),              # x coordinate
    np.arange(32) * 50.0       # y coordinate, tip -> base, 50 um spacing
])

# Keep all channels except the reference channel
keep_ids = [
    ch for ch in ordered_ids_32
    if ch != reference_channel_id
]

keep_positions = [
    i for i, ch in enumerate(ordered_ids_32)
    if ch != reference_channel_id
]

recording_ordered = recording_ordered_32.select_channels(keep_ids)

# Keep the real original locations, including the physical gap
locations_31 = locations_32[keep_positions]

print("Kept 31-channel IDs:", list(recording_ordered.get_channel_ids()))
print("Kept channel locations:")
print(locations_31)

# Build a 31-contact probe using the original 32-site coordinates minus reference
probe = pi.Probe(ndim=2, si_units="um")
probe.set_contacts(
    positions=locations_31,
    shapes="circle",
    shape_params={"radius": 5},
)

probe.set_device_channel_indices(np.arange(len(keep_ids)))

recording_ordered = recording_ordered.set_probe(probe)

print("Channel locations:")
print(recording_ordered.get_channel_locations())

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

