from pathlib import Path
import spikeinterface.full as si
import probeinterface as pi

print("Loading...")

folder = Path(r"C:\Users\15018\Desktop\Data\Neuralynx\2026-04-27_18-54-07") 

recording = si.read_neuralynx(folder)

print(recording)
print("Channel IDs:", recording.get_channel_ids())
print("Sampling frequency:", recording.get_sampling_frequency())
print("Duration:", recording.get_total_duration())

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

site_order_tip_to_base = list(range(32, 0, -1))
ad_order_tip_to_base = [site_to_ad[site] for site in site_order_tip_to_base]

print("Site order tip to base:", site_order_tip_to_base)
print("AD order tip to base:", ad_order_tip_to_base)

# Check channel id type first
channel_ids = recording.get_channel_ids()
print("Original channel IDs:", channel_ids)
#print("Channel ID type:", type(channel_ids[0]))

# Convert AD order to the same type as recording channel IDs
if isinstance(channel_ids[0], str):
    ad_order_tip_to_base_for_si = [str(ch) for ch in ad_order_tip_to_base]
else:
    ad_order_tip_to_base_for_si = ad_order_tip_to_base

# Reorder channels according to physical probe order
recording_ordered = recording.select_channels(ad_order_tip_to_base_for_si)

print("Ordered channel IDs:", recording_ordered.get_channel_ids())

probe = pi.generate_linear_probe(
    num_elec=32,
    ypitch=50
)

probe.set_device_channel_indices(range(32))

recording_ordered = recording_ordered.set_probe(probe)

#print("Channel locations:")
#print(recording_ordered.get_channel_locations())

recording_f = si.bandpass_filter(
    recording_ordered,
    freq_min=10,
    freq_max=5000
)

recording_cmr = si.common_reference(
    recording_f,
    reference="global",
    operator="median"
)

recording_saved = recording_cmr.save(
    folder="preprocessed_M12",
    format="binary",
    overwrite=True
)

