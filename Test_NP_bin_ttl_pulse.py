from pathlib import Path
import numpy as np

bin_path = Path(
    r"C:\Lab\Processing\TG963_site1-2\original_data\Cb_2026_07_14_1site_2_g0_t0.imec0.ap.bin"
)

n_channels = 385
sampling_frequency = 30000.0

raw = np.memmap(
    bin_path,
    dtype=np.int16,
    mode="r",
)

if raw.size % n_channels != 0:
    raise ValueError(
        f"File size is not divisible by {n_channels} channels."
    )

raw = raw.reshape(-1, n_channels)

# Last saved channel: SY0
sync_word = raw[:, 384].astype(np.uint16)

print("Unique SY0 values:")
print(np.unique(sync_word)[:100])

sampling_frequency = 30000.0

for bit in [3, 6, 7]:
    state = ((sync_word >> bit) & 1).astype(np.int8)

    rising_samples = np.flatnonzero(
        np.diff(state, prepend=state[0]) == 1
    )

    falling_samples = np.flatnonzero(
        np.diff(state, prepend=state[0]) == -1
    )

    print(f"\nbit {bit}")
    print("high fraction:", state.mean())
    print("rising edges:", len(rising_samples))
    print("falling edges:", len(falling_samples))
    print("first rising times:", rising_samples[:60] / sampling_frequency)