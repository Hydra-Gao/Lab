from pathlib import Path
import numpy as np

KS_FOLDER = Path(r"C:\Lab\Processing\TG963_site1-2")

# 你想检查的 final units
UNIT_IDS = [42]

pc_features = np.load(KS_FOLDER / "pc_features.npy")
pc_feature_ind = np.load(KS_FOLDER / "pc_feature_ind.npy")
spike_clusters = np.load(KS_FOLDER / "spike_clusters.npy").squeeze()

print("pc_features shape:", pc_features.shape)
print("pc_feature_ind shape:", pc_feature_ind.shape)
print("spike_clusters shape:", spike_clusters.shape)


for unit_id in UNIT_IDS:

    rows = np.flatnonzero(spike_clusters == unit_id)

    if len(rows) == 0:
        print(f"\nUnit {unit_id}: no spikes found")
        continue

    X = pc_features[rows]     # shape: n_unit_spikes × 10 × 6
    chans = pc_feature_ind[unit_id]

    print("\n" + "=" * 100)
    print(f"UNIT {unit_id}")
    print(f"n_spikes = {len(rows)}")
    print(f"pc_feature_ind = {chans}")
    print("=" * 100)

    for j, ch in enumerate(chans):

        vals = X[:, :, j]     # n_spikes × 6 PCs

        zero_fraction_each_pc = np.mean(vals == 0, axis=0)

        pc12_zero_fraction = np.mean(
            np.all(vals[:, :2] == 0, axis=1)
        )

        all6_zero_fraction = np.mean(
            np.all(vals == 0, axis=1)
        )

        means = np.mean(vals, axis=0)
        stds = np.std(vals, axis=0)
        mins = np.min(vals, axis=0)
        maxs = np.max(vals, axis=0)

        print(f"\nChannel {int(ch)}")
        print("-" * 80)

        for pc in range(vals.shape[1]):
            print(
                f"PC{pc+1}: "
                f"zero={zero_fraction_each_pc[pc]:7.3%}  "
                f"mean={means[pc]:10.5f}  "
                f"std={stds[pc]:10.5f}  "
                f"min={mins[pc]:10.5f}  "
                f"max={maxs[pc]:10.5f}"
            )

        print(
            f"PC1+PC2 jointly zero: "
            f"{pc12_zero_fraction:7.3%}"
        )

        print(
            f"PC1-PC6 all jointly zero: "
            f"{all6_zero_fraction:7.3%}"
        )