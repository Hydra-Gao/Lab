import numpy as np
from pathlib import Path
import bombcell as bc
import bombcell.quality_metrics as qm

ks_dir = Path(r"C:\Lab\Processing\TG963_site1-2")

pc_features = np.load(
    ks_dir / "pc_features.npy",
    mmap_mode="r"
)

pc_features_idx = np.load(
    ks_dir / "pc_feature_ind.npy"
)

spike_clusters = np.load(
    ks_dir / "spike_clusters.npy"
).squeeze()

param = bc.get_default_parameters(
    str(ks_dir),
    raw_file=None,
    meta_file=None,
    kilosort_version=4,
)

param["computeDistanceMetrics"] = 1


for this_unit in range(250, 350):

    # 必须同时存在于 spike_clusters，
    # 并且不能超出 pc_feature_ind 行数
    if this_unit not in spike_clusters:
        continue

    if this_unit >= pc_features_idx.shape[0]:
        continue

    print("Testing unit", this_unit)

    try:

        iso, lratio, silhouette = qm.get_distance_metrics(
            pc_features,
            pc_features_idx,
            this_unit,
            spike_clusters,
            param,
        )

        print("  OK")

    except Exception as e:

        print("\n==========================")
        print("FAILED UNIT:", this_unit)
        print("ERROR:", repr(e))
        print("==========================")

        break