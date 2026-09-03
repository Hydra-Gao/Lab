from pathlib import Path

import numpy as np
import pandas as pd

from spikeinterface.metrics.quality.pca_metrics import (
    mahalanobis_metrics,
    d_prime_metric,
    nearest_neighbors_metrics,
    simplified_silhouette_score,
)


KS_FOLDER = Path(r"C:\Lab\Processing\TG963_site1-2")

TARGET_UNIT_ID = 527

# Start conservatively
N_PCS = 2
N_REF_CHANNELS = 2
MAX_SPIKES_PER_UNIT = 500
RANDOM_SEED = 42


# ============================================================
# LOAD KILOSORT PCA DATA
# ============================================================

pc_features = np.load(
    KS_FOLDER / "pc_features.npy",
    mmap_mode="r",
)

pc_feature_ind = np.load(
    KS_FOLDER / "pc_feature_ind.npy",
)

spike_templates = np.load(
    KS_FOLDER / "spike_templates.npy"
).squeeze().astype(int)

spike_clusters = np.load(
    KS_FOLDER / "spike_clusters.npy"
).squeeze().astype(int)

templates = np.load(
    KS_FOLDER / "templates.npy",
    mmap_mode="r",
)


print("pc_features:", pc_features.shape)
print("pc_feature_ind:", pc_feature_ind.shape)
print("spike_templates:", spike_templates.shape)
print("spike_clusters:", spike_clusters.shape)
print("templates:", templates.shape)


# ============================================================
# HANDLE OPTIONAL FEATURE SUBSET
# ============================================================

feature_spike_ids_path = (
    KS_FOLDER / "pc_feature_spike_ids.npy"
)

if feature_spike_ids_path.exists():
    feature_spike_ids = np.load(
        feature_spike_ids_path
    ).astype(int)
else:
    feature_spike_ids = np.arange(
        pc_features.shape[0]
    )

assert len(feature_spike_ids) == pc_features.shape[0]

feature_clusters = spike_clusters[feature_spike_ids]
feature_templates = spike_templates[feature_spike_ids]


# ============================================================
# FIND TARGET'S DOMINANT TEMPLATE
# ============================================================

target_rows = np.flatnonzero(
    feature_clusters == TARGET_UNIT_ID
)

if len(target_rows) == 0:
    raise RuntimeError(
        f"No spikes found for unit {TARGET_UNIT_ID}"
    )

target_template_ids = feature_templates[target_rows]

unique_t, counts = np.unique(
    target_template_ids,
    return_counts=True,
)

target_template = unique_t[np.argmax(counts)]

print("Target unit:", TARGET_UNIT_ID)
print("Dominant template:", target_template)


# ============================================================
# CHOOSE TARGET REFERENCE CHANNELS
# ============================================================

# These are the channels on which KS actually stored
# PC features for this template.
candidate_channels = np.asarray(
    pc_feature_ind[target_template],
    dtype=int,
)

# Rank them using template PTP, similar in spirit
# to choosing the strongest channels.
template = np.asarray(
    templates[target_template]
)

ptp = np.ptp(template, axis=0)

order = np.argsort(
    ptp[candidate_channels]
)[::-1]

reference_channels = candidate_channels[
    order[:N_REF_CHANNELS]
]

print("Reference channels:", reference_channels)


# ============================================================
# BUILD COMMON KILOSORT PC FEATURE SPACE
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

all_units = np.unique(feature_clusters)

X_rows = []
y_rows = []


for unit_id in all_units:

    unit_feature_rows = np.flatnonzero(
        feature_clusters == unit_id
    )

    valid_rows = []

    # First require that the spike's template has
    # ALL chosen reference channels.
    #
    # This avoids treating "missing feature" as an
    # actual zero during metric computation.
    for row in unit_feature_rows:

        template_id = feature_templates[row]

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        if np.all(
            np.isin(reference_channels, local_channels)
        ):
            valid_rows.append(row)

    valid_rows = np.asarray(valid_rows, dtype=int)

    if len(valid_rows) == 0:
        continue

    # Fixed sampling like your SI pipeline
    if len(valid_rows) > MAX_SPIKES_PER_UNIT:
        valid_rows = rng.choice(
            valid_rows,
            size=MAX_SPIKES_PER_UNIT,
            replace=False,
        )

    for row in valid_rows:

        template_id = feature_templates[row]

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        # feature shape:
        # n_pcs × nearest_chans
        this_pc = np.asarray(
            pc_features[row, :N_PCS, :]
        )

        # Construct:
        # n_pcs × N_REF_CHANNELS
        dense = np.empty(
            (N_PCS, len(reference_channels)),
            dtype=np.float32,
        )

        for j, channel_id in enumerate(reference_channels):

            local_index = np.flatnonzero(
                local_channels == channel_id
            )[0]

            dense[:, j] = this_pc[:, local_index]

        X_rows.append(dense.reshape(-1))
        y_rows.append(unit_id)


print("\n=== RAW KS FEATURE CHECK ===")

print("Target dominant template:", target_template)

print(
    "pc_feature_ind[target_template]:",
    pc_feature_ind[target_template]
)

# 找 target unit 里属于 dominant template 的 spikes
target_dom_rows = target_rows[
    feature_templates[target_rows] == target_template
]

print(
    "Target spikes using dominant template:",
    len(target_dom_rows)
)

# 只拿前500个方便检查
check_rows = target_dom_rows[:500]

local_channels = np.asarray(
    pc_feature_ind[target_template],
    dtype=int
)

for local_idx, ch in enumerate(local_channels):

    vals_pc1 = np.asarray(
        pc_features[check_rows, 0, local_idx]
    )

    vals_pc2 = np.asarray(
        pc_features[check_rows, 1, local_idx]
    )

    print(
        f"\nchannel {ch}, local index {local_idx}"
    )

    print(
        "PC1 min/max/std:",
        vals_pc1.min(),
        vals_pc1.max(),
        vals_pc1.std()
    )

    print(
        "PC2 min/max/std:",
        vals_pc2.min(),
        vals_pc2.max(),
        vals_pc2.std()
    )



X = np.asarray(X_rows)
y = np.asarray(y_rows)


print()
print("Final PC matrix:", X.shape)
print("Labels:", y.shape)
print(
    "Target spikes:",
    np.sum(y == TARGET_UNIT_ID),
)
print(
    "Background spikes:",
    np.sum(y != TARGET_UNIT_ID),
)
print(
    "Neighbor units:",
    np.unique(y),
)


target_X = X[y == TARGET_UNIT_ID]

print("\n=== Mahalanobis sanity check ===")
print("target_X shape:", target_X.shape)

cov = np.cov(target_X, rowvar=False)

print("Covariance matrix:")
print(cov)

print("Covariance rank:", np.linalg.matrix_rank(cov))
print("Number of dimensions:", X.shape[1])

print("Condition number:", np.linalg.cond(cov))

print("Feature std:")
print(np.std(target_X, axis=0))

print("Correlation matrix:")
print(np.corrcoef(target_X, rowvar=False))

print("Any NaN in X:", np.isnan(X).any())
print("Any Inf in X:", np.isinf(X).any())

print("Any NaN in target:", np.isnan(target_X).any())
print("Any Inf in target:", np.isinf(target_X).any())

try:
    cov_inv = np.linalg.inv(cov)
    print("Covariance inversion: SUCCESS")
except np.linalg.LinAlgError as e:
    print("Covariance inversion FAILED:", e)


# ============================================================
# SI METRIC FUNCTIONS -- NO SORTING ANALYZER
# ============================================================

isolation_distance, l_ratio = mahalanobis_metrics(
    X,
    y,
    TARGET_UNIT_ID,
)

d_prime = d_prime_metric(
    X,
    y,
    TARGET_UNIT_ID,
)

nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(
    X,
    y,
    TARGET_UNIT_ID,
    max_spikes=10000,
    n_neighbors=5,
)

silhouette = simplified_silhouette_score(
    X,
    y,
    TARGET_UNIT_ID,
)


metrics = pd.Series({
    "isolation_distance": isolation_distance,
    "l_ratio": l_ratio,
    "d_prime": d_prime,
    "silhouette": silhouette,
    "nn_hit_rate": nn_hit_rate,
    "nn_miss_rate": nn_miss_rate,
})

print()
print(metrics)