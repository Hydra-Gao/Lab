import numpy as np
import spikeinterface as si
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA

analyzer = si.load_sorting_analyzer(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\sorting_analyzer"
)

# ============================================================
# SETTINGS
# ============================================================

UNIT_ID = 36
CHANNEL_IDS = [12, 14]
N_COMPONENTS = 5


# ============================================================
# LOAD EXISTING DATA
# ============================================================

waveforms_ext = analyzer.get_extension("waveforms")
random_ext = analyzer.get_extension("random_spikes")

some_waveforms = waveforms_ext.data["waveforms"]
some_spikes = random_ext.get_random_spikes()

sorting = analyzer.sorting

print("Waveforms shape:", some_waveforms.shape)
print("Random spikes:", some_spikes.shape)


# ============================================================
# HELPER: get one unit's sparse waveform slice exactly like SI
# ============================================================

def get_unit_sparse_waveforms(analyzer, unit_id):
    unit_index = analyzer.sorting.id_to_index(unit_id)

    spike_mask = some_spikes["unit_index"] == unit_index
    wfs = some_waveforms[spike_mask]

    sparse_channel_indices = (
        analyzer.sparsity.unit_id_to_channel_indices[unit_id]
    )

    sparse_channel_ids = (
        analyzer.sparsity.unit_id_to_channel_ids[unit_id]
    )

    # SI waveforms are internally stored aligned to this unit's sparse channels
    wfs = wfs[:, :, :len(sparse_channel_indices)]

    return (
        wfs,
        np.asarray(sparse_channel_indices),
        np.asarray(sparse_channel_ids),
        spike_mask,
    )


# ============================================================
# STRICT SI-LIKE FIT FOR ONE CHANNEL
# ============================================================

def fit_si_like_channel_pca(
    analyzer,
    channel_id,
    whiten,
    n_components=5,
):
    """
    Reproduce SI by_channel_local fitting for ONE physical channel.

    Important:
    The PCA model is fit using waveforms from ALL units whose sparsity
    contains this channel, exactly like SI's channel-local PCA.
    """

    pca = IncrementalPCA(
        n_components=n_components,
        whiten=whiten,
    )

    channel_index_global = analyzer.channel_ids_to_indices(
        [channel_id]
    )[0]

    contributing_units = []

    for unit_id in analyzer.unit_ids:

        (
            wfs,
            sparse_channel_indices,
            sparse_channel_ids,
            spike_mask,
        ) = get_unit_sparse_waveforms(analyzer, unit_id)

        # Does this unit contain this physical channel?
        matches = np.flatnonzero(
            sparse_channel_indices == channel_index_global
        )

        if len(matches) == 0:
            continue

        local_pos = matches[0]

        # shape: n_spikes x n_samples
        X = wfs[:, :, local_pos]

        if len(X) < n_components:
            continue

        pca.partial_fit(X)

        contributing_units.append(unit_id)

    return pca, contributing_units


# ============================================================
# PROJECT UNIT 36 USING THAT SI-LIKE PCA MODEL
# ============================================================

def project_unit_on_channel(
    analyzer,
    unit_id,
    channel_id,
    pca,
):
    (
        wfs,
        sparse_channel_indices,
        sparse_channel_ids,
        spike_mask,
    ) = get_unit_sparse_waveforms(analyzer, unit_id)

    channel_index_global = analyzer.channel_ids_to_indices(
        [channel_id]
    )[0]

    matches = np.flatnonzero(
        sparse_channel_indices == channel_index_global
    )

    if len(matches) == 0:
        raise ValueError(
            f"Unit {unit_id} does not contain channel {channel_id} "
            f"in its sparsity."
        )

    local_pos = matches[0]

    X = wfs[:, :, local_pos]

    proj = pca.transform(X)

    return proj


# ============================================================
# RUN BOTH whiten=True AND whiten=False
# ============================================================

results = {}

for whiten in [True, False]:

    print("\n" + "=" * 70)
    print(f"WHITEN = {whiten}")
    print("=" * 70)

    channel_results = {}

    for ch in CHANNEL_IDS:

        print(f"\nFitting SI-like PCA for channel {ch}...")

        pca, contributing_units = fit_si_like_channel_pca(
            analyzer,
            channel_id=ch,
            whiten=whiten,
            n_components=N_COMPONENTS,
        )

        print(
            f"Channel {ch}: "
            f"{len(contributing_units)} contributing units"
        )

        proj = project_unit_on_channel(
            analyzer,
            unit_id=UNIT_ID,
            channel_id=ch,
            pca=pca,
        )

        print(
            f"Projected unit {UNIT_ID} on channel {ch}: "
            f"{proj.shape}"
        )

        print(
            "Explained variance:",
            pca.explained_variance_
        )

        print(
            "Explained variance ratio:",
            pca.explained_variance_ratio_
        )

        channel_results[ch] = {
            "pca": pca,
            "projection": proj,
            "contributing_units": contributing_units,
        }

    results[whiten] = channel_results

print(pca.explained_variance_)

# ============================================================
# PLOT: compare TRUE vs FALSE
# ============================================================

for whiten in [True, False]:

    proj12 = results[whiten][12]["projection"]
    proj14 = results[whiten][14]["projection"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    # 12A vs 14A
    axes[0, 0].scatter(
        proj14[:, 0],
        proj12[:, 0],
        s=6,
        alpha=0.5,
    )
    axes[0, 0].set_xlabel("14A")
    axes[0, 0].set_ylabel("12A")

    # 12A vs 12B
    axes[0, 1].scatter(
        proj12[:, 1],
        proj12[:, 0],
        s=6,
        alpha=0.5,
    )
    axes[0, 1].set_xlabel("12B")
    axes[0, 1].set_ylabel("12A")

    # 14A vs 14B
    axes[1, 0].scatter(
        proj14[:, 1],
        proj14[:, 0],
        s=6,
        alpha=0.5,
    )
    axes[1, 0].set_xlabel("14B")
    axes[1, 0].set_ylabel("14A")

    # 12B vs 14B
    axes[1, 1].scatter(
        proj14[:, 1],
        proj12[:, 1],
        s=6,
        alpha=0.5,
    )
    axes[1, 1].set_xlabel("14B")
    axes[1, 1].set_ylabel("12B")

    fig.suptitle(
        f"Unit {UNIT_ID} | strict SI-like PCA | whiten={whiten}"
    )

    plt.tight_layout()
    plt.show()