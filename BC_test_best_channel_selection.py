from pathlib import Path
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

KS_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2"
)

UNIT_ID = 462
N_REF_CHANNELS = 2
TOP_N = 20


# ============================================================
# LOAD ONLY NECESSARY FILES
# ============================================================

print("Loading files...")

spike_clusters = np.load(
    KS_FOLDER / "spike_clusters.npy"
).squeeze().astype(int)

spike_templates = np.load(
    KS_FOLDER / "spike_templates.npy"
).squeeze().astype(int)

pc_feature_ind = np.load(
    KS_FOLDER / "pc_feature_ind.npy"
)

templates = np.load(
    KS_FOLDER / "templates.npy",
    mmap_mode="r",
)

whitening_mat_inv = np.load(
    KS_FOLDER / "whitening_mat_inv.npy"
)

print("Done.\n")


# ============================================================
# 1. FIND SPIKES BELONGING TO UNIT
# ============================================================

unit_rows = np.flatnonzero(
    spike_clusters == UNIT_ID
)

if len(unit_rows) == 0:
    raise RuntimeError(
        f"Unit {UNIT_ID} not found."
    )

print("=" * 70)
print(f"UNIT {UNIT_ID}")
print("=" * 70)

print("Number of spikes:", len(unit_rows))


# ============================================================
# 2. DOMINANT TEMPLATE
# ============================================================

templates_for_unit = spike_templates[
    unit_rows
]

template_ids, counts = np.unique(
    templates_for_unit,
    return_counts=True,
)

order = np.argsort(counts)[::-1]

print("\nTemplates used by this unit:")

for tid, count in zip(
    template_ids[order],
    counts[order],
):
    print(
        f"template {tid:4d}: "
        f"{count:8d} spikes "
        f"({count / len(unit_rows):.2%})"
    )

dominant_template = int(
    template_ids[np.argmax(counts)]
)

print(
    "\nDominant template:",
    dominant_template
)


# ============================================================
# 3. PC FEATURE SPARSE CHANNELS
# ============================================================

candidate_channels = np.asarray(
    pc_feature_ind[dominant_template],
    dtype=int,
)

print("\npc_feature_ind:")
print(candidate_channels)


# ============================================================
# 4. UNWHITEN TEMPLATE
# ============================================================

template_whitened = np.asarray(
    templates[dominant_template]
)

template_unwhitened = (
    template_whitened
    @ whitening_mat_inv
)

ptp = np.ptp(
    template_unwhitened,
    axis=0,
)


# ============================================================
# 5. FULL TEMPLATE CHANNEL RANKING
# ============================================================

full_order = np.argsort(
    ptp
)[::-1]

print(
    f"\nTop {TOP_N} channels "
    "from FULL unwhitened template:"
)

print(
    "\nrank   channel      PTP      "
    "in_pc_feature_ind"
)

for rank, ch in enumerate(
    full_order[:TOP_N],
    start=1,
):
    print(
        f"{rank:4d}   "
        f"{ch:7d}   "
        f"{ptp[ch]:9.4f}   "
        f"{ch in candidate_channels}"
    )


# ============================================================
# 6. CURRENT BC_01 SELECTION
# ============================================================

candidate_order = candidate_channels[
    np.argsort(
        ptp[candidate_channels]
    )[::-1]
]

reference_channels = candidate_order[
    :N_REF_CHANNELS
]

print(
    "\nChannels ranked ONLY within "
    "pc_feature_ind:"
)

for rank, ch in enumerate(
    candidate_order,
    start=1,
):
    print(
        f"{rank:4d}   "
        f"channel {ch:4d}   "
        f"PTP = {ptp[ch]:.4f}"
    )


print("\n" + "=" * 70)

print(
    "BC_01 reference channels:",
    reference_channels
)

print(
    "FULL-template top channels:",
    full_order[:N_REF_CHANNELS]
)

print("=" * 70)


# ============================================================
# 7. SPECIAL CHECK FOR YOUR OBSERVED CHANNELS
# ============================================================

channels_to_check = [88, 90, 92]

print("\nSpecific channels:")

for ch in channels_to_check:

    print(
        f"channel {ch}: "
        f"PTP={ptp[ch]:.4f}, "
        f"in_pc_feature_ind="
        f"{ch in candidate_channels}"
    )