from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# SETTINGS
# ============================================================

KS_FOLDER = Path(r"C:\Lab\Processing\TG963_site1-2")

# Unit currently selected in Phy.
TARGET_UNIT = 462

# IMPORTANT:
# To compare 1:1 with an existing Phy screenshot, put EXACTLY the same
# channel IDs shown in that FeatureView here.
CHANNEL_IDS = [90, 92]

# Phy FeatureMixin defaults.
N_SPIKES_FEATURES = 10000
N_SPIKES_FEATURES_BACKGROUND = 10000

# The screenshot uses PC A / B = first two PCs.
N_PCS_DISPLAY = 2

# Used only if sample_rate cannot be parsed from params.py.
FALLBACK_SAMPLE_RATE = 30000.0

# Output.
OUTPUT_FOLDER = KS_FOLDER / "phy_featureview_clone"
SAVE_PNG = True
SAVE_PDF = True
SAVE_NPZ = True

# Plot cosmetics only; they do NOT change the selected spikes or feature values.
TARGET_SIZE = 4.0
BACKGROUND_SIZE = 5.0
TARGET_ALPHA = 0.90
BACKGROUND_ALPHA = 0.40
FIGSIZE = (16, 11)

# Set to e.g. (-10, 10) if you want fixed Phy-like feature axes.
# None = matplotlib autoscale.
FEATURE_AXIS_LIMITS = (-10, 10)


# ============================================================
# PHY SOURCE LOGIC REIMPLEMENTED BELOW
# ============================================================
#
# Relevant Phy/Phylib source behavior:
#
# 1) FeatureMixin._get_feature_view_spike_ids(cluster_id=None):
#       spike_ids = self.get_background_spike_ids(
#           self.n_spikes_features_background
#       )
#
# 2) BaseController.get_background_spike_ids(n):
#       ns = len(self.model.spike_times)
#       k = max(1, ns // n) if n is not None else 1
#       return np.arange(0, ns, k)
#
# 3) TemplateModel loads pc_features.npy then transposes:
#       KS:     spikes x PCs x local_channels
#       phylib: spikes x local_channels x PCs
#
# 4) TemplateModel.get_features(): sparse channel columns for each spike
#    are determined from its ORIGINAL spike_template, not spike_cluster.
#
# 5) phylib.from_sparse() creates a zero-filled dense output for the
#    requested channels. If a requested channel is absent from that spike's
#    sparse template feature set, its feature value remains exactly zero.
#
# This script intentionally reproduces that visualization behavior and does
# NOT use the metric-space rule "require every requested channel to exist".
# ============================================================


def load_sample_rate(folder: Path) -> float:
    """Parse sample_rate from params.py the same dataset Phy opens."""
    params_path = folder / "params.py"
    if params_path.exists():
        text = params_path.read_text(errors="ignore")
        match = re.search(
            r"^\s*sample_rate\s*=\s*([0-9.eE+-]+)",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return float(match.group(1))
    return float(FALLBACK_SAMPLE_RATE)


def phy_background_spike_ids(n_spikes_total: int, n=None):
    """
    Reproduce BaseController.get_background_spike_ids().

    This is global: no channel filtering and no cluster filtering occurs here.
    Because spike IDs are time ordered, this is approximately uniform in time.
    """
    k = max(1, n_spikes_total // n) if n is not None else 1
    return np.arange(0, n_spikes_total, k, dtype=np.int64)


def evenly_pick_from_sorted_ids(spike_ids, n):
    """
    For selected-cluster display only.

    Phy delegates selected-cluster sampling to SpikeSelector. For the common
    case len(cluster)<=2500, this returns every spike exactly. If there are
    more than 2500 spikes, use an even-in-order subset so the visualization
    stays deterministic and time distributed.
    """
    spike_ids = np.asarray(spike_ids, dtype=np.int64)
    if n is None or len(spike_ids) <= n:
        return spike_ids
    if n <= 0:
        return np.array([], dtype=np.int64)
    idx = np.linspace(0, len(spike_ids) - 1, n, dtype=np.int64)
    return spike_ids[idx]


def load_feature_row_mapping(folder, n_total_spikes, n_feature_rows):
    """
    Return:
        feature_spike_ids: feature_row -> original spike ID
        spike_to_feature_row: original spike ID -> feature row, -1 if unavailable

    Phy calls this concept features_rows when pc_feature_spike_ids.npy exists.
    """
    path = folder / "pc_feature_spike_ids.npy"

    if path.exists():
        feature_spike_ids = np.load(path).squeeze().astype(np.int64)
        if len(feature_spike_ids) != n_feature_rows:
            raise RuntimeError(
                "pc_feature_spike_ids.npy length does not match pc_features rows"
            )
    else:
        if n_feature_rows != n_total_spikes:
            raise RuntimeError(
                "pc_features rows != total spikes but pc_feature_spike_ids.npy is absent"
            )
        feature_spike_ids = np.arange(n_total_spikes, dtype=np.int64)

    spike_to_feature_row = np.full(n_total_spikes, -1, dtype=np.int64)
    spike_to_feature_row[feature_spike_ids] = np.arange(
        len(feature_spike_ids), dtype=np.int64
    )

    return feature_spike_ids, spike_to_feature_row


def intersect_with_feature_rows(spike_ids, spike_to_feature_row):
    """
    Match FeatureMixin behavior that removes spike IDs not represented in
    model.features_rows when pc_feature_spike_ids.npy is present.
    """
    spike_ids = np.asarray(spike_ids, dtype=np.int64)
    return spike_ids[spike_to_feature_row[spike_ids] >= 0]


def phy_from_sparse_features(
    spike_ids,
    requested_channel_ids,
    pc_features_ks,
    pc_feature_ind,
    spike_templates,
    spike_to_feature_row,
    n_pcs,
):
    """
    Reproduce the essential TemplateModel.get_features() + from_sparse() path.

    Input KS representation:
        pc_features_ks.shape = (feature_rows, PCs, local_channels)

    Phylib internally transposes to:
        (feature_rows, local_channels, PCs)

    For each requested channel:
      - if present in pc_feature_ind[spike_template], copy real feature values
      - if absent, leave zeros

    Returns:
        dense: n_spikes x n_requested_channels x n_pcs
        present: n_spikes x n_requested_channels bool
    """
    spike_ids = np.asarray(spike_ids, dtype=np.int64)
    requested_channel_ids = np.asarray(requested_channel_ids, dtype=np.int64)

    dense = np.zeros(
        (len(spike_ids), len(requested_channel_ids), n_pcs),
        dtype=np.float32,
    )
    present = np.zeros(
        (len(spike_ids), len(requested_channel_ids)),
        dtype=bool,
    )

    # Group by original Kilosort template, because sparse channel columns are
    # template-specific and pc_feature_ind is indexed by template.
    these_templates = spike_templates[spike_ids]

    for template_id in np.unique(these_templates):
        mask = these_templates == template_id
        ids_t = spike_ids[mask]
        rows_t = spike_to_feature_row[ids_t]

        if np.any(rows_t < 0):
            raise RuntimeError("Encountered spike without a pc_features row")

        local_channels = np.asarray(pc_feature_ind[template_id], dtype=np.int64)

        # Raw KS block is spikes x PCs x local_channels.
        block_ks = np.asarray(
            pc_features_ks[rows_t, :n_pcs, :],
            dtype=np.float32,
        )

        # Phylib's internal orientation: spikes x local_channels x PCs.
        block = np.transpose(block_ks, (0, 2, 1))

        output_rows = np.flatnonzero(mask)

        for requested_index, channel_id in enumerate(requested_channel_ids):
            loc = np.flatnonzero(local_channels == channel_id)

            if len(loc) == 0:
                # Exact from_sparse consequence: zero-filled requested column.
                continue

            if len(loc) != 1:
                raise RuntimeError(
                    f"Template {template_id}: channel {channel_id} occurs {len(loc)} times"
                )

            dense[output_rows, requested_index, :] = block[:, int(loc[0]), :]
            present[output_rows, requested_index] = True

    # FeatureMixin._get_spike_features() additionally replaces NaNs by zero.
    dense[np.isnan(dense)] = 0

    return dense, present


def make_feature_names(channel_ids, n_pcs):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names = []
    pairs = []

    # Phy-like ordering for a 2-channel x 2-PC display:
    # Ch24 A, Ch26 A, Ch24 B, Ch26 B
    for pc in range(n_pcs):
        suffix = letters[pc] if pc < len(letters) else f"PC{pc + 1}"
        for ch_i, ch in enumerate(channel_ids):
            names.append(f"{ch}{suffix}")
            pairs.append((ch_i, pc))

    return names, pairs


def flatten_phy_features(dense, channel_ids, n_pcs):
    """
    Convert n_spikes x channels x PCs to the visual dimension ordering:
        ChA-PC1, ChB-PC1, ChA-PC2, ChB-PC2, ...
    """
    columns = []
    for pc in range(n_pcs):
        for ch_i in range(len(channel_ids)):
            columns.append(dense[:, ch_i, pc])
    return np.column_stack(columns)


def add_scatter(ax, x_bg, y_bg, x_target, y_target):
    ax.scatter(
        x_bg,
        y_bg,
        s=BACKGROUND_SIZE,
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
        c="0.65",
        rasterized=True,
    )
    ax.scatter(
        x_target,
        y_target,
        s=TARGET_SIZE,
        alpha=TARGET_ALPHA,
        linewidths=0,
        c="#008cff",
        rasterized=True,
    )

    ax.axhline(0, linewidth=0.6, c="0.5")
    ax.axvline(0, linewidth=0.6, c="0.5")

    if FEATURE_AXIS_LIMITS is not None:
        ax.set_xlim(*FEATURE_AXIS_LIMITS)
        ax.set_ylim(*FEATURE_AXIS_LIMITS)


def make_phy_style_matrix(
    target_X,
    background_X,
    target_times,
    background_times,
    feature_names,
):
    """
    Reproduce the visual organization in Phy's FeatureView:
      - every off-diagonal panel is one feature vs another
      - diagonal panels show feature vs time
      - both mirrored pairwise orientations are shown
    """
    n_dims = target_X.shape[1]
    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    if n_dims == 1:
        axes = np.array([[axes]])

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]

            if row == col:
                # Match the screenshot orientation: time on x, feature on y.
                ax.scatter(
                    background_times,
                    background_X[:, row],
                    s=BACKGROUND_SIZE,
                    alpha=BACKGROUND_ALPHA,
                    linewidths=0,
                    c="0.65",
                    rasterized=True,
                )
                ax.scatter(
                    target_times,
                    target_X[:, row],
                    s=TARGET_SIZE,
                    alpha=TARGET_ALPHA,
                    linewidths=0,
                    c="#008cff",
                    rasterized=True,
                )
                ax.axhline(0, linewidth=0.6, c="0.5")
                ax.set_xlabel("time", fontsize=8)
                ax.set_ylabel(feature_names[row], fontsize=8)

                if FEATURE_AXIS_LIMITS is not None:
                    ax.set_ylim(*FEATURE_AXIS_LIMITS)
            else:
                # x = column feature, y = row feature.
                add_scatter(
                    ax,
                    background_X[:, col],
                    background_X[:, row],
                    target_X[:, col],
                    target_X[:, row],
                )
                ax.set_xlabel(feature_names[col], fontsize=8)
                ax.set_ylabel(feature_names[row], fontsize=8)

            ax.tick_params(labelsize=7)

            # Put the y-dimension label inside the top-right, like Phy.
            ax.text(
                0.97,
                0.95,
                feature_names[row],
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )

    fig.suptitle(
        f"Phy FeatureView clone | unit {TARGET_UNIT} | channels {CHANNEL_IDS}",
        fontsize=14,
    )

    return fig


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHY FEATUREVIEW CLONE")
    print("=" * 80)

    # --------------------------------------------------------
    # Load exactly the files used by Phy for PC FeatureView.
    # --------------------------------------------------------
    pc_features = np.load(
        KS_FOLDER / "pc_features.npy",
        mmap_mode="r",
    )
    pc_feature_ind = np.load(
        KS_FOLDER / "pc_feature_ind.npy",
        mmap_mode="r",
    )
    spike_templates = np.load(
        KS_FOLDER / "spike_templates.npy"
    ).squeeze().astype(np.int64)
    spike_clusters = np.load(
        KS_FOLDER / "spike_clusters.npy"
    ).squeeze().astype(np.int64)
    spike_times_samples = np.load(
        KS_FOLDER / "spike_times.npy"
    ).squeeze().astype(np.int64)

    n_total_spikes = len(spike_clusters)

    if len(spike_templates) != n_total_spikes:
        raise RuntimeError("spike_templates and spike_clusters length mismatch")
    if len(spike_times_samples) != n_total_spikes:
        raise RuntimeError("spike_times and spike_clusters length mismatch")

    sample_rate = load_sample_rate(KS_FOLDER)
    spike_times_sec = spike_times_samples / sample_rate

    print("pc_features shape:", pc_features.shape)
    print("pc_feature_ind shape:", pc_feature_ind.shape)
    print("total spikes:", n_total_spikes)
    print("sample rate:", sample_rate)

    # --------------------------------------------------------
    # Feature-row mapping / features_rows behavior.
    # --------------------------------------------------------
    feature_spike_ids, spike_to_feature_row = load_feature_row_mapping(
        KS_FOLDER,
        n_total_spikes,
        pc_features.shape[0],
    )

    print("feature rows:", len(feature_spike_ids))

    # --------------------------------------------------------
    # SELECTED TARGET CLUSTER
    # --------------------------------------------------------
    target_all = np.flatnonzero(spike_clusters == TARGET_UNIT).astype(np.int64)

    if len(target_all) == 0:
        raise RuntimeError(f"Unit {TARGET_UNIT} not found in spike_clusters.npy")

    target_ids = evenly_pick_from_sorted_ids(
        target_all,
        N_SPIKES_FEATURES,
    )

    # Match FeatureMixin: only retain spikes represented in features_rows.
    target_ids = intersect_with_feature_rows(
        target_ids,
        spike_to_feature_row,
    )

    # --------------------------------------------------------
    # GLOBAL BACKGROUND
    # --------------------------------------------------------
    background_ids = phy_background_spike_ids(
        n_total_spikes,
        N_SPIKES_FEATURES_BACKGROUND,
    )

    # Match FeatureMixin: only retain spikes represented in features_rows.
    background_ids = intersect_with_feature_rows(
        background_ids,
        spike_to_feature_row,
    )

    print("\nTarget cluster spikes total:", len(target_all))
    print("Target spikes shown:", len(target_ids))
    print("Background spikes shown:", len(background_ids))
    print(
        "Background includes target-cluster spikes:",
        int(np.sum(spike_clusters[background_ids] == TARGET_UNIT)),
    )
    print(
        "Unique clusters represented in background:",
        len(np.unique(spike_clusters[background_ids])),
    )

    # --------------------------------------------------------
    # Sparse -> dense feature mapping, Phy style.
    # --------------------------------------------------------
    target_dense, target_present = phy_from_sparse_features(
        target_ids,
        CHANNEL_IDS,
        pc_features,
        pc_feature_ind,
        spike_templates,
        spike_to_feature_row,
        N_PCS_DISPLAY,
    )

    background_dense, background_present = phy_from_sparse_features(
        background_ids,
        CHANNEL_IDS,
        pc_features,
        pc_feature_ind,
        spike_templates,
        spike_to_feature_row,
        N_PCS_DISPLAY,
    )

    feature_names, _ = make_feature_names(
        CHANNEL_IDS,
        N_PCS_DISPLAY,
    )

    target_X = flatten_phy_features(
        target_dense,
        CHANNEL_IDS,
        N_PCS_DISPLAY,
    )
    background_X = flatten_phy_features(
        background_dense,
        CHANNEL_IDS,
        N_PCS_DISPLAY,
    )

    # --------------------------------------------------------
    # Diagnostics that should explain the x=0 / y=0 crosses.
    # --------------------------------------------------------
    print("\nSparse-channel availability among BACKGROUND spikes:")

    for ch_i, channel_id in enumerate(CHANNEL_IDS):
        n_present = int(background_present[:, ch_i].sum())
        n_missing = len(background_present) - n_present
        print(
            f"  Ch {channel_id}: present={n_present}, missing->zero={n_missing} "
            f"({n_missing / max(1, len(background_present)):.3f})"
        )

    all_present = np.all(background_present, axis=1)
    none_present = ~np.any(background_present, axis=1)

    print(
        "  all requested channels present:",
        int(all_present.sum()),
        f"({all_present.mean():.3f})",
    )
    print(
        "  no requested channels present -> origin on all PC panels:",
        int(none_present.sum()),
        f"({none_present.mean():.3f})",
    )

    # Exact zero counts in each visual dimension.
    print("\nExact zero fractions in BACKGROUND visual dimensions:")
    for i, name in enumerate(feature_names):
        frac = np.mean(background_X[:, i] == 0)
        print(f"  {name}: {frac:.3f}")

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------
    fig = make_phy_style_matrix(
        target_X,
        background_X,
        spike_times_sec[target_ids],
        spike_times_sec[background_ids],
        feature_names,
    )

    stem = f"unit_{TARGET_UNIT}_Phy_FeatureView_clone"

    if SAVE_PNG:
        png_path = OUTPUT_FOLDER / f"{stem}.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        print("\nSaved PNG:", png_path)

    if SAVE_PDF:
        pdf_path = OUTPUT_FOLDER / f"{stem}.pdf"
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
        print("Saved PDF:", pdf_path)

    if SAVE_NPZ:
        npz_path = OUTPUT_FOLDER / f"{stem}_data.npz"
        np.savez_compressed(
            npz_path,
            target_spike_ids=target_ids,
            background_spike_ids=background_ids,
            target_cluster_ids=spike_clusters[target_ids],
            background_cluster_ids=spike_clusters[background_ids],
            target_template_ids=spike_templates[target_ids],
            background_template_ids=spike_templates[background_ids],
            target_features=target_X,
            background_features=background_X,
            target_channel_present=target_present,
            background_channel_present=background_present,
            target_times_sec=spike_times_sec[target_ids],
            background_times_sec=spike_times_sec[background_ids],
            feature_names=np.asarray(feature_names, dtype=object),
            channel_ids=np.asarray(CHANNEL_IDS, dtype=np.int64),
        )
        print("Saved NPZ:", npz_path)

    plt.show()


if __name__ == "__main__":
    main()
