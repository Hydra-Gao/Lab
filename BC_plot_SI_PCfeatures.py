import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
from spikeinterface.core.template_tools import get_template_extremum_channel


def _guess_sample_field(spikes):
    for name in ("sample_index", "sample_ind", "sample_idx"):
        if name in spikes.dtype.names:
            return name
    raise KeyError(
        f"Could not find sample index field in random_spikes dtype names: {spikes.dtype.names}"
    )


def get_si_metric_neighbors(analyzer, unit_id, peak_sign="neg"):
    """
    Reproduce the same neighbor-unit selection logic used by
    spikeinterface.metrics.quality.quality_metrics._prepare_data().
    """
    unit_ids = list(analyzer.unit_ids)

    if analyzer.is_sparse():
        extremum_channels = get_template_extremum_channel(analyzer, peak_sign=peak_sign)
        neighbor_channel_ids = list(analyzer.sparsity.unit_id_to_channel_ids[unit_id])

        neighbor_unit_ids = [
            other_unit
            for other_unit in unit_ids
            if extremum_channels[other_unit] in neighbor_channel_ids
        ]
    else:
        neighbor_channel_ids = list(analyzer.channel_ids)
        neighbor_unit_ids = unit_ids

    return neighbor_unit_ids, neighbor_channel_ids


def summarize_si_metric_space(analyzer, unit_id, peak_sign="neg"):
    pc = analyzer.get_extension("principal_components")
    if pc is None:
        raise ValueError("No principal_components extension found.")

    neighbor_unit_ids, neighbor_channel_ids = get_si_metric_neighbors(
        analyzer, unit_id, peak_sign=peak_sign
    )

    n_components = pc.params["n_components"]
    n_channels_metric = len(neighbor_channel_ids)
    flattened_dim = n_components * n_channels_metric

    print("=" * 60)
    print(f"Target unit: {unit_id}")
    print(f"PCA mode: {pc.params['mode']}")
    print(f"PCA n_components per channel: {n_components}")
    print(f"PCA whiten: {pc.params.get('whiten', None)}")
    print(f"Metric neighbor unit count: {len(neighbor_unit_ids)}")
    print(f"Metric neighbor units: {neighbor_unit_ids}")
    print(f"Metric neighbor channel count: {n_channels_metric}")
    print(f"Metric neighbor channels: {neighbor_channel_ids}")
    print(f"Flattened feature dimension used by SI metrics: {flattened_dim}")
    print("=" * 60)


def _subsample_indices(mask, max_n=None, seed=0):
    inds = np.flatnonzero(mask)
    if max_n is None or len(inds) <= max_n:
        return inds
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(inds, size=max_n, replace=False))


def plot_si_featureview_like(
    analyzer,
    unit_id,
    plot_channel_ids=None,
    peak_sign="neg",
    max_target_spikes=2500,
    max_background_spikes=2500,
    seed=0,
    figsize=(10, 8.5),
):
    """
    Plot SI PCA projections in a Phy-like FeatureView style.

    Important:
    - This uses SI's principal_components extension, not KS4 pc_features.npy.
    - Gray points are SI metric-neighbor units.
    - Only 2 channels x first 2 PCs are shown (like a Phy-style visual summary).
    """
    pc = analyzer.get_extension("principal_components")
    if pc is None:
        raise ValueError("No principal_components extension found.")

    if pc.params["mode"] == "concatenated":
        raise ValueError(
            "This plotting helper expects per-channel PCs. "
            "mode='concatenated' is not compatible with a Phy-like channel-by-channel view."
        )

    if pc.params["n_components"] < 2:
        raise ValueError("Need at least 2 PCA components to emulate A/B display.")

    # Reproduce SI metric neighbor-unit logic
    neighbor_unit_ids, metric_neighbor_channel_ids = get_si_metric_neighbors(
        analyzer, unit_id, peak_sign=peak_sign
    )

    # Choose 2 channels for plotting
    if plot_channel_ids is None:
        if analyzer.is_sparse():
            sparse_chs = list(analyzer.sparsity.unit_id_to_channel_ids[unit_id])
            if len(sparse_chs) < 2:
                raise ValueError(
                    f"Unit {unit_id} has fewer than 2 sparse channels: {sparse_chs}"
                )
            plot_channel_ids = sparse_chs[:2]
        else:
            all_chs = list(analyzer.channel_ids)
            plot_channel_ids = all_chs[:2]

    if len(plot_channel_ids) != 2:
        raise ValueError("plot_channel_ids must contain exactly 2 channel IDs.")

    # Get SI PCA projections on those two channels, for target + SI-neighbor units
    some_projections, spike_unit_indices = pc.get_some_projections(
        channel_ids=plot_channel_ids,
        unit_ids=neighbor_unit_ids,
    )
    # some_projections shape: (n_spikes, n_components, 2)

    # Recover the corresponding sampled spikes in the same unit subset
    some_spikes = analyzer.get_extension("random_spikes").get_random_spikes()
    sorting = analyzer.sorting
    unit_id_array = np.asarray(analyzer.unit_ids)

    neighbor_unit_indices = sorting.ids_to_indices(neighbor_unit_ids)
    selected_inds = np.flatnonzero(np.isin(some_spikes["unit_index"], neighbor_unit_indices))
    selected_spikes = some_spikes[selected_inds]

    if len(selected_spikes) != some_projections.shape[0]:
        raise RuntimeError(
            f"Mismatch between selected_spikes ({len(selected_spikes)}) and "
            f"some_projections ({some_projections.shape[0]})."
        )

    sample_field = _guess_sample_field(selected_spikes)

    # Sampling frequency
    if hasattr(analyzer, "sampling_frequency"):
        fs = analyzer.sampling_frequency
    elif analyzer.has_recording():
        fs = analyzer.recording.get_sampling_frequency()
    else:
        raise ValueError("Could not determine sampling frequency from analyzer.")

    times_sec = selected_spikes[sample_field] / fs
    labels = unit_id_array[spike_unit_indices]
    is_target = labels == unit_id
    is_background = ~is_target

    # Subsample for plotting, like Phy does
    target_inds = _subsample_indices(is_target, max_target_spikes, seed=seed)
    bg_inds = _subsample_indices(is_background, max_background_spikes, seed=seed + 1)

    ch1, ch2 = plot_channel_ids

    # First two PCs on each of the two channels
    # shape indexing: [spike, component, channel]
    features = {
        f"{ch1}A": some_projections[:, 0, 0],
        f"{ch2}A": some_projections[:, 0, 1],
        f"{ch1}B": some_projections[:, 1, 0],
        f"{ch2}B": some_projections[:, 1, 1],
    }
    feature_names = list(features.keys())

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.patch.set_facecolor("black")

    for i, y_name in enumerate(feature_names):
        for j, x_name in enumerate(feature_names):
            ax = axes[i, j]
            ax.set_facecolor("black")

            y_all = features[y_name]

            if i == j:
                # diagonal: feature vs time
                x_bg = times_sec[bg_inds]
                y_bg = y_all[bg_inds]
                x_t = times_sec[target_inds]
                y_t = y_all[target_inds]

                ax.scatter(x_bg, y_bg, s=4, c="0.6", alpha=0.45, linewidths=0)
                ax.scatter(x_t, y_t, s=4, c="dodgerblue", alpha=0.85, linewidths=0)

                ax.axhline(0, color="0.35", lw=0.6)
                bottom_label = "time"
            else:
                x_all = features[x_name]

                x_bg = x_all[bg_inds]
                y_bg = y_all[bg_inds]
                x_t = x_all[target_inds]
                y_t = y_all[target_inds]

                ax.scatter(x_bg, y_bg, s=4, c="0.6", alpha=0.45, linewidths=0)
                ax.scatter(x_t, y_t, s=4, c="dodgerblue", alpha=0.85, linewidths=0)

                ax.axhline(0, color="0.35", lw=0.6)
                ax.axvline(0, color="0.35", lw=0.6)
                bottom_label = x_name

            # Remove ticks to mimic Phy
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels placed like Phy-ish panels
            ax.text(
                0.97, 0.95, y_name,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=14, color="white"
            )
            ax.text(
                0.50, 0.02, bottom_label,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=12, color="white"
            )

            for spine in ax.spines.values():
                spine.set_color("0.4")
                spine.set_linewidth(0.6)

    fig.suptitle(
        f"SI principal_components shown in a Phy-like FeatureView style\n"
        f"Target unit={unit_id} | plot channels={plot_channel_ids} | "
        f"metric neighbors={len(neighbor_unit_ids)} units",
        fontsize=13,
        color="white",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return {
        "plot_channel_ids": plot_channel_ids,
        "neighbor_unit_ids": neighbor_unit_ids,
        "metric_neighbor_channel_ids": metric_neighbor_channel_ids,
        "n_target_plotted": len(target_inds),
        "n_background_plotted": len(bg_inds),
    }


def get_top_template_channels(analyzer, unit_id, n=2):
    templates_ext = analyzer.get_extension("templates")

    # average template
    template = templates_ext.get_unit_template(
        unit_id,
        operator="average",
    )

    # template shape: samples x channels
    ptp = np.ptp(template, axis=0)

    top_indices = np.argsort(ptp)[::-1][:n]

    return list(analyzer.channel_ids[top_indices])


analyzer = si.load_sorting_analyzer(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\sorting_analyzer"
)

unit_id = 100

top_channels = get_top_template_channels(
    analyzer,
    unit_id=unit_id,
    n=2,
)

print(top_channels)

summarize_si_metric_space(analyzer, unit_id, peak_sign="neg")

info = plot_si_featureview_like(
    analyzer,
    unit_id=unit_id,
    plot_channel_ids=[44, 42], # or = None, to auto-select the first 2 channels
    # plot_channel_ids=top_channels,
    peak_sign="neg",
    max_target_spikes=2500,
    max_background_spikes=2500,
    seed=0,
)
print(info)

