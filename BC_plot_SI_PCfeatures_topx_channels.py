from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si

from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.metrics.quality.pca_metrics import (
    mahalanobis_metrics,
    d_prime_metric,
    nearest_neighbors_metrics,
    simplified_silhouette_score,
)


# ============================================================
# USER SETTINGS
# ============================================================

ANALYZER_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\sorting_analyzer"
)

OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\top_channel_pc_test"
)

TARGET_UNIT_ID = 36
TOP_N_CHANNELS = [1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
PEAK_SIGN = "neg"

# SI 0.104.8 NearestNeighbor defaults
NN_MAX_SPIKES = 10000
NN_N_NEIGHBORS = 5

MAKE_PHY_LIKE_PLOTS = True
PHY_LIKE_N_VALUES = [2, 4, 8]
MAX_TARGET_PLOT = 2500
MAX_BACKGROUND_PLOT = 2500
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def get_metric_neighbors(analyzer, unit_id, peak_sign="neg"):
    """Reproduce SI quality_metrics._prepare_data() neighbor selection."""
    unit_ids = list(analyzer.unit_ids)

    if analyzer.is_sparse():
        extremum_channels = get_template_extremum_channel(
            analyzer, peak_sign=peak_sign
        )
        full_sparse_channel_ids = list(
            analyzer.sparsity.unit_id_to_channel_ids[unit_id]
        )
        neighbor_unit_ids = [
            other_unit
            for other_unit in unit_ids
            if extremum_channels[other_unit] in full_sparse_channel_ids
        ]
    else:
        full_sparse_channel_ids = list(analyzer.channel_ids)
        neighbor_unit_ids = unit_ids

    return neighbor_unit_ids, full_sparse_channel_ids


def get_ranked_template_channels(analyzer, unit_id):
    """Rank available channels by target-unit average-template PTP amplitude."""
    templates_ext = analyzer.get_extension("templates")
    if templates_ext is None:
        raise RuntimeError("Analyzer has no 'templates' extension.")

    template = np.asarray(
        templates_ext.get_unit_template(unit_id, operator="average")
    )

    if template.ndim != 2:
        raise RuntimeError(f"Expected 2D template, got {template.shape}")

    if analyzer.is_sparse():
        sparse_channel_ids = np.asarray(
            analyzer.sparsity.unit_id_to_channel_ids[unit_id]
        )
    else:
        sparse_channel_ids = np.asarray(analyzer.channel_ids)

    if template.shape[1] == len(sparse_channel_ids):
        template_channel_ids = sparse_channel_ids
    elif template.shape[1] == analyzer.get_num_channels():
        template_channel_ids = np.asarray(analyzer.channel_ids)
    else:
        raise RuntimeError(
            "Cannot map template columns to channels.\n"
            f"template.shape={template.shape}\n"
            f"n_sparse={len(sparse_channel_ids)}\n"
            f"n_total={analyzer.get_num_channels()}"
        )

    ptp = np.ptp(template, axis=0)
    order = np.argsort(ptp)[::-1]

    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "channel_id": template_channel_ids[order],
            "template_ptp": ptp[order],
        }
    )


def prepare_full_si_pc_space(analyzer, neighbor_unit_ids):
    """
    Reproduce SI 0.104.8 quality_metrics._prepare_data() as closely as possible.

    Important:
    First get PCA projections for ALL analyzer units/channels,
    then select neighbor units afterwards.
    """
    pca_ext = analyzer.get_extension("principal_components")
    if pca_ext is None:
        raise RuntimeError("Analyzer has no 'principal_components' extension.")

    if pca_ext.params["mode"] == "concatenated":
        raise RuntimeError(
            "This test expects per-channel PCA, not concatenated mode."
        )

    # This matches SI quality_metrics._prepare_data():
    dense_projections, spike_unit_indices = pca_ext.get_some_projections(
        channel_ids=None,
        unit_ids=analyzer.unit_ids,
    )

    all_labels = np.asarray(analyzer.unit_ids)[spike_unit_indices]

    # Now restrict to the same neighbor units used for this target unit
    neighbor_mask = np.isin(all_labels, neighbor_unit_ids)

    projections = dense_projections[neighbor_mask]
    labels = all_labels[neighbor_mask]

    return projections, labels


def make_pc_space_from_channels(
    analyzer,
    full_projections,
    labels,
    channel_ids,
):
    """
    Take channel dimensions from the already-aligned full SI PCA matrix.
    Nothing except the selected feature dimensions changes.
    """
    channel_ids = list(channel_ids)

    channel_indices = analyzer.channel_ids_to_indices(channel_ids)

    projections = full_projections[:, :, channel_indices]

    pcs_flat = projections.reshape(projections.shape[0], -1)

    return pcs_flat


def compute_metrics_for_space(pcs_flat, labels, unit_id):
    """Use the same underlying functions as SI 0.104.8 quality metrics."""
    try:
        isolation_distance, l_ratio = mahalanobis_metrics(
            pcs_flat, labels, unit_id
        )
    except Exception as exc:
        print(f"[WARN] Mahalanobis failed: {exc}")
        isolation_distance, l_ratio = np.nan, np.nan

    try:
        d_prime = d_prime_metric(pcs_flat, labels, unit_id)
    except Exception as exc:
        print(f"[WARN] d_prime failed: {exc}")
        d_prime = np.nan

    try:
        nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(
            pcs_flat,
            labels,
            unit_id,
            max_spikes=NN_MAX_SPIKES,
            n_neighbors=NN_N_NEIGHBORS,
        )
    except Exception as exc:
        print(f"[WARN] nearest-neighbor failed: {exc}")
        nn_hit_rate, nn_miss_rate = np.nan, np.nan

    try:
        silhouette = simplified_silhouette_score(
            pcs_flat, labels, unit_id
        )
    except Exception as exc:
        print(f"[WARN] silhouette failed: {exc}")
        silhouette = np.nan

    return {
        "isolation_distance": isolation_distance,
        "l_ratio": l_ratio,
        "d_prime": d_prime,
        "silhouette": silhouette,
        "nn_hit_rate": nn_hit_rate,
        "nn_miss_rate": nn_miss_rate,
    }


def _subsample(indices, max_n, rng):
    indices = np.asarray(indices)
    if len(indices) <= max_n:
        return indices
    return np.sort(rng.choice(indices, size=max_n, replace=False))


def get_sample_times_for_units(analyzer, neighbor_unit_ids):
    random_ext = analyzer.get_extension("random_spikes")
    some_spikes = random_ext.get_random_spikes()

    unit_indices = analyzer.sorting.ids_to_indices(neighbor_unit_ids)
    selected = np.flatnonzero(
        np.isin(some_spikes["unit_index"], unit_indices)
    )
    selected_spikes = some_spikes[selected]

    sample_field = None
    for candidate in ("sample_index", "sample_ind", "sample_idx"):
        if candidate in selected_spikes.dtype.names:
            sample_field = candidate
            break

    if sample_field is None:
        return None

    fs = analyzer.sorting.get_sampling_frequency()
    return selected_spikes[sample_field] / fs


def plot_phy_like(analyzer, unit_id, selected_channel_ids, neighbor_unit_ids, output_file):
    """Phy-like 4x4 display using first 2 selected channels and first 2 SI PCs."""
    selected_channel_ids = list(selected_channel_ids)
    if len(selected_channel_ids) < 2:
        return

    plot_channels = selected_channel_ids[:2]
    pca_ext = analyzer.get_extension("principal_components")

    projections, spike_unit_indices = pca_ext.get_some_projections(
        channel_ids=plot_channels,
        unit_ids=neighbor_unit_ids,
    )
    labels = np.asarray(analyzer.unit_ids)[spike_unit_indices]

    if projections.shape[1] < 2:
        return

    times_sec = get_sample_times_for_units(analyzer, neighbor_unit_ids)
    if times_sec is None or len(times_sec) != len(labels):
        times_sec = np.arange(len(labels))

    target = np.flatnonzero(labels == unit_id)
    background = np.flatnonzero(labels != unit_id)

    rng = np.random.default_rng(RANDOM_SEED)
    target = _subsample(target, MAX_TARGET_PLOT, rng)
    background = _subsample(background, MAX_BACKGROUND_PLOT, rng)

    ch1, ch2 = plot_channels
    features = {
        f"{ch1}A": projections[:, 0, 0],
        f"{ch2}A": projections[:, 0, 1],
        f"{ch1}B": projections[:, 1, 0],
        f"{ch2}B": projections[:, 1, 1],
    }
    names = list(features)

    fig, axes = plt.subplots(4, 4, figsize=(10, 8.5))
    fig.suptitle(
        f"SI PCA, Phy-like view | unit {unit_id} | top channels {plot_channels}",
        fontsize=13,
    )

    for i, y_name in enumerate(names):
        for j, x_name in enumerate(names):
            ax = axes[i, j]
            y = features[y_name]

            if i == j:
                ax.scatter(times_sec[background], y[background], s=3, alpha=0.25)
                ax.scatter(times_sec[target], y[target], s=3, alpha=0.7)
                ax.set_xlabel("time")
            else:
                x = features[x_name]
                ax.scatter(x[background], y[background], s=3, alpha=0.25)
                ax.scatter(x[target], y[target], s=3, alpha=0.7)
                ax.set_xlabel(x_name)

            ax.set_ylabel(y_name)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(results_df, output_file):
    metric_names = [
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "silhouette",
        "nn_hit_rate",
        "nn_miss_rate",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    axes = axes.ravel()

    x = results_df["n_channels"].to_numpy()

    for ax, metric in zip(axes, metric_names):
        ax.plot(x, results_df[metric].to_numpy(), marker="o")
        ax.set_xlabel("Number of strongest channels used")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading analyzer:")
    print(ANALYZER_FOLDER)
    analyzer = si.load_sorting_analyzer(
        ANALYZER_FOLDER,
        load_extensions=True,
    )

    if TARGET_UNIT_ID not in analyzer.unit_ids:
        raise ValueError(f"Unit {TARGET_UNIT_ID} not found")

    pca_ext = analyzer.get_extension("principal_components")
    if pca_ext is None:
        raise RuntimeError("principal_components extension is missing")

    print("\nPCA params:")
    print(pca_ext.params)

    neighbor_unit_ids, full_sparse_channel_ids = get_metric_neighbors(
        analyzer,
        TARGET_UNIT_ID,
        peak_sign=PEAK_SIGN,
    )
    print("\nPreparing full SI PCA space...")

    full_projections, labels = prepare_full_si_pc_space(
        analyzer,
        neighbor_unit_ids,
    )

    print("Full aligned PCA shape:", full_projections.shape)
    print("Neighbor spikes:", len(labels))
    print("Target spikes:", np.sum(labels == TARGET_UNIT_ID))
    print("Other spikes:", np.sum(labels != TARGET_UNIT_ID))

    print("\n" + "=" * 72)
    print("SANITY CHECK: reproduce original SI metrics")
    print("=" * 72)

    original_pcs_flat = make_pc_space_from_channels(
        analyzer,
        full_projections,
        labels,
        full_sparse_channel_ids,
    )

    original_manual_metrics = compute_metrics_for_space(
        original_pcs_flat,
        labels,
        TARGET_UNIT_ID,
    )

    print("\nManual reproduction from full SI PCA space:")
    print(pd.Series(original_manual_metrics).to_string())

    qm = analyzer.get_extension("quality_metrics").get_data()

    print("\nStored SI quality_metrics:")
    stored_cols = [
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "silhouette",
        "nn_hit_rate",
        "nn_miss_rate",
    ]
    print(qm.loc[TARGET_UNIT_ID, stored_cols])

    print("\nTarget unit:", TARGET_UNIT_ID)
    print("SI metric neighbor units:", len(neighbor_unit_ids))
    print("Full sparsity channel count:", len(full_sparse_channel_ids))
    print("Full sparsity channels:", full_sparse_channel_ids)

    ranking_df = get_ranked_template_channels(analyzer, TARGET_UNIT_ID)
    ranking_df = ranking_df[
        ranking_df["channel_id"].isin(full_sparse_channel_ids)
    ].reset_index(drop=True)

    ranking_csv = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_channel_ranking.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    print("\nTop template channels:")
    print(ranking_df.head(20).to_string(index=False))
    print("Saved:", ranking_csv)

    available = len(ranking_df)
    n_values = sorted(
        set(n for n in TOP_N_CHANNELS if 1 <= n <= available)
    )
    if available not in n_values:
        n_values.append(available)

    rows = []

    for n in n_values:
        selected_channels = ranking_df.loc[: n - 1, "channel_id"].tolist()

        print("\n" + "=" * 72)
        print(f"TOP {n} CHANNELS")
        print(selected_channels)

        # Preserve analyzer's natural channel order after choosing Top-N
        selected_channel_set = set(selected_channels)

        selected_channels_ordered = [
            ch
            for ch in full_sparse_channel_ids
            if ch in selected_channel_set
        ]

        pcs_flat = make_pc_space_from_channels(
            analyzer,
            full_projections,
            labels,
            selected_channels_ordered,
        )

        metrics = compute_metrics_for_space(
            pcs_flat,
            labels,
            TARGET_UNIT_ID,
        )

        row = {
            "unit_id": TARGET_UNIT_ID,
            "n_channels": n,
            "selected_channels": ",".join(map(str, selected_channels_ordered)),
            "n_pcs_per_channel": pca_ext.params["n_components"],
            "n_features_flat": pcs_flat.shape[1],
            "n_neighbor_units": len(neighbor_unit_ids),
            "n_target_sampled_spikes": int(np.sum(labels == TARGET_UNIT_ID)),
            "n_other_sampled_spikes": int(np.sum(labels != TARGET_UNIT_ID)),
            **metrics,
        }
        rows.append(row)

        print(pd.Series(metrics).to_string())

        if MAKE_PHY_LIKE_PLOTS and n in PHY_LIKE_N_VALUES:
            output_png = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_top_{n}_channels_phy_like.png"
            plot_phy_like(
                analyzer,
                TARGET_UNIT_ID,
                selected_channels,
                neighbor_unit_ids,
                output_png,
            )
            print("Saved plot:", output_png)

    results_df = pd.DataFrame(rows)

    results_csv = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_topN_pc_metrics.csv"
    results_df.to_csv(results_csv, index=False)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    show_cols = [
        "n_channels",
        "n_features_flat",
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "silhouette",
        "nn_hit_rate",
        "nn_miss_rate",
    ]
    print(results_df[show_cols].to_string(index=False))
    print("\nSaved:", results_csv)

    curves_png = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_metric_vs_topN_channels.png"
    plot_metric_curves(results_df, curves_png)
    print("Saved:", curves_png)
    print("\nDone.")


if __name__ == "__main__":
    main()
