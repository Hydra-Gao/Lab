from pathlib import Path

import numpy as np
import pandas as pd

from spikeinterface.metrics.quality.pca_metrics import (
    mahalanobis_metrics,
    d_prime_metric,
    nearest_neighbors_metrics,
    simplified_silhouette_score,
)


# ============================================================
# SETTINGS
# ============================================================

KS_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2"
)

OUTPUT_CSV = KS_FOLDER / "bombcell_output_whitened" / "unwhitened_KS_PC_isolation_metrics_with_covariance_diagnostics.csv"

# IMPORTANT:
N_PCS = 2
N_REF_CHANNELS = 2

MAX_SPIKES_PER_UNIT = 500

NN_MAX_SPIKES = 10000
NN_N_NEIGHBORS = 5

RANDOM_SEED = 42

SAVE_EVERY = 20

pc_quality_rank_threshold = 4
pc_quality_condition_number_threshold = 1e10

# ============================================================
# METRIC FUNCTION
# ============================================================

def compute_metrics_for_space(pcs_flat, labels, unit_id):

    try:
        isolation_distance, l_ratio = mahalanobis_metrics(
            pcs_flat,
            labels,
            unit_id,
        )
    except Exception as exc:
        print(f"[WARN] unit {unit_id}: mahalanobis failed: {exc}")
        isolation_distance = np.nan
        l_ratio = np.nan

    try:
        d_prime = d_prime_metric(
            pcs_flat,
            labels,
            unit_id,
        )
    except Exception as exc:
        print(f"[WARN] unit {unit_id}: d_prime failed: {exc}")
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
        print(f"[WARN] unit {unit_id}: nearest neighbor failed: {exc}")
        nn_hit_rate = np.nan
        nn_miss_rate = np.nan

    try:
        silhouette = simplified_silhouette_score(
            pcs_flat,
            labels,
            unit_id,
        )
    except Exception as exc:
        print(f"[WARN] unit {unit_id}: silhouette failed: {exc}")
        silhouette = np.nan

    return {
        "isolation_distance": isolation_distance,
        "l_ratio": l_ratio,
        "d_prime": d_prime,
        "silhouette": silhouette,
        "nn_hit_rate": nn_hit_rate,
        "nn_miss_rate": nn_miss_rate,
    }


# ============================================================
# BUILD CLUSTER ROW INDEX ONCE
# ============================================================

def build_cluster_index(feature_clusters):
    """
    Build:
        unit_id -> rows in pc_features

    IMPORTANT:
    This is done ONCE rather than np.flatnonzero()
    over 14 million spikes for every target unit.
    """

    print("\nBuilding cluster row index...")

    order = np.argsort(feature_clusters, kind="stable")

    sorted_clusters = feature_clusters[order]

    unit_ids, start_indices, counts = np.unique(
        sorted_clusters,
        return_index=True,
        return_counts=True,
    )

    cluster_rows = {}

    for unit_id, start, count in zip(
        unit_ids,
        start_indices,
        counts,
    ):
        cluster_rows[int(unit_id)] = order[
            start:start + count
        ]

    print("Indexed units:", len(cluster_rows))

    return cluster_rows


# ============================================================
# GET DOMINANT TEMPLATE
# ============================================================

def get_dominant_template(
    unit_rows,
    feature_templates,
):
    templates_for_unit = feature_templates[unit_rows]

    template_ids, counts = np.unique(
        templates_for_unit,
        return_counts=True,
    )

    return int(
        template_ids[np.argmax(counts)]
    )


# ============================================================
# REFERENCE CHANNELS FOR TARGET UNIT
# ============================================================

def get_reference_channels(
    target_template,
    pc_feature_ind,
    templates,
    whitening_mat_inv,
):

    candidate_channels = np.asarray(
        pc_feature_ind[target_template],
        dtype=int,
    )

    template_whitened = np.asarray(
        templates[target_template]
    )

    # Phy/phylib style:
    # samples × channels  @  channels × channels
    template_unwhitened = (
        template_whitened @ whitening_mat_inv
    )

    ptp = np.ptp(
        template_unwhitened,
        axis=0,
    )

    order = np.argsort(
        ptp[candidate_channels]
    )[::-1]

    reference_channels = candidate_channels[
        order[:N_REF_CHANNELS]
    ]

    return np.asarray(
        reference_channels,
        dtype=int,
    )


# ============================================================
# FIND VALID ROWS FOR ONE UNIT
# ============================================================

def get_valid_rows(
    rows,
    reference_channels,
    feature_templates,
    pc_feature_ind,
):
    """
    Keep only spikes whose template contains ALL
    target reference channels.

    Missing KS sparse features are NOT treated as zero.
    """

    if len(rows) == 0:
        return np.array([], dtype=int)

    template_ids = feature_templates[rows]

    valid_mask = np.zeros(
        len(rows),
        dtype=bool,
    )

    # Much faster than testing every spike independently:
    # test once per template
    for template_id in np.unique(template_ids):

        template_mask = (
            template_ids == template_id
        )

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        has_all_channels = np.all(
            np.isin(
                reference_channels,
                local_channels,
            )
        )

        if has_all_channels:
            valid_mask[template_mask] = True

    return rows[valid_mask]


# ============================================================
# CONVERT KS SPARSE PCS -> COMMON FEATURE MATRIX
# ============================================================

def extract_pc_matrix(
    rows,
    reference_channels,
    feature_templates,
    pc_features,
    pc_feature_ind,
):
    """
    Returns:
        X: n_spikes × (N_PCS * N_REF_CHANNELS)
    """

    X = np.empty(
        (
            len(rows),
            N_PCS * len(reference_channels),
        ),
        dtype=np.float32,
    )

    template_ids = feature_templates[rows]

    # Again process by template rather than spike-by-spike
    for template_id in np.unique(template_ids):

        mask = (
            template_ids == template_id
        )

        these_rows = rows[mask]

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        local_indices = []

        for channel_id in reference_channels:

            idx = np.flatnonzero(
                local_channels == channel_id
            )

            if len(idx) != 1:
                raise RuntimeError(
                    f"Template {template_id}: "
                    f"cannot uniquely find channel {channel_id}"
                )

            local_indices.append(int(idx[0]))

        # pc_features:
        # spikes × pcs × local channels

        block = np.asarray(
            pc_features[
                these_rows,
                :N_PCS,
                :
            ]
        )

        block = block[
            :,
            :,
            local_indices
        ]

        X[mask] = block.reshape(
            len(these_rows),
            -1,
        )

    return X


# ============================================================
# COMPUTE ONE TARGET UNIT
# ============================================================

def compute_one_unit(
    target_unit_id,
    cluster_rows,
    feature_templates,
    pc_features,
    pc_feature_ind,
    templates,
    whitening_mat_inv,
):

    rng = np.random.default_rng(
        RANDOM_SEED + int(target_unit_id)
    )

    target_rows_all = cluster_rows[
        target_unit_id
    ]

    dominant_template = get_dominant_template(
        target_rows_all,
        feature_templates,
    )

    reference_channels = get_reference_channels(
        dominant_template,
        pc_feature_ind,
        templates,
        whitening_mat_inv,
    )

    X_blocks = []
    y_blocks = []

    neighbor_units = []

    n_target = 0
    n_other = 0

    # --------------------------------------------------------
    # Compare against spatially relevant units.
    #
    # A unit contributes only if it has spikes whose KS
    # template actually contains all reference channels.
    # --------------------------------------------------------

    for other_unit_id, rows in cluster_rows.items():

        valid_rows = get_valid_rows(
            rows,
            reference_channels,
            feature_templates,
            pc_feature_ind,
        )

        if len(valid_rows) == 0:
            continue

        # Same sampling philosophy as your previous SI PCA
        if len(valid_rows) > MAX_SPIKES_PER_UNIT:
            valid_rows = rng.choice(
                valid_rows,
                size=MAX_SPIKES_PER_UNIT,
                replace=False,
            )

        valid_rows = np.asarray(
            valid_rows,
            dtype=int,
        )

        X_unit = extract_pc_matrix(
            valid_rows,
            reference_channels,
            feature_templates,
            pc_features,
            pc_feature_ind,
        )

        X_blocks.append(X_unit)

        y_blocks.append(
            np.full(
                len(valid_rows),
                other_unit_id,
                dtype=int,
            )
        )

        neighbor_units.append(
            other_unit_id
        )

        if other_unit_id == target_unit_id:
            n_target += len(valid_rows)
        else:
            n_other += len(valid_rows)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if n_target == 0:
        raise RuntimeError(
            "No valid target spikes after "
            "reference-channel filtering."
        )

    if n_other == 0:
        raise RuntimeError(
            "No valid background spikes."
        )

    X = np.concatenate(
        X_blocks,
        axis=0,
    )

    labels = np.concatenate(
        y_blocks,
        axis=0,
    )

    cov_diag = compute_covariance_diagnostics(
        X,
        labels,
        target_unit_id,
    )

    pc_quality = (
        "fail"
        if (
            cov_diag["cov_rank"] < pc_quality_rank_threshold
            or not np.isfinite(cov_diag["cov_condition_number"])
            or cov_diag["cov_condition_number"] > pc_quality_condition_number_threshold
        )
        else "pass"
    )

    metrics = compute_metrics_for_space(
        X,
        labels,
        target_unit_id,
    )

    return {
        "unit_id": target_unit_id,

        "dominant_template":
            dominant_template,

        "reference_channels":
            ",".join(
                map(str, reference_channels)
            ),

        "n_pcs":
            N_PCS,

        "n_channels":
            len(reference_channels),

        "n_features":
            X.shape[1],

        "n_neighbor_units":
            len(neighbor_units),

        "n_target_spikes":
            n_target,

        "n_other_spikes":
            n_other,

        **metrics,
        **cov_diag,
        "pc_quality": pc_quality,

        "status":
            "ok",
    }


# ============================================================
# COMPUTE covariance_diagnostics
# ============================================================


def compute_covariance_diagnostics(
    pcs_flat,
    labels,
    unit_id,
):
    target_X = pcs_flat[labels == unit_id]

    n_spikes, n_features = target_X.shape

    if n_spikes < 2:
        return {
            "cov_rank": np.nan,
            "cov_full_rank": False,
            "cov_condition_number": np.nan,
            "log10_cov_condition_number": np.nan,
            "cov_min_eigenvalue": np.nan,
            "cov_max_eigenvalue": np.nan,
            "min_feature_std": np.nan,
            "n_unique_target_vectors": np.nan,
            "unique_fraction": np.nan,
        }

    cov = np.cov(
        target_X,
        rowvar=False,
    )

    # Make sure 1-feature edge case is still 2D
    cov = np.atleast_2d(cov)

    cov_rank = np.linalg.matrix_rank(cov)

    try:
        cond = np.linalg.cond(cov)
    except Exception:
        cond = np.nan

    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))
    except Exception:
        min_eig = np.nan
        max_eig = np.nan

    feature_std = np.std(
        target_X,
        axis=0,
        ddof=1,
    )

    n_unique = np.unique(
        target_X,
        axis=0,
    ).shape[0]

    if np.isfinite(cond) and cond > 0:
        log10_cond = np.log10(cond)
    elif np.isinf(cond):
        log10_cond = np.inf
    else:
        log10_cond = np.nan

    return {
        "cov_rank": int(cov_rank),
        "cov_full_rank": bool(cov_rank == n_features),
        "cov_condition_number": cond,
        "log10_cov_condition_number": log10_cond,
        "cov_min_eigenvalue": min_eig,
        "cov_max_eigenvalue": max_eig,
        "min_feature_std": float(np.min(feature_std)),
        "n_unique_target_vectors": int(n_unique),
        "unique_fraction": n_unique / n_spikes,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Loading Kilosort PCA files")
    print("=" * 72)

    pc_features = np.load(
        KS_FOLDER / "pc_features.npy",
        mmap_mode="r",
    )

    pc_feature_ind = np.load(
        KS_FOLDER / "pc_feature_ind.npy"
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

    whitening_mat_inv = np.load(
        KS_FOLDER / "whitening_mat_inv.npy"
    )

    print(
        "pc_features:",
        pc_features.shape
    )

    print(
        "pc_feature_ind:",
        pc_feature_ind.shape
    )

    print(
        "spike_templates:",
        spike_templates.shape
    )

    print(
        "spike_clusters:",
        spike_clusters.shape
    )

    print(
        "templates:",
        templates.shape
    )

    # --------------------------------------------------------
    # Usually KS4 has one pc_features row per spike.
    # --------------------------------------------------------

    if pc_features.shape[0] != len(spike_clusters):

        feature_ids_path = (
            KS_FOLDER /
            "pc_feature_spike_ids.npy"
        )

        if not feature_ids_path.exists():
            raise RuntimeError(
                "pc_features rows != spike count, "
                "and pc_feature_spike_ids.npy "
                "was not found."
            )

        feature_spike_ids = np.load(
            feature_ids_path
        ).astype(int)

    else:

        feature_spike_ids = np.arange(
            pc_features.shape[0]
        )

    feature_clusters = spike_clusters[
        feature_spike_ids
    ]

    feature_templates = spike_templates[
        feature_spike_ids
    ]

    # --------------------------------------------------------
    # Build index ONCE
    # --------------------------------------------------------

    cluster_rows = build_cluster_index(
        feature_clusters
    )

    unit_ids = sorted(
        cluster_rows.keys()
    )

    print("\nTotal units:", len(unit_ids))

    # --------------------------------------------------------
    # Optional resume
    # --------------------------------------------------------

    completed_units = set()
    rows = []

    if OUTPUT_CSV.exists():

        print(
            "\nExisting output found. "
            "Resuming from:"
        )
        print(OUTPUT_CSV)

        existing = pd.read_csv(
            OUTPUT_CSV
        )

        rows = existing.to_dict(
            orient="records"
        )

        completed_units = set(
            existing.loc[
                existing["status"] == "ok",
                "unit_id"
            ].astype(int)
        )

        print(
            "Already completed:",
            len(completed_units)
        )

    # --------------------------------------------------------
    # All units
    # --------------------------------------------------------

    for i, target_unit_id in enumerate(
        unit_ids,
        start=1,
    ):

        if target_unit_id in completed_units:

            print(
                f"[{i}/{len(unit_ids)}] "
                f"unit {target_unit_id}: SKIP"
            )

            continue

        print("\n" + "=" * 72)

        print(
            f"[{i}/{len(unit_ids)}] "
            f"UNIT {target_unit_id}"
        )

        print("=" * 72)

        try:

            row = compute_one_unit(
                target_unit_id,
                cluster_rows,
                feature_templates,
                pc_features,
                pc_feature_ind,
                templates,
                whitening_mat_inv,
            )

            print(
                pd.Series(row).to_string()
            )

        except Exception as exc:

            print(
                f"[FAILED] unit "
                f"{target_unit_id}: {exc}"
            )

            row = {
                "unit_id":
                    target_unit_id,

                "dominant_template":
                    np.nan,

                "reference_channels":
                    "",

                "n_pcs":
                    N_PCS,

                "n_channels":
                    N_REF_CHANNELS,

                "n_features":
                    N_PCS * N_REF_CHANNELS,

                "n_neighbor_units":
                    np.nan,

                "n_target_spikes":
                    np.nan,

                "n_other_spikes":
                    np.nan,

                "isolation_distance":
                    np.nan,

                "l_ratio":
                    np.nan,

                "d_prime":
                    np.nan,

                "silhouette":
                    np.nan,

                "nn_hit_rate":
                    np.nan,

                "nn_miss_rate":
                    np.nan,

                "status":
                    f"failed: {exc}",
            }

        rows.append(row)

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            i % SAVE_EVERY == 0
            or i == len(unit_ids)
        ):

            results = pd.DataFrame(
                rows
            )

            results = results.sort_values(
                "unit_id"
            )

            results.to_csv(
                OUTPUT_CSV,
                index=False,
            )

            print(
                "\n[CHECKPOINT] Saved:"
            )
            print(OUTPUT_CSV)

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "unit_id"
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        "Units:",
        len(results)
    )

    print(
        "Successful:",
        np.sum(
            results["status"] == "ok"
        )
    )

    print(
        "Failed:",
        np.sum(
            results["status"] != "ok"
        )
    )

    print(
        "\nSaved:"
    )
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()