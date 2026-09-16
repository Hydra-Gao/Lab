from pathlib import Path
from itertools import combinations
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# Units to inspect manually.
# Example: TARGET_UNITS = [36, 82, 88, 527, 685]
TARGET_UNITS = [802]

# IMPORTANT: keep these identical to the metric script if you want the
# plotted feature space to be exactly the same space used for QC metrics.
N_PCS = 2
N_REF_CHANNELS = 2

# Same per-unit sampling philosophy as the metric script.
MAX_SPIKES_PER_UNIT = 1000
RANDOM_SEED = 42

# Same NN metric settings as the metric script.
NN_MAX_SPIKES = 10000
NN_N_NEIGHBORS = 5

# Same covariance-QC thresholds as the metric script.
PC_QUALITY_RANK_THRESHOLD = 4
PC_QUALITY_CONDITION_NUMBER_THRESHOLD = 1e10

# Output folder.
OUTPUT_FOLDER = KS_FOLDER / "pc_manual_review"

# One combined PDF containing all selected units.
SAVE_COMBINED_PDF = False
COMBINED_PDF_NAME = "selected_units_PC_manual_review.pdf"

# Also save one PDF per unit.
SAVE_INDIVIDUAL_PDFS = True

# Save the exact sampled X / labels used for the plots and metrics.
# This is useful if you later want to replot or audit a discordant unit.
SAVE_NPZ_PER_UNIT = False

# Save 2D metric values for every pairwise projection to CSV.
# These values are NOT printed on the figure by default, so they do not
# bias the manual visual rating.
SAVE_PAIRWISE_METRICS = True

# For a blinded manual review, keep this True. The PDF will show unit ID,
# channels, dimensions, spike counts, covariance validity, and the plots,
# but it will hide the six automated separation metrics.
# The values are still written to review_summary.csv.
BLIND_REVIEW = False

# Rendering only. The metrics always use the full X assembled below.
# If background contains more points than this, the PDF scatter plots use
# a deterministic subsample for speed/readability.
MAX_BACKGROUND_POINTS_TO_PLOT = 20000
MAX_TARGET_POINTS_TO_PLOT = 5000

# Robust axis limits for pairwise plots. 0.5 means [0.5, 99.5] percentiles.
# Set to None to use matplotlib autoscaling with every point.
PAIRWISE_ROBUST_PERCENTILE = 0.5

# Figure cosmetics.
PAIRWISE_POINT_SIZE = 5
TARGET_ALPHA = 0.75
BACKGROUND_ALPHA = 0.18
HIST_BINS = 80


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def compute_metrics_for_space(pcs_flat, labels, unit_id):
    """Same six PCA-space metrics as the original metric script."""

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


def compute_covariance_diagnostics(pcs_flat, labels, unit_id):
    """Same target-cluster covariance diagnostics as the metric script."""

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

    cov = np.cov(target_X, rowvar=False)
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

    feature_std = np.std(target_X, axis=0, ddof=1)
    n_unique = np.unique(target_X, axis=0).shape[0]

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
# BUILD CLUSTER ROW INDEX ONCE
# ============================================================

def build_cluster_index(feature_clusters):
    """Build unit_id -> rows in pc_features once."""

    print("\nBuilding cluster row index...")

    order = np.argsort(feature_clusters, kind="stable")
    sorted_clusters = feature_clusters[order]

    unit_ids, start_indices, counts = np.unique(
        sorted_clusters,
        return_index=True,
        return_counts=True,
    )

    cluster_rows = {}

    for unit_id, start, count in zip(unit_ids, start_indices, counts):
        cluster_rows[int(unit_id)] = order[start:start + count]

    print("Indexed units:", len(cluster_rows))
    return cluster_rows


# ============================================================
# DOMINANT TEMPLATE / REFERENCE CHANNELS
# ============================================================

def get_dominant_template(unit_rows, feature_templates):
    templates_for_unit = feature_templates[unit_rows]

    template_ids, counts = np.unique(
        templates_for_unit,
        return_counts=True,
    )

    return int(template_ids[np.argmax(counts)])


def get_reference_channels(
    target_template,
    pc_feature_ind,
    templates,
    whitening_mat_inv,
):
    """
    Match the original script / Phy-style logic:
    unwhiten the target template, compute channel PTP, then rank only the
    channels present in pc_feature_ind[target_template].
    """

    candidate_channels = np.asarray(
        pc_feature_ind[target_template],
        dtype=int,
    )

    template_whitened = np.asarray(
        templates[target_template]
    )

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

    return np.asarray(reference_channels, dtype=int)


# ============================================================
# VALID ROWS / COMMON PC MATRIX
# ============================================================

def get_valid_rows(
    rows,
    reference_channels,
    feature_templates,
    pc_feature_ind,
):
    """
    Keep only spikes whose sparse KS template contains ALL target reference
    channels. Missing sparse PC features are never filled with zero.
    """

    if len(rows) == 0:
        return np.array([], dtype=int)

    template_ids = feature_templates[rows]
    valid_mask = np.zeros(len(rows), dtype=bool)

    for template_id in np.unique(template_ids):
        template_mask = template_ids == template_id

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        has_all_channels = np.all(
            np.isin(reference_channels, local_channels)
        )

        if has_all_channels:
            valid_mask[template_mask] = True

    return rows[valid_mask]


def extract_pc_matrix(
    rows,
    reference_channels,
    feature_templates,
    pc_features,
    pc_feature_ind,
):
    """
    Convert Kilosort sparse PC features into a common matrix:

        X.shape = (n_spikes, N_PCS * N_REF_CHANNELS)

    IMPORTANT feature ordering follows the original reshape exactly:
        PC1-ChA, PC1-ChB, ..., PC2-ChA, PC2-ChB, ...
    """

    X = np.empty(
        (
            len(rows),
            N_PCS * len(reference_channels),
        ),
        dtype=np.float32,
    )

    template_ids = feature_templates[rows]

    for template_id in np.unique(template_ids):
        mask = template_ids == template_id
        these_rows = rows[mask]

        local_channels = np.asarray(
            pc_feature_ind[template_id],
            dtype=int,
        )

        local_indices = []

        for channel_id in reference_channels:
            idx = np.flatnonzero(local_channels == channel_id)

            if len(idx) != 1:
                raise RuntimeError(
                    f"Template {template_id}: cannot uniquely find "
                    f"channel {channel_id}"
                )

            local_indices.append(int(idx[0]))

        block = np.asarray(
            pc_features[
                these_rows,
                :N_PCS,
                :,
            ]
        )

        block = block[
            :,
            :,
            local_indices,
        ]

        X[mask] = block.reshape(
            len(these_rows),
            -1,
        )

    return X


def make_feature_names(reference_channels):
    """Names matching extract_pc_matrix() C-order flattening."""

    return [
        f"Ch {int(channel_id)} / PC{pc_index + 1}"
        for pc_index in range(N_PCS)
        for channel_id in reference_channels
    ]


# ============================================================
# BUILD EXACT REVIEW SPACE FOR ONE UNIT
# ============================================================

def build_review_space(
    target_unit_id,
    cluster_rows,
    feature_templates,
    pc_features,
    pc_feature_ind,
    templates,
    whitening_mat_inv,
):
    """
    Build the exact sampled X and labels using the same logic as the metric
    script. This is the key function that makes the figures directly auditable
    against the quantitative PCA-space metrics.
    """

    if target_unit_id not in cluster_rows:
        raise KeyError(f"Unit {target_unit_id} is not present in spike_clusters.")

    rng = np.random.default_rng(
        RANDOM_SEED + int(target_unit_id)
    )

    target_rows_all = cluster_rows[target_unit_id]

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
    source_row_blocks = []
    neighbor_units = []

    for other_unit_id, rows in cluster_rows.items():
        valid_rows = get_valid_rows(
            rows,
            reference_channels,
            feature_templates,
            pc_feature_ind,
        )

        if len(valid_rows) == 0:
            continue

        if len(valid_rows) > MAX_SPIKES_PER_UNIT:
            valid_rows = rng.choice(
                valid_rows,
                size=MAX_SPIKES_PER_UNIT,
                replace=False,
            )

        valid_rows = np.asarray(valid_rows, dtype=int)

        X_unit = extract_pc_matrix(
            valid_rows,
            reference_channels,
            feature_templates,
            pc_features,
            pc_feature_ind,
        )

        X_blocks.append(X_unit)
        y_blocks.append(
            np.full(len(valid_rows), other_unit_id, dtype=int)
        )
        source_row_blocks.append(valid_rows)
        neighbor_units.append(int(other_unit_id))

    if not X_blocks:
        raise RuntimeError("No valid spikes in the selected PC/reference-channel space.")

    X = np.concatenate(X_blocks, axis=0)
    labels = np.concatenate(y_blocks, axis=0)
    source_rows = np.concatenate(source_row_blocks, axis=0)

    target_mask = labels == target_unit_id

    if not np.any(target_mask):
        raise RuntimeError(
            "No valid target spikes after reference-channel filtering."
        )

    if np.all(target_mask):
        raise RuntimeError("No valid background spikes.")

    feature_names = make_feature_names(reference_channels)

    return {
        "X": X,
        "labels": labels,
        "source_rows": source_rows,
        "target_mask": target_mask,
        "dominant_template": dominant_template,
        "reference_channels": reference_channels,
        "feature_names": feature_names,
        "neighbor_units": neighbor_units,
    }


# ============================================================
# LDA / FISHER PROJECTION
# ============================================================

def compute_lda_projection(X, labels, target_unit_id):
    """
    Use the same sklearn LDA construction used by SpikeInterface d_prime_metric:

        y = labels == target_unit_id
        LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)

    Returns the 1D Fisher/LDA projection and coefficients so that the PDF can
    show BOTH the projected distributions and which PC/channel dimensions drive
    the discriminant direction.
    """

    y = labels == target_unit_id

    lda = LinearDiscriminantAnalysis(n_components=1)
    projection = lda.fit_transform(X, y).ravel()

    target_proj = projection[y]
    other_proj = projection[~y]

    denom = np.sqrt(
        (
            np.var(target_proj)
            + np.var(other_proj)
        ) / 2
    )

    if denom > 0 and np.isfinite(denom):
        lda_d_prime = (
            np.mean(target_proj)
            - np.mean(other_proj)
        ) / denom
    else:
        lda_d_prime = np.nan

    # For sklearn's default SVD solver, fit_transform() uses scalings_.
    # Therefore scalings_[:, 0] is the exact input-space direction of the
    # plotted Fisher/LDA component. coef_ is a classifier coefficient and is
    # related, but it is not the safest quantity to label as the projection
    # direction.
    projection_weights = np.asarray(lda.scalings_)[:, 0].reshape(-1)

    # Normalize only for plotting relative weights. This does NOT affect the
    # projection or the SI d-prime calculation.
    max_abs = (
        np.max(np.abs(projection_weights))
        if projection_weights.size
        else np.nan
    )

    if np.isfinite(max_abs) and max_abs > 0:
        projection_weights_relative = projection_weights / max_abs
    else:
        projection_weights_relative = projection_weights.copy()

    return {
        "projection": projection,
        "target_projection": target_proj,
        "other_projection": other_proj,
        "lda_d_prime": float(lda_d_prime),
        "projection_weights": projection_weights,
        "projection_weights_relative": projection_weights_relative,
        "classifier_coef": np.asarray(lda.coef_).reshape(-1),
        "intercept": float(np.asarray(lda.intercept_).reshape(-1)[0]),
    }


# ============================================================
# PAIRWISE 2D METRICS
# ============================================================

def compute_pairwise_metrics(X, labels, target_unit_id, feature_names):
    """
    Compute 2D d-prime and simplified silhouette for each pair of dimensions.

    These are saved separately and are deliberately not written onto the plots
    by default, so a human visual rating can remain blinded.
    """

    rows = []

    for i, j in combinations(range(X.shape[1]), 2):
        X_2d = X[:, [i, j]]

        try:
            pair_d_prime = d_prime_metric(
                X_2d,
                labels,
                target_unit_id,
            )
        except Exception:
            pair_d_prime = np.nan

        try:
            pair_silhouette = simplified_silhouette_score(
                X_2d,
                labels,
                target_unit_id,
            )
        except Exception:
            pair_silhouette = np.nan

        rows.append(
            {
                "unit_id": target_unit_id,
                "feature_i": i,
                "feature_j": j,
                "x_feature": feature_names[i],
                "y_feature": feature_names[j],
                "d_prime_2d": pair_d_prime,
                "silhouette_2d": pair_silhouette,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def deterministic_plot_indices(mask, max_points, seed):
    idx = np.flatnonzero(mask)

    if len(idx) <= max_points:
        return idx

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(
            idx,
            size=max_points,
            replace=False,
        )
    )


def robust_limits(values, percentile):
    if percentile is None:
        return None

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    lo, hi = np.percentile(
        values,
        [percentile, 100 - percentile],
    )

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None

    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def format_number(value, digits=4):
    if value is None:
        return "NA"

    try:
        value = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(value):
        return str(value)

    return f"{value:.{digits}g}"


def add_summary_panel(
    ax,
    target_unit_id,
    info,
    metrics,
    cov_diag,
    pc_quality,
    lda_info,
):
    ax.axis("off")

    lines = [
        f"Unit {target_unit_id}",
        "",
        f"Dominant template: {info['dominant_template']}",
        "Reference channels: "
        + ", ".join(map(str, info["reference_channels"])),
        f"Feature space: {N_PCS} PCs × {len(info['reference_channels'])} channels "
        f"= {info['X'].shape[1]}D",
        f"Target spikes: {np.sum(info['target_mask'])}",
        f"Background spikes: {np.sum(~info['target_mask'])}",
        f"Contributing units: {len(info['neighbor_units'])}",
        "",
        f"Covariance rank: {cov_diag['cov_rank']} / {info['X'].shape[1]}",
        "Condition number: "
        + format_number(cov_diag["cov_condition_number"]),
        f"PC numerical QC: {pc_quality}",
        "",
        "LDA/Fisher projection:",
        "  d' on plotted LDA axis: "
        + format_number(lda_info["lda_d_prime"]),
    ]

    if not BLIND_REVIEW:
        lines.extend(
            [
                "",
                "Automated full-space metrics:",
                "  isolation_distance: "
                + format_number(metrics["isolation_distance"]),
                "  l_ratio: "
                + format_number(metrics["l_ratio"]),
                "  d_prime: "
                + format_number(metrics["d_prime"]),
                "  silhouette: "
                + format_number(metrics["silhouette"]),
                "  nn_hit_rate: "
                + format_number(metrics["nn_hit_rate"]),
                "  nn_miss_rate: "
                + format_number(metrics["nn_miss_rate"]),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "BLIND_REVIEW=True",
                "Full-space separation metrics are hidden",
                "from the PDF and saved in review_summary.csv.",
            ]
        )

    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
    )


def add_lda_histogram(ax, lda_info):
    target_proj = lda_info["target_projection"]
    other_proj = lda_info["other_projection"]

    all_values = np.concatenate([target_proj, other_proj])

    limits = robust_limits(
        all_values,
        PAIRWISE_ROBUST_PERCENTILE,
    )

    if limits is None:
        lo = np.min(all_values)
        hi = np.max(all_values)
    else:
        lo, hi = limits

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1, 1

    bins = np.linspace(lo, hi, HIST_BINS + 1)

    ax.hist(
        other_proj,
        bins=bins,
        density=True,
        alpha=0.45,
        label="other units",
    )
    ax.hist(
        target_proj,
        bins=bins,
        density=True,
        alpha=0.65,
        label="target unit",
    )

    ax.set_title("Fisher / LDA 1D projection")
    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.98,
        "LDA-axis d' = " + format_number(lda_info["lda_d_prime"]),
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=9,
    )


def add_lda_weights(ax, lda_info, feature_names):
    weights = lda_info["projection_weights_relative"]
    y = np.arange(len(feature_names))

    ax.barh(y, weights)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("relative LDA coefficient")
    ax.set_title("Dimensions driving LDA separation")


def add_pairwise_panel(
    ax,
    X,
    target_mask,
    feature_names,
    i,
    j,
    target_plot_idx,
    background_plot_idx,
    limits_by_feature,
):
    ax.scatter(
        X[background_plot_idx, i],
        X[background_plot_idx, j],
        s=PAIRWISE_POINT_SIZE,
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
        rasterized=True,
        label="other units",
    )

    ax.scatter(
        X[target_plot_idx, i],
        X[target_plot_idx, j],
        s=PAIRWISE_POINT_SIZE,
        alpha=TARGET_ALPHA,
        linewidths=0,
        rasterized=True,
        label="target unit",
    )

    ax.set_xlabel(feature_names[i], fontsize=8)
    ax.set_ylabel(feature_names[j], fontsize=8)
    ax.tick_params(labelsize=7)

    if limits_by_feature[i] is not None:
        ax.set_xlim(*limits_by_feature[i])

    if limits_by_feature[j] is not None:
        ax.set_ylim(*limits_by_feature[j])


def make_review_figures(
    target_unit_id,
    info,
    metrics,
    cov_diag,
    pc_quality,
    lda_info,
):
    """
    For 4D (2 PCs × 2 channels), the first page is exactly:
        summary + LDA histogram + LDA weights + all 6 pairwise projections.

    For higher dimensions, the first 6 pairs stay on page 1 and remaining
    pairwise projections spill onto additional pages.
    """

    X = info["X"]
    target_mask = info["target_mask"]
    feature_names = info["feature_names"]

    target_plot_idx = deterministic_plot_indices(
        target_mask,
        MAX_TARGET_POINTS_TO_PLOT,
        RANDOM_SEED + 100000 + int(target_unit_id),
    )

    background_plot_idx = deterministic_plot_indices(
        ~target_mask,
        MAX_BACKGROUND_POINTS_TO_PLOT,
        RANDOM_SEED + 200000 + int(target_unit_id),
    )

    limits_by_feature = [
        robust_limits(
            X[np.concatenate([target_plot_idx, background_plot_idx]), k],
            PAIRWISE_ROBUST_PERCENTILE,
        )
        for k in range(X.shape[1])
    ]

    pairs = list(combinations(range(X.shape[1]), 2))

    figures = []

    # --------------------------------------------------------
    # Overview page: designed so 4D fits in one 3x3 page.
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    fig.suptitle(
        f"Unit {target_unit_id}: PCA-space manual review",
        fontsize=15,
    )

    add_summary_panel(
        axes[0],
        target_unit_id,
        info,
        metrics,
        cov_diag,
        pc_quality,
        lda_info,
    )

    add_lda_histogram(
        axes[1],
        lda_info,
    )

    add_lda_weights(
        axes[2],
        lda_info,
        feature_names,
    )

    first_pairs = pairs[:6]

    for ax, (i, j) in zip(axes[3:], first_pairs):
        add_pairwise_panel(
            ax,
            X,
            target_mask,
            feature_names,
            i,
            j,
            target_plot_idx,
            background_plot_idx,
            limits_by_feature,
        )

    for ax in axes[3 + len(first_pairs):]:
        ax.axis("off")

    # One legend for pairwise panels if any exist.
    if first_pairs:
        axes[3].legend(fontsize=7, loc="best")

    figures.append(fig)

    # --------------------------------------------------------
    # Extra pairwise pages for dimensions > 4.
    # --------------------------------------------------------
    remaining_pairs = pairs[6:]
    pairs_per_page = 12

    for page_start in range(0, len(remaining_pairs), pairs_per_page):
        page_pairs = remaining_pairs[
            page_start:page_start + pairs_per_page
        ]

        ncols = 4
        nrows = math.ceil(len(page_pairs) / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 3.4 * nrows),
            constrained_layout=True,
        )

        axes = np.atleast_1d(axes).ravel()

        fig.suptitle(
            f"Unit {target_unit_id}: additional 2D PCA-space projections",
            fontsize=14,
        )

        for ax, (i, j) in zip(axes, page_pairs):
            add_pairwise_panel(
                ax,
                X,
                target_mask,
                feature_names,
                i,
                j,
                target_plot_idx,
                background_plot_idx,
                limits_by_feature,
            )

        for ax in axes[len(page_pairs):]:
            ax.axis("off")

        if page_pairs:
            axes[0].legend(fontsize=7, loc="best")

        figures.append(fig)

    return figures


# ============================================================
# ONE UNIT
# ============================================================

def process_one_unit(
    target_unit_id,
    cluster_rows,
    feature_templates,
    pc_features,
    pc_feature_ind,
    templates,
    whitening_mat_inv,
    feature_spike_ids,
    combined_pdf=None,
):
    print("\n" + "=" * 78)
    print(f"UNIT {target_unit_id}")
    print("=" * 78)

    info = build_review_space(
        target_unit_id,
        cluster_rows,
        feature_templates,
        pc_features,
        pc_feature_ind,
        templates,
        whitening_mat_inv,
    )

    X = info["X"]
    labels = info["labels"]

    # ============================================================
    # TARGET FEATURE DIAGNOSTICS
    # ============================================================

    X_target = X[info["target_mask"]]

    print("\n" + "=" * 72)
    print(f"TARGET FEATURE DIAGNOSTICS: unit {target_unit_id}")
    print("=" * 72)

    for i, feature_name in enumerate(info["feature_names"]):
        values = X_target[:, i]

        n = len(values)
        n_unique = len(np.unique(values))

        exact_zero_fraction = np.mean(values == 0)
        near_zero_fraction = np.mean(np.abs(values) < 1e-6)

        q01, q25, q50, q75, q99 = np.percentile(
            values,
            [1, 25, 50, 75, 99],
        )

        print(
            f"{feature_name:15s} "
            f"std={np.std(values, ddof=1):9.5f}  "
            f"P01={q01:9.4f}  "
            f"P25={q25:9.4f}  "
            f"P50={q50:9.4f}  "
            f"P75={q75:9.4f}  "
            f"P99={q99:9.4f}  "
            f"unique={n_unique:4d}/{n:<4d}  "
            f"zero={exact_zero_fraction:7.3%}"
        )

    metrics = compute_metrics_for_space(
        X,
        labels,
        target_unit_id,
    )

    cov_diag = compute_covariance_diagnostics(
        X,
        labels,
        target_unit_id,
    )

    pc_quality = (
        "fail"
        if (
            cov_diag["cov_rank"] < PC_QUALITY_RANK_THRESHOLD
            or not np.isfinite(cov_diag["cov_condition_number"])
            or cov_diag["cov_condition_number"] > PC_QUALITY_CONDITION_NUMBER_THRESHOLD
        )
        else "pass"
    )

    lda_info = compute_lda_projection(
        X,
        labels,
        target_unit_id,
    )

    # Numerical check: because both use sklearn LinearDiscriminantAnalysis
    # on the same X / binary target labels, this should match SI d_prime_metric
    # to floating-point precision.
    if np.isfinite(metrics["d_prime"]) and np.isfinite(lda_info["lda_d_prime"]):
        lda_match_error = abs(
            float(metrics["d_prime"])
            - float(lda_info["lda_d_prime"])
        )
    else:
        lda_match_error = np.nan

    summary = {
        "unit_id": target_unit_id,
        "dominant_template": info["dominant_template"],
        "reference_channels": ",".join(
            map(str, info["reference_channels"])
        ),
        "n_pcs": N_PCS,
        "n_channels": len(info["reference_channels"]),
        "n_features": X.shape[1],
        "feature_names": " | ".join(info["feature_names"]),
        "n_neighbor_units": len(info["neighbor_units"]),
        "n_target_spikes": int(np.sum(info["target_mask"])),
        "n_other_spikes": int(np.sum(~info["target_mask"])),
        **metrics,
        **cov_diag,
        "pc_quality": pc_quality,
        "lda_projection_d_prime": lda_info["lda_d_prime"],
        "lda_vs_si_dprime_abs_error": lda_match_error,
        "manual_pc_separation": "",
        "manual_split_candidate": "",
        "manual_merge_candidate": "",
        "manual_notes": "",
        "status": "ok",
    }

    print(pd.Series(summary).to_string())

    # --------------------------------------------------------
    # Optional exact plot-data dump.
    # --------------------------------------------------------
    if SAVE_NPZ_PER_UNIT:
        np.savez_compressed(
            OUTPUT_FOLDER / f"unit_{target_unit_id}_PC_review_data.npz",
            X=X,
            labels=labels,
            source_rows=info["source_rows"],
            original_spike_ids=feature_spike_ids[info["source_rows"]],
            target_mask=info["target_mask"],
            reference_channels=info["reference_channels"],
            feature_names=np.asarray(info["feature_names"], dtype=object),
            lda_projection=lda_info["projection"],
            lda_projection_weights=lda_info["projection_weights"],
            lda_projection_weights_relative=lda_info["projection_weights_relative"],
            lda_classifier_coef=lda_info["classifier_coef"],
        )

    # --------------------------------------------------------
    # Optional 2D projection metrics.
    # --------------------------------------------------------
    if SAVE_PAIRWISE_METRICS:
        pairwise_df = compute_pairwise_metrics(
            X,
            labels,
            target_unit_id,
            info["feature_names"],
        )

        pairwise_df.to_csv(
            OUTPUT_FOLDER / f"unit_{target_unit_id}_pairwise_2D_metrics.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Figures / PDFs.
    # --------------------------------------------------------
    figures = make_review_figures(
        target_unit_id,
        info,
        metrics,
        cov_diag,
        pc_quality,
        lda_info,
    )

    individual_pdf = None

    if SAVE_INDIVIDUAL_PDFS:
        individual_pdf = PdfPages(
            OUTPUT_FOLDER / f"unit_{target_unit_id}_PC_manual_review.pdf"
        )

    try:
        for fig in figures:
            if combined_pdf is not None:
                combined_pdf.savefig(fig, dpi=200)

            if individual_pdf is not None:
                individual_pdf.savefig(fig, dpi=200)

            plt.close(fig)
    finally:
        if individual_pdf is not None:
            individual_pdf.close()

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Loading Kilosort PCA files")
    print("=" * 78)

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

    print("pc_features:", pc_features.shape)
    print("pc_feature_ind:", pc_feature_ind.shape)
    print("spike_templates:", spike_templates.shape)
    print("spike_clusters:", spike_clusters.shape)
    print("templates:", templates.shape)

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
                "pc_features rows != spike count, and "
                "pc_feature_spike_ids.npy was not found."
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

    cluster_rows = build_cluster_index(
        feature_clusters
    )

    available_units = set(cluster_rows.keys())

    missing_units = [
        unit_id
        for unit_id in TARGET_UNITS
        if unit_id not in available_units
    ]

    if missing_units:
        print(
            "[WARN] requested units not present and will be skipped:",
            missing_units,
        )

    units_to_run = [
        int(unit_id)
        for unit_id in TARGET_UNITS
        if unit_id in available_units
    ]

    if not units_to_run:
        raise RuntimeError("None of TARGET_UNITS are present in the sorting.")

    print("\nUnits to review:", units_to_run)

    combined_pdf = None

    if SAVE_COMBINED_PDF:
        combined_pdf = PdfPages(
            OUTPUT_FOLDER / COMBINED_PDF_NAME
        )

    summary_rows = []

    try:
        for target_unit_id in units_to_run:
            try:
                row = process_one_unit(
                    target_unit_id,
                    cluster_rows,
                    feature_templates,
                    pc_features,
                    pc_feature_ind,
                    templates,
                    whitening_mat_inv,
                    feature_spike_ids,
                    combined_pdf=combined_pdf,
                )
            except Exception as exc:
                print(
                    f"[FAILED] unit {target_unit_id}: {exc}"
                )

                row = {
                    "unit_id": target_unit_id,
                    "status": f"failed: {exc}",
                    "manual_pc_separation": "",
                    "manual_split_candidate": "",
                    "manual_merge_candidate": "",
                    "manual_notes": "",
                }

            summary_rows.append(row)

    finally:
        if combined_pdf is not None:
            combined_pdf.close()

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        OUTPUT_FOLDER /
        "review_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Output folder:", OUTPUT_FOLDER)
    print("Summary CSV:", summary_path)

    if SAVE_COMBINED_PDF:
        print(
            "Combined PDF:",
            OUTPUT_FOLDER / COMBINED_PDF_NAME,
        )


if __name__ == "__main__":
    main()
