from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

KILOSORT_FOLDER = Path(r"C:\Lab\Processing\TG963_site1-2")
OUTPUT_DIR = KILOSORT_FOLDER / "ks_pc_feature_plot_test"

TARGET_UNIT_ID = 100
PLOT_CHANNELS = [44, 42]      # exactly the two channels shown in your Phy screenshot
PC_INDICES = [0, 1]          # A = PC0, B = PC1

MAX_TARGET_PLOT = 2500
MAX_BACKGROUND_PLOT = 4000
RANDOM_SEED = 42

# If ops.npy cannot provide fs, use this fallback.
SAMPLE_RATE_FALLBACK = 30000.0

# Produce BOTH versions using exactly the same sampled spikes:
#   NaN  : sparse channel not present -> not plotted
#   zero : sparse channel not present -> 0, which can create axis-aligned stripes
MAKE_NAN_FILL_PLOT = True
MAKE_ZERO_FILL_PLOT = True

# Keep this False for a clean comparison to your existing Matplotlib SI plot.
# If True, only colors change; geometry is identical.
USE_PHY_LIKE_COLORS = True


# ============================================================
# HELPERS
# ============================================================

def load_sample_rate(folder: Path) -> float:
    ops_path = folder / "ops.npy"
    if ops_path.exists():
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
            if "fs" in ops:
                return float(ops["fs"])
        except Exception as exc:
            print(f"[WARN] Could not read fs from ops.npy: {exc}")
    return SAMPLE_RATE_FALLBACK


def subsample_indices(indices, max_n, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= max_n:
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=max_n, replace=False))


def sample_background_indices(spike_clusters, target_unit_id, max_n, rng):
    """Memory-light random background sampling from all non-target spikes."""
    n_spikes = len(spike_clusters)
    selected = np.empty(0, dtype=np.int64)

    # Usually target spikes are a tiny fraction of all spikes, so this converges fast.
    while len(selected) < max_n:
        need = max_n - len(selected)
        draw_n = max(need * 3, 1000)
        cand = rng.integers(0, n_spikes, size=draw_n, endpoint=False)
        cand = cand[np.asarray(spike_clusters[cand]) != target_unit_id]
        if len(cand) == 0:
            continue
        selected = np.unique(np.concatenate([selected, cand]))

    if len(selected) > max_n:
        selected = rng.choice(selected, size=max_n, replace=False)

    return np.sort(selected.astype(np.int64))


def extract_ks_features(
    pc_features,
    pc_feature_ind,
    spike_templates,
    spike_indices,
    plot_channels,
    pc_indices,
    fill_value=np.nan,
):
    """
    Extract Kilosort PC features on specific physical channels.

    Uses ONLY Kilosort arrays:
      pc_features.npy
      pc_feature_ind.npy
      spike_templates.npy

    For each spike, pc_feature_ind[spike_template] tells which physical channels
    correspond to the sparse channel axis of pc_features for that spike.

    No Phy plotting/model code is used here.
    """
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    templates = np.asarray(spike_templates[spike_indices], dtype=np.int64)

    n_rows = len(spike_indices)
    n_channels = len(plot_channels)
    n_pcs = len(pc_indices)

    # Feature order will be:
    # [ch1-PC0, ch2-PC0, ch1-PC1, ch2-PC1, ...]
    out = np.full(
        (n_rows, n_channels * n_pcs),
        fill_value,
        dtype=np.float32,
    )
    present = np.zeros((n_rows, n_channels), dtype=bool)

    # Vectorize by template because spikes sharing a template share pc_feature_ind.
    for template_id in np.unique(templates):
        row_mask = templates == template_id
        row_positions = np.flatnonzero(row_mask)
        ks_channel_list = np.asarray(pc_feature_ind[template_id], dtype=np.int64)

        for channel_offset, physical_channel in enumerate(plot_channels):
            matches = np.flatnonzero(ks_channel_list == physical_channel)
            if len(matches) == 0:
                continue

            sparse_channel_pos = int(matches[0])
            present[row_positions, channel_offset] = True

            source_spike_indices = spike_indices[row_positions]

            for pc_offset, pc_index in enumerate(pc_indices):
                dest_col = pc_offset * n_channels + channel_offset
                out[row_positions, dest_col] = np.asarray(
                    pc_features[
                        source_spike_indices,
                        pc_index,
                        sparse_channel_pos,
                    ],
                    dtype=np.float32,
                )

    return out, present


def build_feature_dict(matrix, plot_channels, pc_indices):
    names = []
    features = {}

    for pc_offset, pc_index in enumerate(pc_indices):
        letter = chr(ord("A") + pc_offset)
        for channel_offset, channel in enumerate(plot_channels):
            name = f"{channel}{letter}"
            col = pc_offset * len(plot_channels) + channel_offset
            names.append(name)
            features[name] = matrix[:, col]

    return names, features


def plot_phy_layout(
    matrix,
    times_sec,
    is_target,
    plot_channels,
    pc_indices,
    title,
    output_file,
):
    """4x4 Phy-style matrix layout, but rendered only with Matplotlib."""
    names, features = build_feature_dict(matrix, plot_channels, pc_indices)

    if len(names) != 4:
        raise ValueError(
            "This plotting layout expects exactly 2 channels x 2 PCs = 4 features."
        )

    target = np.flatnonzero(is_target)
    background = np.flatnonzero(~is_target)

    fig, axes = plt.subplots(4, 4, figsize=(10, 8.5))
    fig.suptitle(title, fontsize=13)

    if USE_PHY_LIKE_COLORS:
        bg_kwargs = dict(s=3, alpha=0.30, c="0.55")
        target_kwargs = dict(s=3, alpha=0.85, c="#0096ff")
    else:
        # Same visual convention as your current SI diagnostic plot.
        bg_kwargs = dict(s=3, alpha=0.25)
        target_kwargs = dict(s=3, alpha=0.75)

    for i, y_name in enumerate(names):
        for j, x_name in enumerate(names):
            ax = axes[i, j]
            y = features[y_name]

            if i == j:
                ax.scatter(times_sec[background], y[background], **bg_kwargs)
                ax.scatter(times_sec[target], y[target], **target_kwargs)
                ax.set_xlabel("time")
            else:
                x = features[x_name]
                ax.scatter(x[background], y[background], **bg_kwargs)
                ax.scatter(x[target], y[target], **target_kwargs)
                ax.set_xlabel(x_name)

            ax.set_ylabel(y_name)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_missingness_table(
    target_present,
    background_present,
    plot_channels,
):
    rows = []
    for channel_offset, channel in enumerate(plot_channels):
        rows.append(
            {
                "channel": int(channel),
                "target_n": int(len(target_present)),
                "target_feature_present_n": int(target_present[:, channel_offset].sum()),
                "target_feature_present_fraction": float(
                    target_present[:, channel_offset].mean()
                ),
                "background_n": int(len(background_present)),
                "background_feature_present_n": int(
                    background_present[:, channel_offset].sum()
                ),
                "background_feature_present_fraction": float(
                    background_present[:, channel_offset].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Kilosort arrays from:")
    print(KILOSORT_FOLDER)

    # memmap pc_features because this file can be several GB.
    pc_features = np.load(KILOSORT_FOLDER / "pc_features.npy", mmap_mode="r")
    pc_feature_ind = np.load(KILOSORT_FOLDER / "pc_feature_ind.npy", mmap_mode="r")
    spike_templates = np.load(KILOSORT_FOLDER / "spike_templates.npy", mmap_mode="r")
    spike_clusters = np.load(KILOSORT_FOLDER / "spike_clusters.npy", mmap_mode="r")
    spike_times = np.load(KILOSORT_FOLDER / "spike_times.npy", mmap_mode="r")

    # Kilosort arrays are usually (n_spikes, 1) for these vectors; flatten safely.
    spike_templates = spike_templates.reshape(-1)
    spike_clusters = spike_clusters.reshape(-1)
    spike_times = spike_times.reshape(-1)

    print("pc_features:", pc_features.shape)
    print("pc_feature_ind:", pc_feature_ind.shape)
    print("spike_templates:", spike_templates.shape)
    print("spike_clusters:", spike_clusters.shape)
    print("spike_times:", spike_times.shape)

    n_spikes = pc_features.shape[0]
    if not (
        len(spike_templates) == n_spikes
        and len(spike_clusters) == n_spikes
        and len(spike_times) == n_spikes
    ):
        raise RuntimeError(
            "Kilosort arrays do not have matching spike counts. "
            f"pc_features={n_spikes}, spike_templates={len(spike_templates)}, "
            f"spike_clusters={len(spike_clusters)}, spike_times={len(spike_times)}"
        )

    if max(PC_INDICES) >= pc_features.shape[1]:
        raise ValueError(
            f"Requested PC_INDICES={PC_INDICES}, but pc_features only has "
            f"{pc_features.shape[1]} PCs."
        )

    target_all = np.flatnonzero(np.asarray(spike_clusters) == TARGET_UNIT_ID)
    if len(target_all) == 0:
        raise ValueError(f"Target unit {TARGET_UNIT_ID} has no spikes.")

    rng = np.random.default_rng(RANDOM_SEED)
    target_idx = subsample_indices(target_all, MAX_TARGET_PLOT, rng)
    background_idx = sample_background_indices(
        spike_clusters,
        TARGET_UNIT_ID,
        MAX_BACKGROUND_PLOT,
        rng,
    )

    # Same spike sample is used for every diagnostic plot.
    selected_idx = np.concatenate([background_idx, target_idx])
    is_target = np.concatenate(
        [
            np.zeros(len(background_idx), dtype=bool),
            np.ones(len(target_idx), dtype=bool),
        ]
    )

    fs = load_sample_rate(KILOSORT_FOLDER)
    times_sec = np.asarray(spike_times[selected_idx], dtype=np.float64) / fs

    target_templates = np.asarray(spike_templates[target_idx], dtype=np.int64)
    vals, counts = np.unique(target_templates, return_counts=True)
    dominant_template = int(vals[np.argmax(counts)])

    print("\nTarget unit:", TARGET_UNIT_ID)
    print("Target spikes total:", len(target_all))
    print("Target spikes plotted:", len(target_idx))
    print("Background spikes plotted:", len(background_idx))
    print("Dominant target template:", dominant_template)
    print("Dominant template pc_feature_ind:", pc_feature_ind[dominant_template])
    print("Requested plot channels:", PLOT_CHANNELS)
    print("Sampling rate:", fs)

    # --------------------------------------------------------
    # Extract once with NaN for genuinely absent sparse features.
    # --------------------------------------------------------
    matrix_nan, present = extract_ks_features(
        pc_features=pc_features,
        pc_feature_ind=pc_feature_ind,
        spike_templates=spike_templates,
        spike_indices=selected_idx,
        plot_channels=PLOT_CHANNELS,
        pc_indices=PC_INDICES,
        fill_value=np.nan,
    )

    bg_present = present[: len(background_idx)]
    target_present = present[len(background_idx) :]

    missingness_df = make_missingness_table(
        target_present=target_present,
        background_present=bg_present,
        plot_channels=PLOT_CHANNELS,
    )
    missingness_csv = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_KS_pc_channel_presence.csv"
    missingness_df.to_csv(missingness_csv, index=False)

    print("\nSparse-feature channel presence:")
    print(missingness_df.to_string(index=False))
    print("Saved:", missingness_csv)

    # --------------------------------------------------------
    # Plot 1: missing sparse channel -> NaN / omitted
    # --------------------------------------------------------
    if MAKE_NAN_FILL_PLOT:
        out_nan = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_KS_pc_phy_layout_nan_fill.png"
        plot_phy_layout(
            matrix=matrix_nan,
            times_sec=times_sec,
            is_target=is_target,
            plot_channels=PLOT_CHANNELS,
            pc_indices=PC_INDICES,
            title=(
                f"Kilosort PC features, Matplotlib Phy layout | unit {TARGET_UNIT_ID} | "
                f"channels {PLOT_CHANNELS} | missing=NaN"
            ),
            output_file=out_nan,
        )
        print("Saved:", out_nan)

    # --------------------------------------------------------
    # Plot 2: same exact values/spikes, only absent sparse dims -> 0
    # --------------------------------------------------------
    if MAKE_ZERO_FILL_PLOT:
        matrix_zero = np.nan_to_num(matrix_nan, nan=0.0)
        out_zero = OUTPUT_DIR / f"unit_{TARGET_UNIT_ID}_KS_pc_phy_layout_zero_fill.png"
        plot_phy_layout(
            matrix=matrix_zero,
            times_sec=times_sec,
            is_target=is_target,
            plot_channels=PLOT_CHANNELS,
            pc_indices=PC_INDICES,
            title=(
                f"Kilosort PC features, Matplotlib Phy layout | unit {TARGET_UNIT_ID} | "
                f"channels {PLOT_CHANNELS} | missing=0"
            ),
            output_file=out_zero,
        )
        print("Saved:", out_zero)

    print("\nInterpretation:")
    print("1) If KS+Matplotlib already looks like Phy, the difference is NOT Phy's renderer.")
    print("2) If NaN-fill looks compact but zero-fill creates axis stripes, the stripes come from sparse-feature missing-value handling.")
    print("3) If both KS versions differ from SI, the main difference is the underlying KS-vs-SI feature space / sampling, not plotting.")
    print("\nDone.")


if __name__ == "__main__":
    main()