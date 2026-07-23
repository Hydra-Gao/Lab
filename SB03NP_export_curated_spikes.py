# SB03_export_curated_spikes_phyllum.py

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    PHY_FOLDER,
    SORTER_NAME,
)

# Export only units carrying one of these final Phy/Phyllum labels.
# Change to ("good", "mua") if both are wanted.
EXPORT_GROUPS = ("good",)


def read_sample_rate(params_path: Path) -> float:
    """Read sample_rate from a standard Phy params.py without executing it."""
    if not params_path.exists():
        raise FileNotFoundError(f"params.py not found: {params_path}")

    text = params_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"^\s*sample_rate\s*=\s*([0-9.eE+-]+)",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise ValueError(
            "Could not find a numeric `sample_rate = ...` entry in params.py."
        )

    sample_rate = float(match.group(1))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in params.py: {sample_rate}")

    return sample_rate


def load_cluster_info(info_path: Path) -> tuple[pd.DataFrame, str]:
    """Load cluster_info.tsv and identify its cluster-ID column."""
    if not info_path.exists():
        raise FileNotFoundError(f"cluster_info.tsv not found: {info_path}")

    info = pd.read_csv(info_path, sep="\t")

    if "cluster_id" in info.columns:
        id_col = "cluster_id"
    elif "id" in info.columns:
        id_col = "id"
    else:
        raise ValueError(
            "cluster_info.tsv must contain either `cluster_id` or `id`."
        )

    if "group" not in info.columns:
        raise ValueError(
            "cluster_info.tsv has no `group` column, so final curated labels "
            "cannot be selected."
        )

    info[id_col] = pd.to_numeric(info[id_col], errors="raise").astype(np.int64)
    info["group"] = info["group"].astype(str).str.strip().str.lower()

    if info[id_col].duplicated().any():
        duplicate_ids = info.loc[info[id_col].duplicated(False), id_col].tolist()
        raise ValueError(
            f"cluster_info.tsv contains duplicate cluster IDs: {duplicate_ids[:20]}"
        )

    return info, id_col


def load_phy_spike_arrays(phy_folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load final Phy/Phyllum spike sample indices and cluster assignments."""
    spike_times_path = phy_folder / "spike_times.npy"
    spike_clusters_path = phy_folder / "spike_clusters.npy"

    if not spike_times_path.exists():
        raise FileNotFoundError(f"spike_times.npy not found: {spike_times_path}")
    if not spike_clusters_path.exists():
        raise FileNotFoundError(f"spike_clusters.npy not found: {spike_clusters_path}")

    spike_frames = np.asarray(np.load(spike_times_path, mmap_mode="r")).reshape(-1)
    spike_clusters = np.asarray(
        np.load(spike_clusters_path, mmap_mode="r")
    ).reshape(-1)

    if len(spike_frames) != len(spike_clusters):
        raise ValueError(
            "spike_times.npy and spike_clusters.npy have different lengths: "
            f"{len(spike_frames)} vs {len(spike_clusters)}"
        )

    if not np.issubdtype(spike_frames.dtype, np.integer):
        if np.any(spike_frames != np.floor(spike_frames)):
            raise ValueError("spike_times.npy contains non-integer sample indices.")

    spike_frames = spike_frames.astype(np.int64, copy=False)
    spike_clusters = spike_clusters.astype(np.int64, copy=False)

    if np.any(spike_frames < 0):
        raise ValueError("spike_times.npy contains negative sample indices.")

    return spike_frames, spike_clusters


def optional_numeric(row: pd.Series, column: str) -> float:
    """Return one optional numeric cluster-info value, otherwise NaN."""
    if column not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def main() -> None:
    print("===== Export final Phy/Phyllum curated spikes =====")

    phy_folder = Path(PHY_FOLDER)
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params_path = phy_folder / "params.py"
    info_path = phy_folder / "cluster_info.tsv"

    sample_rate = read_sample_rate(params_path)
    cluster_info, id_col = load_cluster_info(info_path)
    spike_frames, spike_clusters = load_phy_spike_arrays(phy_folder)

    print(f"Phy/Phyllum folder: {phy_folder}")
    print(f"Sampling frequency: {sample_rate:.9f} Hz")
    print(f"Total spikes in final Phy files: {len(spike_frames)}")
    print(f"Cluster-info rows: {len(cluster_info)}")
    print(f"Cluster ID column: {id_col}")

    selected_info = cluster_info.loc[
        cluster_info["group"].isin(EXPORT_GROUPS)
    ].copy()

    selected_units = selected_info[id_col].to_numpy(dtype=np.int64)
    selected_unit_set = set(selected_units.tolist())

    print(f"Export groups: {EXPORT_GROUPS}")
    print(f"Selected units: {len(selected_units)}")

    if len(selected_units) == 0:
        available_groups = sorted(cluster_info["group"].dropna().unique().tolist())
        raise ValueError(
            f"No units matched EXPORT_GROUPS={EXPORT_GROUPS}. "
            f"Available group values: {available_groups}"
        )

    # Validate that selected cluster IDs actually occur in spike_clusters.npy.
    clusters_in_spikes = set(np.unique(spike_clusters).tolist())
    missing_spike_clusters = sorted(selected_unit_set - clusters_in_spikes)
    if missing_spike_clusters:
        print(
            "Warning: selected cluster IDs absent from spike_clusters.npy: "
            f"{missing_spike_clusters[:20]}"
        )

    # Keep only spikes assigned to the selected final curated clusters.
    selected_mask = np.isin(spike_clusters, selected_units)

    selected_frames = spike_frames[selected_mask]
    selected_clusters = spike_clusters[selected_mask]

    spikes = pd.DataFrame(
        {
            "unit_id": selected_clusters,
            "spike_frame": selected_frames,
            "spike_time_sec": selected_frames.astype(np.float64) / sample_rate,
            "sorter": SORTER_NAME,
        }
    )

    spikes = spikes.sort_values(
        ["spike_time_sec", "unit_id"],
        kind="stable",
    ).reset_index(drop=True)

    spike_counts = spikes.groupby("unit_id").size().rename("n_spikes")

    # Build unit metadata directly from final cluster_info.tsv.
    unit_rows = []

    for _, row in selected_info.iterrows():
        unit_id = int(row[id_col])

        unit_rows.append(
            {
                "unit_id": unit_id,
                "sorter": SORTER_NAME,
                "phy_group": row["group"],
                "n_spikes": int(spike_counts.get(unit_id, 0)),
                # Phy/Phyllum metadata. These are no longer analyzer-derived.
                "shank_id": optional_numeric(row, "sh"),
                "best_channel": optional_numeric(row, "ch"),
                "depth_um": optional_numeric(row, "depth"),
                # Preserve x if the installed Phyllum version supplies it.
                "x_um": optional_numeric(row, "x"),
            }
        )

    curated_units = pd.DataFrame(unit_rows).sort_values("unit_id").reset_index(drop=True)

    # Cross-check exported counts against the selected spike array.
    if int(curated_units["n_spikes"].sum()) != len(spikes):
        raise RuntimeError(
            "Internal count mismatch between spikes.csv and curated_units.csv."
        )

    spikes_path = ANALYSIS_OUTPUT_DIR / "spikes.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    spikes.to_csv(spikes_path, index=False)
    curated_units.to_csv(units_path, index=False)

    print("\n===== Saved =====")
    print(spikes_path)
    print(units_path)
    print(f"Exported spikes: {len(spikes)}")
    print(f"Exported units: {len(curated_units)}")

    print("\nCurated units:")
    print(curated_units.head())

    print("\nFirst few spikes:")
    print(spikes.head())


if __name__ == "__main__":
    main()
