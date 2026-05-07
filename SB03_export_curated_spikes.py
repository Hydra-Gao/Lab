# 03_export_curated_spikes.py

from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    CURATED_SORTING_FOLDER,
    PHY_FOLDER,
    SORTER_NAME,
    SAMPLING_FREQUENCY,
)


def load_phy_good_units(phy_folder: Path):
    """Read good units manually labeled in Phy."""
    group_path = phy_folder / "cluster_group.tsv"

    if not group_path.exists():
        print("cluster_group.tsv not found. Exporting all units instead.")
        return None

    groups = pd.read_csv(group_path, sep="\t")

    if "cluster_id" not in groups.columns or "group" not in groups.columns:
        raise ValueError("cluster_group.tsv must contain cluster_id and group columns.")

    good_units = groups.loc[groups["group"] == "good", "cluster_id"].tolist()

    print(f"Good units from Phy: {good_units}")

    return good_units


def main():
    print("===== Export curated spikes =====")

    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sorting = si.load(CURATED_SORTING_FOLDER)

    print("Loaded curated sorting:")
    print(sorting)

    all_units = list(sorting.get_unit_ids())
    good_units = load_phy_good_units(PHY_FOLDER)

    if good_units is None or len(good_units) == 0:
        selected_units = all_units
        print("Using all curated units.")
    else:
        # Keep only units that actually exist in the curated sorting
        selected_units = [u for u in good_units if u in all_units]

        missing = [u for u in good_units if u not in all_units]
        if missing:
            print(f"Warning: these Phy good units are not in sorting: {missing}")

    print(f"Selected units: {selected_units}")

    # -----------------------------
    # Export spikes.csv
    # -----------------------------

    spike_rows = []

    for unit_id in selected_units:
        spike_frames = sorting.get_unit_spike_train(unit_id=unit_id)

        for frame in spike_frames:
            spike_rows.append(
                {
                    "unit_id": unit_id,
                    "spike_frame": int(frame),
                    "spike_time_sec": float(frame) / SAMPLING_FREQUENCY,
                    "sorter": SORTER_NAME,
                }
            )

    spikes = pd.DataFrame(spike_rows)

    if len(spikes) > 0:
        spikes = spikes.sort_values(["spike_time_sec", "unit_id"]).reset_index(drop=True)

    # -----------------------------
    # Export curated_units.csv
    # -----------------------------

    unit_rows = []

    for unit_id in selected_units:
        n_spikes = len(sorting.get_unit_spike_train(unit_id=unit_id))

        unit_rows.append(
            {
                "unit_id": unit_id,
                "sorter": SORTER_NAME,
                "phy_group": "good" if good_units is not None and unit_id in good_units else "curated",
                "n_spikes": n_spikes,
            }
        )

    curated_units = pd.DataFrame(unit_rows)

    # -----------------------------
    # Save
    # -----------------------------

    spikes_path = ANALYSIS_OUTPUT_DIR / "spikes.csv"
    units_path = ANALYSIS_OUTPUT_DIR / "curated_units.csv"

    spikes.to_csv(spikes_path, index=False)
    curated_units.to_csv(units_path, index=False)

    print("\n===== Saved =====")
    print(spikes_path)
    print(units_path)

    print("\nCurated units:")
    print(curated_units)

    print("\nFirst few spikes:")
    print(spikes.head())


if __name__ == "__main__":
    main()