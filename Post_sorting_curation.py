import spikeinterface.full as si
from spikeinterface.curation import CurationSorting, remove_duplicated_spikes
from multiprocessing import freeze_support
from config_local import WORKING_DIR


# =====================
# Global settings
# =====================

SORTER_FOLDERS = {
    "kilosort4": WORKING_DIR / f"kilosort4_M12_test_output",
    "mountainsort5": WORKING_DIR / f"mountainsort5_M12_test_output",
}

CURATED_SORTING_FOLDERS = {
    name: WORKING_DIR / f"sorting_M12_{name}_curated"
    for name in SORTER_FOLDERS
}

# Put sorter-specific manual merges here.
# Use Phy only to inspect possible merges, then record the final decisions here.
MANUAL_MERGES = {
    "kilosort4": [
        # {"unit_ids": [3, 7], "new_unit_id": 307},
        {"unit_ids": [8, 18], "new_unit_id": 8018},

    ],
    "mountainsort5": [
        # {"unit_ids": [1, 5], "new_unit_id": 105},
    ],
}

# Set enabled=True only when you really want to apply it.
REMOVE_DUPLICATE_SETTINGS = {
    "kilosort4": {
        "enabled": False,
        "censored_period_ms": 0.25,
        "method": "keep_first",
    },
    "mountainsort5": {
        "enabled": False,
        "censored_period_ms": 0.25,
        "method": "keep_first",
    },
}


def apply_manual_merges(sorting, sorter_name):
    """Apply the manual merge decisions for one sorter."""
    merges = MANUAL_MERGES.get(sorter_name, [])

    if len(merges) == 0:
        print("No manual merges applied.")
        return sorting

    print(f"\nApplying manual merges for {sorter_name}")

    cs = CurationSorting(sorting)

    for merge in merges:
        unit_ids = merge["unit_ids"]
        new_unit_id = merge["new_unit_id"]

        print(f"Merging units {unit_ids} -> {new_unit_id}")
        cs.merge(unit_ids, new_unit_id=new_unit_id)

    sorting = cs.sorting

    print("After manual merge:")
    print(sorting)
    print("Spike counts after merge:")
    print(sorting.count_num_spikes_per_unit())

    return sorting


def apply_duplicate_removal(sorting, sorter_name):
    """Optionally remove duplicated spikes for one sorter."""
    dup_cfg = REMOVE_DUPLICATE_SETTINGS.get(sorter_name, {})

    if not dup_cfg.get("enabled", False):
        print("Duplicate spike removal disabled.")
        return sorting

    print(f"\nRemoving duplicated spikes for {sorter_name}")

    sorting = remove_duplicated_spikes(
        sorting,
        censored_period_ms=dup_cfg["censored_period_ms"],
        method=dup_cfg["method"],
    )

    print("After removing duplicated spikes:")
    print(sorting)
    print("Cleaned spike counts:")
    print(sorting.count_num_spikes_per_unit())

    return sorting


def save_curated_sorting(sorting, sorter_name):
    """Save curated sorting so Export_phy.py can load it later."""
    output_folder = CURATED_SORTING_FOLDERS[sorter_name]

    print(f"\nSaving curated sorting for {sorter_name} to:")
    print(output_folder)

    # This creates a SpikeInterface-loadable folder.
    sorting_saved = sorting.save(
        folder=output_folder,
        overwrite=True,
    )

    print("Saved curated sorting:")
    print(sorting_saved)

    return sorting_saved


def compare_curated_sorters(sortings):
    """Compare curated sorter outputs after merge / duplicate removal."""
    if len(sortings) < 2:
        print("\nNot enough valid sorters for comparison.")
        return

    print("\n===== Comparing curated sorters =====")

    sorter_names = list(sortings.keys())

    # For your current two-sorter setup, this gives the most readable output.
    if len(sorter_names) == 2:
        name1, name2 = sorter_names
        cmp = si.compare_two_sorters(
            sorting1=sortings[name1],
            sorting2=sortings[name2],
            sorting1_name=name1,
            sorting2_name=name2,
        )

        print("\n===== Agreement scores =====")
        print(cmp.agreement_scores)

        print(f"\n===== Best matches: {name1} -> {name2} =====")
        print(cmp.best_match_12)

        print(f"\n===== Best matches: {name2} -> {name1} =====")
        print(cmp.best_match_21)

        print(f"\n===== Possible matches: {name1} -> {name2} =====")
        print(cmp.possible_match_12)

        print(f"\n===== Possible matches: {name2} -> {name1} =====")
        print(cmp.possible_match_21)

        print(f"\n===== Hungarian match: {name1} -> {name2} =====")
        print(cmp.hungarian_match_12)

        print(f"\n===== Hungarian match: {name2} -> {name1} =====")
        print(cmp.hungarian_match_21)

        cmp.agreement_scores.to_csv(
            WORKING_DIR / f"comparison_agreement_scores_M12_curated.csv"
        )

    # If later you add more sorters, this still works.
    else:
        comparison = si.compare_multiple_sorters(
            sorting_list=list(sortings.values()),
            name_list=sorter_names,
        )
        print(comparison)


def main():
    sortings = {}

    print("Sorter list to process:")
    for name, folder in SORTER_FOLDERS.items():
        print(name, folder)

    for sorter_name, sorter_folder in SORTER_FOLDERS.items():
        print(f"\n===== Processing {sorter_name} =====")

        if not sorter_folder.exists():
            print(f"Skipping {sorter_name}: folder not found")
            continue

        # 1. Load raw sorter output.
        sorting = si.read_sorter_folder(sorter_folder)

        print("Original sorting:")
        print(sorting)
        print("Original spike counts:")
        print(sorting.count_num_spikes_per_unit())

        # 2. Curate sorting-level results.
        sorting = apply_manual_merges(sorting, sorter_name)
        sorting = apply_duplicate_removal(sorting, sorter_name)

        # 3. Save curated sorting for later analyzer / Phy export.
        sorting = save_curated_sorting(sorting, sorter_name)
        sortings[sorter_name] = sorting

    # 4. Compare curated sorting results.
    compare_curated_sorters(sortings)


if __name__ == "__main__":
    freeze_support()
    main()
