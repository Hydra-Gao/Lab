import spikeinterface.full as si
from spikeinterface.curation import CurationSorting, remove_duplicated_spikes
from multiprocessing import freeze_support
from config_local import WORKING_DIR
import pandas as pd
from pathlib import Path


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
        # {"unit_ids": [9, 19], "new_unit_id": 9019},

    ],
    "mountainsort5": [
        # {"unit_ids": [1, 5], "new_unit_id": 105},
        # {"unit_ids": [17, 18], "new_unit_id": 17018},
        # {"unit_ids": [17, 24], "new_unit_id": 17024}
    ],
}

# Set enabled=True only when you really want to apply it.
REMOVE_DUPLICATE_SETTINGS = {
    "kilosort4": {
        "enabled": True,
        "censored_period_ms": 0.1,
        "method": "keep_first",
    },
    "mountainsort5": {
        "enabled": True,
        "censored_period_ms": 0.1,
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

        """ print("\n===== Agreement scores =====")
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
        ) """
        
        comparison_output_folder = WORKING_DIR / "comparison_M12_curated_report"
        save_comparison_report(
            cmp=cmp,
            name1=name1,
            name2=name2,
            output_folder=comparison_output_folder,
            threshold=0.5,
        )

    # If later you add more sorters, this still works.
    else:
        comparison = si.compare_multiple_sorters(
            sorting_list=list(sortings.values()),
            name_list=sorter_names,
        )
        print(comparison)

def save_comparison_report(cmp, name1, name2, output_folder, threshold):
    """Save comparison outputs into readable txt/csv files."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. Full agreement matrix
    agreement_df = cmp.agreement_scores
    agreement_df.to_csv(output_folder / "agreement_scores.csv")

    # 2. High agreement pairs table
    pairs = []
    for unit1 in agreement_df.index:
        for unit2 in agreement_df.columns:
            score = agreement_df.loc[unit1, unit2]
            if score >= threshold:
                pairs.append({
                    f"{name1}_unit": unit1,
                    f"{name2}_unit": unit2,
                    "agreement_score": score,
                })

    high_pairs_df = pd.DataFrame(pairs)

    if len(high_pairs_df) > 0:
        high_pairs_df = high_pairs_df.sort_values(
            by="agreement_score",
            ascending=False,
        )

    # 3. Human-readable summary
    with open(output_folder / "summary.txt", "w", encoding="utf-8") as f:
        f.write("===== CURATED SORTER COMPARISON SUMMARY =====\n\n")
        f.write(f"Sorter 1: {name1}\n")
        f.write(f"Sorter 2: {name2}\n")
        f.write(f"Agreement threshold: {threshold}\n\n")

        f.write("===== Unit counts =====\n")
        f.write(f"{name1}: {len(agreement_df.index)} units\n")
        f.write(f"{name2}: {len(agreement_df.columns)} units\n\n")

        f.write("===== High agreement pairs =====\n")
        if len(high_pairs_df) == 0:
            f.write("No pairs above threshold.\n\n")
        else:
            for _, row in high_pairs_df.iterrows():
                f.write(
                    f"{name1} unit {row[f'{name1}_unit']} "
                    f"<-> {name2} unit {row[f'{name2}_unit']} "
                    f": agreement = {row['agreement_score']:.4f}\n"
                )
            f.write("\n")

        f.write(f"===== Best matches: {name1} -> {name2} =====\n")
        f.write(str(cmp.best_match_12))
        f.write("\n\n")

        f.write(f"===== Best matches: {name2} -> {name1} =====\n")
        f.write(str(cmp.best_match_21))
        f.write("\n\n")

        f.write(f"===== Possible matches: {name1} -> {name2} =====\n")
        f.write(str(cmp.possible_match_12))
        f.write("\n\n")

        f.write(f"===== Possible matches: {name2} -> {name1} =====\n")
        f.write(str(cmp.possible_match_21))
        f.write("\n\n")

        f.write(f"===== Hungarian match: {name1} -> {name2} =====\n")
        f.write(str(cmp.hungarian_match_12))
        f.write("\n\n")

        f.write(f"===== Hungarian match: {name2} -> {name1} =====\n")
        f.write(str(cmp.hungarian_match_21))
        f.write("\n")

    print(f"\nComparison report saved to: {output_folder}")

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
