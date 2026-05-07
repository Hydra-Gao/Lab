import spikeinterface.full as si
from spikeinterface.curation import CurationSorting, remove_duplicated_spikes
from config_local import OUTPUT_DIR, WORKING_DIR


def main():
    # =====================
    # Paths
    # =====================

    preprocessed_folder = WORKING_DIR / "preprocessed_M12"

    sorter_folders = {
        "kilosort4": WORKING_DIR / "kilosort4_M12_test_output",
        "mountainsort5": WORKING_DIR / "mountainsort5_M12_test_output",
    }

    # =====================
    # Manual merge settings
    # =====================
    # Put sorter-specific manual merges here.

    manual_merges = {
        "kilosort4": [
            # {"unit_ids": [3, 7], "new_unit_id": 307},
        ],
        "mountainsort5": [
            # {"unit_ids": [1, 5], "new_unit_id": 105},
        ],
    }

    # =====================
    # Duplicate spike removal settings
    # =====================
    # Set enabled=True only when you really want to apply it.

    remove_duplicate_settings = {
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

    # =====================
    # Load recording
    # =====================

    recording_saved = si.load(preprocessed_folder)

    # Your sorter outputs are 1 segment, so use the first segment only.
    recording_test = recording_saved.select_segments([0])

    sortings = {}

    # =====================
    # Check sorter folders
    # =====================

    print("Sorter list to process:")
    for name, folder in sorter_folders.items():
        print(name, folder)

    # =====================
    # Process each sorter
    # =====================

    for sorter_name, sorter_folder in sorter_folders.items():
        print(f"\n===== Processing {sorter_name} =====")

        if not sorter_folder.exists():
            print(f"Skipping {sorter_name}: folder not found")
            continue

        # ---------------------
        # 1. Load sorter output
        # ---------------------

        sorting = si.read_sorter_folder(sorter_folder)

        print("Original sorting:")
        print(sorting)
        print("Original spike counts:")
        print(sorting.count_num_spikes_per_unit())

        # ---------------------
        # 2. Manual merge
        # ---------------------
        # This happens BEFORE analyzer creation.
        # After this point, sorting may contain new merged unit IDs.

        merges = manual_merges.get(sorter_name, [])

        if len(merges) > 0:
            print(f"\nApplying manual merges for {sorter_name}")

            cs = CurationSorting(parent_sorting=sorting)

            for merge in merges:
                unit_ids = merge["unit_ids"]
                new_unit_id = merge["new_unit_id"]

                print(f"Merging units {unit_ids} -> {new_unit_id}")

                cs.merge(
                    unit_ids=unit_ids,
                    new_unit_id=new_unit_id,
                )

            sorting = cs.sorting

            print("After manual merge:")
            print(sorting)
            print("Spike counts after merge:")
            print(sorting.count_num_spikes_per_unit())
        else:
            print("No manual merges applied.")

        # ---------------------
        # 3. Remove duplicated spikes
        # ---------------------
        # This also happens BEFORE analyzer creation.

        dup_cfg = remove_duplicate_settings.get(sorter_name, {})

        if dup_cfg.get("enabled", False):
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
        else:
            print("Duplicate spike removal disabled.")

        # Save cleaned / curated sorting object for comparison later.
        sortings[sorter_name] = sorting

        # ---------------------
        # 4. Create analyzer
        # ---------------------

        analyzer_folder = WORKING_DIR / f"analyzer_M12_{sorter_name}_curated"

        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording_test,
            format="binary_folder",
            folder=analyzer_folder,
            overwrite=True,
        )

        # ---------------------
        # 5. Compute extensions
        # ---------------------

        analyzer.compute("random_spikes")
        analyzer.compute("waveforms")
        analyzer.compute("templates")
        analyzer.compute("noise_levels")

        analyzer.compute(
            "quality_metrics",
            metric_names=[
                "snr",
                "isi_violation",
                "firing_rate",
                "presence_ratio",
            ],
        )

        metrics = analyzer.get_extension("quality_metrics").get_data()

        print("\nQuality metrics:")
        print(metrics)

        metrics.to_csv(
            WORKING_DIR / f"quality_metrics_M12_{sorter_name}_curated.csv"
        )

        # ---------------------
        # 6. Export to Phy
        # ---------------------

        phy_folder = OUTPUT_DIR / f"phy_M12_{sorter_name}_curated"

        si.export_to_phy(
            analyzer,
            output_folder=phy_folder,
            remove_if_exists=True,
        )

        print(f"Exported phy folder: {phy_folder}")

    # =====================
    # Compare sorters after curation
    # =====================
    # This must stay OUTSIDE the for loop.
    # The if __name__ == "__main__" block below prevents Windows multiprocessing
    # from re-running the whole script during comparison.

    if len(sortings) >= 2:
        print("\n===== Comparing curated sorters =====")

        comparison = si.compare_multiple_sorters(
            sorting_list=list(sortings.values()),
            name_list=list(sortings.keys()),
        )

        print(comparison)
    else:
        print("\nNot enough valid sorters for comparison.")


if __name__ == "__main__":
    main()