import spikeinterface.full as si
from multiprocessing import freeze_support
from config_local import WORKING_DIR


def main():
    sorting_ks4 = si.read_sorter_folder(
        WORKING_DIR / "kilosort4_M12_test_output"
    )

    sorting_ms5 = si.read_sorter_folder(
        WORKING_DIR / "mountainsort5_M12_test_output"
    )

    print("Kilosort4:")
    print(sorting_ks4)
    print("Unit IDs:", sorting_ks4.get_unit_ids())

    print("\nMountainSort5:")
    print(sorting_ms5)
    print("Unit IDs:", sorting_ms5.get_unit_ids())

    cmp = si.compare_two_sorters(
        sorting1=sorting_ks4,
        sorting2=sorting_ms5,
        sorting1_name="kilosort4",
        sorting2_name="mountainsort5",
    )

    print("\n===== Agreement scores =====")
    print(cmp.agreement_scores)

    print("\n===== Best matches: KS4 -> MS5 =====")
    print(cmp.best_match_12)

    print("\n===== Best matches: MS5 -> KS4 =====")
    print(cmp.best_match_21)

    print("\n===== Possible matches: KS4 -> MS5 =====")
    print(cmp.possible_match_12)

    print("\n===== Possible matches: MS5 -> KS4 =====")
    print(cmp.possible_match_21)

    print("\n===== Hungarain match: KS4 -> MS5 =====")
    print(cmp.hungarian_match_12)

    print("\n===== Hungarain match: MS5 -> KS4 =====")
    print(cmp.hungarian_match_21)


if __name__ == "__main__":
    freeze_support()
    main()