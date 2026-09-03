from pathlib import Path

import bombcell as bc


def main():
    # Must match the dataset/output directory used in run_bombcell_compute.py
    ks_dir = r"C:\Lab\Processing\TG963_site1-2"
    save_path = Path(ks_dir) / "bombcell_v3"

    print(f"Loading Bombcell results from: {save_path}")

    # Load the quality metrics and parameters already computed on disk.
    param, quality_metrics, fractions_RPVs_all_taur = bc.load_bc_results(save_path)

    # Reconstruct Bombcell unit classifications from the saved metrics/parameters.
    unit_type, unit_type_string = bc.qm.get_quality_unit_type(
        param,
        quality_metrics,
    )

    # Launch the interactive Bombcell GUI.
    gui = bc.unit_quality_gui(
        ks_dir=ks_dir,
        quality_metrics=quality_metrics,
        unit_types=unit_type,
        param=param,
        save_path=save_path,
    )

    # Keep a reference alive until the GUI is closed.
    return gui


if __name__ == "__main__":
    gui = main()
