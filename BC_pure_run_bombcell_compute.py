from pathlib import Path
from pprint import pprint

import numpy as np
import bombcell as bc


def main():
    # ===== Paths =====
    # Kilosort output directory
    ks_dir = r"/arc/project/st-douga-1/gg5683/TG963_site1-2/"

    # Bombcell output directory
    save_path = r"/scratch/st-douga-1/gg5683/Output_dir/"

    print(f"Using kilosort directory: {ks_dir}")
    print(f"Saving Bombcell results to: {save_path}")

    # ===== Raw / meta files =====
    # These are kept exactly as in your notebook: None.
    raw_file_path = None
    meta_file_path = None

    # ===== Default Bombcell parameters =====
    param = bc.get_default_parameters(
        ks_dir,
        raw_file=raw_file_path,
        meta_file=meta_file_path,
        kilosort_version=4,
    )

    print("\nBombcell parameters before customization:")
    pprint(param)

    # ===== Your customized parameters =====

    # 1. Classification thresholds
    param["maxNTroughs"] = 2
    param["maxScndPeakToTroughRatio_noise"] = 1
    param["maxWvDuration"] = 1401
    param["minSpatialDecaySlopeExp"] = 0.005
    param["minWvDuration"] = 66

    # 2. Quality metrics to compute
    param["computeDistanceMetrics"] = 1
    param["computeDrift"] = 0
    param["splitGoodAndMua_NonSomatic"] = 1
    # 3. Refractory-period violation settings
    param["rpv_method"] = "llobet"
    param["tauR_values"] = np.arange(0.0005, 0.005, 0.0005)
    param["tauC"] = 0.0001

    print("\nBombcell parameters after customization:")
    pprint(param)

    print("\nStarting Bombcell quality-metric computation...")

    quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
        ks_dir,
        save_path,
        param,
    )

    print("\nBombcell computation finished.")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
