from pathlib import Path
from multiprocessing import freeze_support

import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sc


# ============================================================
# USER SETTINGS
# ============================================================

# ---- Raw SpikeGLX folder ----
SPIKEGLX_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2\original_data"
)

# ---- Kilosort4 output folder ----
KILOSORT4_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2"
)

# ---- Output folder ----
OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output"
)

ANALYZER_FOLDER = OUTPUT_DIR / "sorting_analyzer"
GOOD_SORTING_FOLDER = OUTPUT_DIR / "sorting_bombcell_good"

METRICS_CSV = OUTPUT_DIR / "all_quality_metrics.csv"
BOMBCELL_CSV = OUTPUT_DIR / "bombcell_labels.csv"
COMBINED_CSV = OUTPUT_DIR / "metrics_with_bombcell_labels.csv"


# ============================================================
# COMPUTATION SETTINGS
# ============================================================

N_JOBS = 16
CHUNK_DURATION = "1s"

# Number of randomly sampled spikes per unit used for waveform/PCA analysis.
# For NP data, 500-1000 is a reasonable starting point.
MAX_SPIKES_PER_UNIT = 1000

# PCA settings
N_COMPONENTS = 5
PCA_MODE = "by_channel_local"


# ============================================================
# QUALITY METRICS
# ============================================================

QUALITY_METRIC_NAMES = [

    # --------------------------------------------------------
    # Spike count / firing properties
    # --------------------------------------------------------
    "num_spikes",
    "firing_rate",
    "presence_ratio",

    # --------------------------------------------------------
    # Signal / amplitude quality
    # --------------------------------------------------------
    "snr",
    "amplitude_median",
    "amplitude_cutoff",

    # --------------------------------------------------------
    # Refractory period metrics
    # --------------------------------------------------------

    # Traditional ISI violation metric
    "isi_violation",

    # RP contamination / violations used by SI Bombcell
    "rp_violation",

    # Optional newer sliding refractory metric
    #"sliding_rp_violations",

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------
    "drift",

    # --------------------------------------------------------
    # PCA / cluster separation metrics
    # --------------------------------------------------------

    # produces: isolation_distance & l_ratio
    "mahalanobis",
    "d_prime",
    "silhouette",

    # produces: nn_hit_rate & nn_miss_rate
    "nearest_neighbor",
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    si.set_global_job_kwargs(
        n_jobs=N_JOBS,
        chunk_duration=CHUNK_DURATION,
        progress_bar=True,
    )

    print("\n========================================")
    print("Loading SpikeGLX recording")
    print("========================================")

    recording = si.read_spikeglx(
        SPIKEGLX_FOLDER,
        stream_name="imec0.ap",
    )

    print(recording)

    print(
        "Recording duration:",
        recording.get_total_duration(),
        "seconds"
    )

    print(
        "Number of channels:",
        recording.get_num_channels()
    )

    print("\n========================================")
    print("Loading Kilosort4 sorting")
    print("========================================")

    sorting = si.read_kilosort(
        KILOSORT4_FOLDER
    )

    print(sorting)

    print(
        "Number of units:",
        sorting.get_num_units()
    )

    print("\nSpike counts:")
    print(sorting.count_num_spikes_per_unit())

    print("\n========================================")
    print("Checking recording / sorting")
    print("========================================")

    print(
        "Recording sampling frequency:",
        recording.get_sampling_frequency()
    )

    print(
        "Sorting sampling frequency:",
        sorting.get_sampling_frequency()
    )

    if (
        recording.get_sampling_frequency()
        != sorting.get_sampling_frequency()
    ):
        raise ValueError(
            "Recording and sorting sampling frequencies do not match!"
        )

    print("\n========================================")
    print("Creating SortingAnalyzer")
    print("========================================")

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=ANALYZER_FOLDER,
        overwrite=True,
        sparse=True,
    )

    print(analyzer)

    print("\n========================================")
    print("Computing random spikes")
    print("========================================")

    analyzer.compute(
        "random_spikes",
        method="uniform",
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        seed=42,
    )

    analyzer.compute(
        "waveforms",
        ms_before=1.0,
        ms_after=2.0,
    )

    print("\n========================================")
    print("Computing templates")
    print("========================================")

    analyzer.compute(
        "templates",
        operators=[
            "average",
            "median",
            "std",
        ],
    )

    analyzer.compute(
        "noise_levels"
    )

    analyzer.compute(
        "spike_amplitudes"
    )

    analyzer.compute(
        "spike_locations",
        method="center_of_mass",
    )

    print("\n========================================")
    print("Computing PCA")
    print("========================================")

    analyzer.compute(
        "principal_components",
        n_components=N_COMPONENTS,
        mode=PCA_MODE,
    )

    print("\n========================================")
    print("Computing quality metrics")
    print("========================================")

    analyzer.compute(
        "quality_metrics",
        metric_names=QUALITY_METRIC_NAMES,
        skip_pc_metrics=False,
    )

    quality_metrics = (
        analyzer
        .get_extension("quality_metrics")
        .get_data()
    )

    print("\nQuality metric columns:")
    print(list(quality_metrics.columns))

    print("\n========================================")
    print("Computing template metrics")
    print("========================================")

    analyzer.compute(
        "template_metrics",
        include_multi_channel_metrics=True,
    )

    template_metrics = (
        analyzer
        .get_extension("template_metrics")
        .get_data()
    )

    print("\nTemplate metric columns:")
    print(list(template_metrics.columns))

    print("\n========================================")
    print("Collecting all metrics")
    print("========================================")

    all_metrics = analyzer.get_metrics_extension_data()

    all_metrics.index.name = "unit_id"

    print(all_metrics)

    all_metrics.to_csv(
        METRICS_CSV
    )

    print("\n========================================")
    print("DONE")
    print("========================================")


if __name__ == "__main__":
    freeze_support()
    main()