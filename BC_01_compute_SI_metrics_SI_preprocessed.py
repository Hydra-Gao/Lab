from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sc
import spikeinterface.preprocessing as spre
from spikeinterface.core import ChannelSparsity


# ============================================================
# USER SETTINGS
# ============================================================

# ---- Raw SpikeGLX folder ----
SPIKEGLX_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3\original_data"
)
# TEMP_WH_PATH = r"C:\Lab\Processing\TG963_site1-2\temp_wh.dat"

# ---- Kilosort4 output folder ----
KILOSORT4_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3"
)

# ---- Output folder ----
OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_filtered_raw"
)

ANALYZER_FOLDER = OUTPUT_DIR / "sorting_analyzer"
GOOD_SORTING_FOLDER = OUTPUT_DIR / "sorting_bombcell_good"

METRICS_CSV = OUTPUT_DIR / "all_quality_metrics.csv"
BOMBCELL_CSV = OUTPUT_DIR / "bombcell_labels.csv"
COMBINED_CSV = OUTPUT_DIR / "metrics_with_bombcell_labels.csv"

# ============================================================
# COMPUTATION SETTINGS
# ============================================================

N_JOBS = 1
CHUNK_DURATION = "30s"
total_memory = "16G"

# Number of randomly sampled spikes per unit used for waveform/PCA analysis.
# For NP data, 500-1000 is a reasonable starting point.
MAX_SPIKES_PER_UNIT = 1000

# PCA settings
N_COMPONENTS = 5
PCA_MODE = "by_channel_global"

highpass_filter_min_freq = 10.0  # Hz


# ============================================================
# QUALITY METRICS
# ============================================================

QUALITY_METRIC_NAMES = [
    # --------------------------------------------------------
    # Signal / amplitude quality / drift
    # --------------------------------------------------------
    "snr",
    "amplitude_median",
    "amplitude_cutoff",
    "drift",
    "amplitude_cv",
    "firing_range",

    # # --------------------------------------------------------
    # # Spike count / firing properties
    # # --------------------------------------------------------
    # "num_spikes",
    # "firing_rate",
    # "presence_ratio",

    # # --------------------------------------------------------
    # # Refractory period metrics
    # # --------------------------------------------------------

    # # Traditional ISI violation metric
    # "isi_violation",

    # # RP contamination / violations used by SI Bombcell
    # "rp_violation",

    # # Optional newer sliding refractory metric
    # #"sliding_rp_violations",

    # # --------------------------------------------------------
    # # PCA / cluster separation metrics
    # # --------------------------------------------------------

    # # produces: isolation_distance & l_ratio
    # "mahalanobis",
    # "d_prime",
    # "silhouette",

    # # produces: nn_hit_rate & nn_miss_rate
    # "nearest_neighbor",
]


TEMPLATE_METRIC_NAMES = [
    "spread",
    "exp_decay",
    "velocity_fits",
]

def compute_if_missing(analyzer, extension_name, **kwargs):
    if analyzer.has_extension(extension_name):
        print(f"[SKIP] {extension_name} already computed")
        return analyzer.get_extension(extension_name)

    print(f"[COMPUTE] {extension_name}")
    return analyzer.compute(
        extension_name,
        **kwargs,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    si.set_global_job_kwargs(
        n_jobs=N_JOBS,
        chunk_duration=CHUNK_DURATION,
        # total_memory=total_memory,
        progress_bar=True,
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

    print("Getting main_channel_indices from Kilosort4 cluster_info.tsv...")
    cluster_info = pd.read_csv(
        KILOSORT4_FOLDER / "cluster_info.tsv",
        sep="\t"
    )    

    unit_to_main_ch = dict(
        zip(
            cluster_info["cluster_id"].astype(int),
            cluster_info["ch"].astype(int),
        )
    )

    main_channel_indices = np.array(
        [
            unit_to_main_ch[int(unit_id)]
            for unit_id in sorting.unit_ids
        ],
        dtype=int,
    )

    print("\n========================================")
    print("Loading SpikeGLX recording")
    print("========================================")

    # recording = si.read_spikeglx(
    #     SPIKEGLX_FOLDER,
    #     stream_name="imec0.ap",
    # )

    recording_raw = si.read_spikeglx(
        SPIKEGLX_FOLDER,
        stream_name="imec0.ap",
    )

    recording = spre.highpass_filter(
        recording_raw,
        freq_min=highpass_filter_min_freq,
        filter_order=3,
        filter_mode="ba",
        ftype="butter",
        dtype="float32",
        ignore_low_freq_error=True
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
    print("Building sparsity and Creating SortingAnalyzer")
    print("========================================")

    if ANALYZER_FOLDER.exists():
        print("Loading existing SortingAnalyzer...")
        
        analyzer = si.load_sorting_analyzer(
            ANALYZER_FOLDER,
            load_extensions=True,
        )

    else:
        print("Creating new SortingAnalyzer...")

        radius_um = 100.0

        unit_ids = sorting.unit_ids
        channel_ids = recording.channel_ids
        channel_locations = recording.get_channel_locations()

        main_channel_indices = np.asarray(main_channel_indices, dtype=int)

        assert len(main_channel_indices) == len(unit_ids)
        assert np.all(main_channel_indices >= 0)
        assert np.all(main_channel_indices < len(channel_ids))

        distances = np.linalg.norm(
            channel_locations[:, None, :] -
            channel_locations[None, :, :],
            axis=2,
        )

        mask = np.zeros(
            (len(unit_ids), len(channel_ids)),
            dtype=bool,
        )

        for unit_ind, main_chan_ind in enumerate(main_channel_indices):
            mask[unit_ind] = distances[main_chan_ind] <= radius_um

        sparsity = ChannelSparsity(
            mask,
            unit_ids,
            channel_ids,
        )

        print(sparsity)

        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            format="binary_folder",
            folder=ANALYZER_FOLDER,
            overwrite=True,
            sparsity=sparsity,
        )

    print(analyzer)
    print("Saved extensions:", analyzer.get_saved_extension_names())

    print("\n========================================")
    print("Computing random spikes")
    print("========================================")

    compute_if_missing(
        analyzer,
        "random_spikes",
        method="uniform",
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        seed=42,
    )

    compute_if_missing(
        analyzer,
        "waveforms",
        ms_before=1.0,
        ms_after=2.0,
    )

    print("\n========================================")
    print("Computing templates")
    print("========================================")

    compute_if_missing(
        analyzer,
        "templates",
        operators=[
            "average",
            "median",
            "std",
        ],
    )

    compute_if_missing(
        analyzer,
        "noise_levels"
    )

    compute_if_missing(
        analyzer,
        "spike_amplitudes",
        peak_sign="both",
    )

    # print("\n========================================")
    # print("Computing PCA")
    # print("========================================")

    # analyzer.compute(
    #     "principal_components",
    #     n_components=N_COMPONENTS,
    #     mode=PCA_MODE,
    # )

    print("\n========================================")
    print("Computing quality metrics")
    print("========================================")

    compute_if_missing(
        analyzer,
        "quality_metrics",
        metric_names=QUALITY_METRIC_NAMES,
        # skip_pc_metrics=False,
        skip_pc_metrics=True, 
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

    compute_if_missing(
        analyzer,
        "template_metrics",
        metric_names=TEMPLATE_METRIC_NAMES,
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