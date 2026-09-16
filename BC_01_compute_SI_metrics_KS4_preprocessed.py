from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sc
from spikeinterface.core import ChannelSparsity


# ============================================================
# USER SETTINGS
# ============================================================

# ---- Raw SpikeGLX folder ----
SPIKEGLX_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3\original_data"
)
TEMP_WH_PATH = r"C:\Lab\Processing\TG963_site1-3\temp_wh.dat"

# ---- Kilosort4 output folder ----
KILOSORT4_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3"
)

# ---- Output folder ----
OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened"
)

ANALYZER_FOLDER = OUTPUT_DIR / "sorting_analyzer"
GOOD_SORTING_FOLDER = OUTPUT_DIR / "sorting_bombcell_good"

METRICS_CSV = OUTPUT_DIR / "all_quality_metrics.csv"
BOMBCELL_CSV = OUTPUT_DIR / "bombcell_labels.csv"
COMBINED_CSV = OUTPUT_DIR / "metrics_with_bombcell_labels.csv"

# ============================================================
# COMPUTATION SETTINGS
# ============================================================

N_JOBS = 2
CHUNK_DURATION = "30s"
total_memory = "16G"

# Number of randomly sampled spikes per unit used for waveform/PCA analysis.
# For NP data, 500-1000 is a reasonable starting point.
MAX_SPIKES_PER_UNIT = 500

# PCA settings
N_COMPONENTS = 5
PCA_MODE = "by_channel_global"


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

    # # --------------------------------------------------------
    # # Signal / amplitude quality
    # # --------------------------------------------------------
    # "snr",
    # "amplitude_median",
    # "amplitude_cutoff",

    # --------------------------------------------------------
    # Refractory period metrics
    # --------------------------------------------------------

    # Traditional ISI violation metric
    "isi_violation",

    # RP contamination / violations used by SI Bombcell
    "rp_violation",

    # Optional newer sliding refractory metric
    #"sliding_rp_violations",

    # # --------------------------------------------------------
    # # Drift
    # # --------------------------------------------------------
    # "drift",

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
    print("Loading SpikeGLX recording")
    print("========================================")

    # recording = si.read_spikeglx(
    #     SPIKEGLX_FOLDER,
    #     stream_name="imec0.ap",
    # )
    raw_recording = si.read_spikeglx(
        SPIKEGLX_FOLDER,
        stream_name="imec0.ap",
    )

    channel_ids = raw_recording.channel_ids[:384]

    temp_recording = si.read_binary(
        TEMP_WH_PATH,
        sampling_frequency=raw_recording.get_sampling_frequency(),
        dtype="int16",
        num_channels=385,
        channel_ids=np.arange(385),
        is_filtered=True,
    )

    # remove the last sync channel, keep AP0-AP383
    recording = temp_recording.select_channels(
        channel_ids=np.arange(384)
    )

    recording = recording.set_probe(
        raw_recording.get_probe()
    )
    print(recording.channel_ids[:10])
    print(raw_recording.channel_ids[:10])

    print(recording)
    print(recording.channel_ids[:10])
    print(recording.get_channel_locations()[:10])

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

    n_channels = recording.get_num_channels()
    # Dummy scaling:
    # 1 temp_wh unit -> SI 1 "uV"
    recording.set_channel_gains(
        np.ones(n_channels, dtype=np.float64)
    )
    recording.set_channel_offsets(
        np.zeros(n_channels, dtype=np.float64)
    )

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
        "spike_amplitudes"
    )

    compute_if_missing(
        analyzer,
        "spike_locations",
        method="center_of_mass",
    )

    # print("\n========================================")
    # print("Computing PCA")
    # print("========================================")

    # compute_if_missing(
    #     analyzer,
    #     "principal_components",
    #     n_components=N_COMPONENTS,
    #     mode=PCA_MODE,
    #     whiten=False,
    # )

    print("\n========================================")
    print("Computing quality metrics")
    print("========================================")

    analyzer.compute(
        "quality_metrics",
        metric_names=QUALITY_METRIC_NAMES,
        metric_params={
            "rp_violation": {
                # for CS:
                # "refractory_period_ms": 3,
                # "censored_period_ms": 1.5,
                # for SS & all other types:
                "refractory_period_ms": 1,
                "censored_period_ms":0.25,
            },
        },
        skip_pc_metrics=True,
    )

    quality_metrics = (
        analyzer
        .get_extension("quality_metrics")
        .get_data()
    )

    # ============================================================
    # Extra RP contamination for low-firing-rate units
    # 1.5–3 ms window
    # ============================================================

    from spikeinterface.qualitymetrics import compute_refrac_period_violations

    low_fr_unit_ids = quality_metrics.index[
        quality_metrics["firing_rate"] < 10
    ].tolist()

    print(
        f"Computing 1.5–3 ms RP contamination for "
        f"{len(low_fr_unit_ids)} units with firing_rate < 10 Hz"
    )

    rp_contam_1p5_3ms, rp_viol_1p5_3ms = compute_refrac_period_violations(
        analyzer,
        refractory_period_ms=3.0,
        censored_period_ms=1.5,
        unit_ids=low_fr_unit_ids,
    )

    quality_metrics["rp_contamination_1p5_3ms"] = np.nan
    quality_metrics["rp_violations_1p5_3ms"] = np.nan

    for unit_id in low_fr_unit_ids:
        quality_metrics.loc[
            unit_id, "rp_contamination_1p5_3ms"
        ] = rp_contam_1p5_3ms[unit_id]

        quality_metrics.loc[
            unit_id, "rp_violations_1p5_3ms"
        ] = rp_viol_1p5_3ms[unit_id]

    print("\nQuality metric columns:")
    print(list(quality_metrics.columns))

    print("\n========================================")
    print("Computing template metrics")
    print("========================================")

    compute_if_missing(
        analyzer,
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

    all_metrics["rp_contamination_1p5_3ms"] = (
        quality_metrics["rp_contamination_1p5_3ms"]
    )

    all_metrics["rp_violations_1p5_3ms"] = (
        quality_metrics["rp_violations_1p5_3ms"]
    )

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