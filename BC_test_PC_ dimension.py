from pathlib import Path

import pandas as pd
import spikeinterface as si
from spikeinterface.core import ChannelSparsity


# ============================================================
# USER SETTINGS
# ============================================================

ANALYZER_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened\sorting_analyzer"
)

OUTPUT_DIR = Path(
    r"C:\Lab\Processing\TG963_site1-3\PC_dimension_test"
)

# Conditions to test
CHANNEL_NUMBERS = [2, 4, 6]
PC_NUMBERS = [2, 3, 4]

# ------------------------------------------------------------
# RANDOM SPIKES
# ------------------------------------------------------------

MAX_SPIKES_PER_UNIT = 500

# CRITICAL:
# identical seed for all three analyzers
RANDOM_SEED = 2205


# ------------------------------------------------------------
# WAVEFORMS
#
# IMPORTANT:
# ideally match the waveform settings of your current pipeline
# ------------------------------------------------------------

MS_BEFORE = 1.0
MS_AFTER = 2.0


# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

PCA_MODE = "by_channel_global"
PCA_WHITEN = False


# ------------------------------------------------------------
# CHANNEL RANKING
# ------------------------------------------------------------

PEAK_SIGN = "both"
AMPLITUDE_MODE = "extremum"


# ------------------------------------------------------------
# PARALLELIZATION
# ------------------------------------------------------------

JOB_KWARGS = dict(
    n_jobs=1,
    chunk_duration="20s",
    progress_bar=True,
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ORIGINAL ANALYZER
#
# We use this analyzer mainly for:
#   1. sorting
#   2. recording
#   3. existing templates for ranking best channels
# ============================================================

original_analyzer = si.load_sorting_analyzer(
    ANALYZER_FOLDER
)

sorting = original_analyzer.sorting
recording = original_analyzer.recording

print("\nOriginal analyzer:")
print(original_analyzer)


# ============================================================
# QUALITY METRICS
#
# These four metric groups produce the six outputs we want.
# ============================================================

QUALITY_METRIC_NAMES = [
    "mahalanobis",
    "d_prime",
    "silhouette",
    "nearest_neighbor",
]

OUTPUT_COLUMNS = [
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "silhouette",
    "nn_hit_rate",
    "nn_miss_rate",
]


# ============================================================
# OPTIONAL:
# collect everything into one master dataframe
# ============================================================

all_results = []


# ============================================================
# OUTER LOOP = CHANNEL NUMBER
#
# Only THREE waveform extractions happen:
#
#   best 2 ch
#   best 4 ch
#   best 6 ch
#
# ============================================================

for n_channels in CHANNEL_NUMBERS:

    print("\n")
    print("=" * 100)
    print(
        f"CREATING ANALYZER: BEST {n_channels} CHANNELS"
    )
    print("=" * 100)


    # ========================================================
    # 1. CREATE NESTED CHANNEL SPARSITY
    #
    # Channel ranking comes from the ORIGINAL analyzer.
    #
    # Example:
    #
    # best 2 = [24, 26]
    # best 4 = [24, 26, 22, 28]
    # best 6 = [24, 26, 22, 28, 20, 30]
    #
    # ========================================================

    sparsity = ChannelSparsity.from_best_channels(
        original_analyzer,
        num_channels=n_channels,
        peak_sign=PEAK_SIGN,
        amplitude_mode=AMPLITUDE_MODE,
    )


    # ========================================================
    # 2. CREATE A SEPARATE ANALYZER FOR THIS CHANNEL NUMBER
    # ========================================================

    analyzer_folder = (
        OUTPUT_DIR /
        f"analyzer_best{n_channels}ch"
    )

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=analyzer_folder,
        sparsity=sparsity,
        overwrite=True,
    )

    print("\nNew analyzer:")
    print(analyzer)


    # ========================================================
    # 3. RANDOM SPIKES
    #
    # SAME:
    #   sorting
    #   method
    #   max_spikes_per_unit
    #   seed
    #
    # Therefore all three channel conditions should select
    # exactly the same sorting spike indices.
    #
    # ========================================================

    analyzer.compute(
        "random_spikes",
        method="uniform",
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        seed=RANDOM_SEED,
    )


    # ========================================================
    # 4. EXTRACT WAVEFORMS
    #
    # THIS HAPPENS ONLY ONCE PER CHANNEL CONDITION.
    #
    # Total waveform extractions = 3.
    #
    # ========================================================

    analyzer.compute(
        "waveforms",
        ms_before=MS_BEFORE,
        ms_after=MS_AFTER,
        **JOB_KWARGS,
    )

    analyzer.compute(
        "templates",
        operators=["average"],
    )


    # ========================================================
    # OPTIONAL SANITY CHECK
    #
    # Show number of selected random spikes.
    # ========================================================

    random_ext = analyzer.get_extension(
        "random_spikes"
    )

    random_indices = random_ext.get_data()

    print(
        f"\nRandom spikes selected: "
        f"{len(random_indices)}"
    )

    print(
        "First 10 global spike indices:",
        random_indices[:10],
    )


    # ========================================================
    # INNER LOOP = NUMBER OF PCs
    #
    # NO NEW WAVEFORM EXTRACTION HERE.
    # ========================================================

    for n_pcs in PC_NUMBERS:

        print("\n")
        print("-" * 100)

        print(
            f"RUNNING: "
            f"{n_channels} channels × "
            f"{n_pcs} PCs"
        )

        print("-" * 100)


        # ====================================================
        # 5. RECOMPUTE PCA ONLY
        #
        # Same waveforms.
        # Same random spikes.
        #
        # Only PCA dimensionality changes.
        # ====================================================

        analyzer.compute(
            "principal_components",
            n_components=n_pcs,
            mode=PCA_MODE,
            whiten=PCA_WHITEN,
        )


        # ====================================================
        # 6. COMPUTE PCA-BASED QUALITY METRICS
        # ====================================================

        metrics_ext = analyzer.compute(
            "quality_metrics",
            metric_names=QUALITY_METRIC_NAMES,
        )

        qm = metrics_ext.get_data().copy()


        # ====================================================
        # 7. PREPARE OUTPUT
        # ====================================================

        qm.index.name = "unit_id"

        qm = qm.reset_index()

        wanted_columns = (
            ["unit_id"] +
            OUTPUT_COLUMNS
        )


        # ----------------------------------------------------
        # Check output columns
        # ----------------------------------------------------

        missing = [
            col
            for col in wanted_columns
            if col not in qm.columns
        ]

        if missing:
            print(
                "\nWARNING: missing columns:"
            )
            print(missing)


        available_columns = [
            col
            for col in wanted_columns
            if col in qm.columns
        ]

        qm = qm[available_columns]


        # ====================================================
        # 8. SAVE INDIVIDUAL CSV
        # ====================================================

        csv_path = (
            OUTPUT_DIR /
            f"PC_metrics_"
            f"{n_channels}ch_"
            f"{n_pcs}PC.csv"
        )

        qm.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"\nSaved individual CSV:\n"
            f"{csv_path}"
        )


        # ====================================================
        # 9. ALSO ADD TO MASTER TABLE
        # ====================================================

        master = qm.copy()

        master.insert(
            1,
            "n_channels",
            n_channels,
        )

        master.insert(
            2,
            "n_pcs",
            n_pcs,
        )

        master.insert(
            3,
            "feature_dimension",
            n_channels * n_pcs,
        )

        all_results.append(master)


# ============================================================
# 10. SAVE MASTER CSV
# ============================================================

all_results_df = pd.concat(
    all_results,
    ignore_index=True,
)

master_csv = (
    OUTPUT_DIR /
    "PC_metrics_ALL_conditions.csv"
)

all_results_df.to_csv(
    master_csv,
    index=False,
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    f"\nMaster CSV:\n{master_csv}"
)