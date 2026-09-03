from pathlib import Path

import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sc

MERGED_METRICS_CSV = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_merged\Comprehensive_metrics_unwhitened.csv"
)

OUTPUT_CSV = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_merged\bombcell_labels_custom_unwhitened.csv"
)

DETAIL_OUTPUT_CSV = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output_merged\unit_classification_details_unwhitened.csv"
)

metrics = pd.read_csv(
    MERGED_METRICS_CSV,
    index_col="unit_id"
)

# ============================================================
# LOAD DEFAULT BOMBCELL THRESHOLDS
# ============================================================

thresholds = sc.bombcell_get_default_thresholds()

print("\n===== DEFAULT THRESHOLDS =====")
print(thresholds)


# ============================================================
# MODIFY THRESHOLDS HERE
# ============================================================

# -------- MUA / GOOD criteria --------

# Example:
thresholds["mua"]["snr"]["greater"] = 2
thresholds["mua"]["num_spikes"]["greater"] = 500
thresholds["mua"]["amplitude_median"]["greater"] = 233


# ============================================================
# OPTIONAL: DISABLE A THRESHOLD
# ============================================================

# Example:
# thresholds["mua"]["drift_ptp"]["less"] = None
thresholds["noise"]["peak_to_trough_duration"]["less"] = None
thresholds["noise"]["peak_to_trough_duration"]["greater"] = None

# ============================================================
# MODIFY NOISE CRITERIA
# ============================================================

# thresholds["noise"]["num_positive_peaks"]["less"] = ...
# thresholds["noise"]["num_negative_peaks"]["less"] = 2

thresholds["noise"]["peak_after_to_trough_ratio"]["less"] = 1.0
thresholds["noise"]["exp_decay"]["greater"] = 0.005
thresholds["noise"]["num_positive_peaks"]["less"] = 2
thresholds["noise"]["num_negative_peaks"]["less"] = 2

# ============================================================
# OPTIONAL: MODIFY NON-SOMATIC CRITERIA
# ============================================================

# Example:
# thresholds["non-somatic"]["trough_width"]["less"] = ...

# ============================================================
# CUSTOM PC QUALITY / ISOLATION / OTHER THRESHOLDS
# ============================================================

MIN_PC_METRICS_PASS = 4

MIN_ISOLATION_DISTANCE = 20
MAX_L_RATIO = 0.3
MIN_D_PRIME = 2
MIN_SILHOUETTE = 0.2
MIN_NN_HIT_RATE = 0.8
MAX_NN_MISS_RATE = 0.2

PC_METRIC_THRESHOLDS = {
    "isolation_distance": ("greater", MIN_ISOLATION_DISTANCE),
    "l_ratio": ("less", MAX_L_RATIO),
    "d_prime": ("greater", MIN_D_PRIME),
    "silhouette": ("greater", MIN_SILHOUETTE),
    "nn_hit_rate": ("greater", MIN_NN_HIT_RATE),
    "nn_miss_rate": ("less", MAX_NN_MISS_RATE),
}

MAX_ISI_VIOLATIONS_RATIO = 0.5

print("\n===== CUSTOM THRESHOLDS =====")
print(thresholds)


# ============================================================
# RUN BOMBCELL WITH CUSTOM THRESHOLDS
# ============================================================

labels = sc.bombcell_label_units(
    sorting_analyzer=None,
    external_metrics=metrics,
    thresholds=thresholds,
    label_non_somatic=True,
    split_non_somatic_good_mua=True,
)


print("\n===== LABELS =====")
print(labels)

# Different SI versions may call this column either
# "bombcell_label" or "label".
if "bombcell_label" in labels.columns:
    label_column = "bombcell_label"
elif "label" in labels.columns:
    label_column = "label"
else:
    raise ValueError(
        f"Cannot find Bombcell label column. "
        f"Available columns: {list(labels.columns)}"
    )


print("\n===== COUNTS =====")
print(
    labels[label_column].value_counts()
)


# ============================================================
# LOAD ALL PRECOMPUTED METRICS
# ============================================================

result = metrics.join(labels)

# Standardize the Bombcell label column name
if label_column != "bombcell_label":
    result["bombcell_label"] = result[label_column]


# ============================================================
# HELPER FUNCTION:
# FIND FAILED BOMBCELL CRITERIA
# ============================================================

def find_failed_criteria(row, criteria_dict):
    """
    Return all criteria that this unit fails.

    SpikeInterface threshold semantics:

        "greater": X  -> value >= X passes
                        value < X fails

        "less": X     -> value <= X passes
                        value > X fails

    Returns strings such as:

        snr=3.82 < 5
        rp_contamination=0.154 > 0.1
    """

    failed = []

    for metric_name, limits in criteria_dict.items():

        # --------------------------------------------
        # Metric missing
        # --------------------------------------------

        if metric_name not in row.index:
            failed.append(
                f"{metric_name}=MISSING"
            )
            continue


        value = row[metric_name]


        # --------------------------------------------
        # NaN
        # --------------------------------------------

        if pd.isna(value):
            failed.append(
                f"{metric_name}=NaN"
            )
            continue


        # --------------------------------------------
        # Some threshold definitions can specify abs
        # --------------------------------------------

        use_abs = limits.get("abs", False)

        value_for_test = (
            abs(value)
            if use_abs
            else value
        )


        greater_threshold = limits.get(
            "greater",
            None
        )

        less_threshold = limits.get(
            "less",
            None
        )


        # --------------------------------------------
        # Minimum threshold
        #
        # value >= threshold = PASS
        # value < threshold  = FAIL
        # --------------------------------------------

        if greater_threshold is not None:

            if value_for_test < greater_threshold:

                failed.append(
                    f"{metric_name}={value_for_test:.6g} "
                    f"< {greater_threshold}"
                )


        # --------------------------------------------
        # Maximum threshold
        #
        # value <= threshold = PASS
        # value > threshold  = FAIL
        # --------------------------------------------

        if less_threshold is not None:

            if value_for_test > less_threshold:

                failed.append(
                    f"{metric_name}={value_for_test:.6g} "
                    f"> {less_threshold}"
                )


    return failed


# ============================================================
# RECORD BOMBCELL FAILURE CRITERIA
# ============================================================

failed_noise_criteria = []
failed_mua_criteria = []


for unit_id, row in result.iterrows():

    # --------------------------------------------------------
    # Noise criteria
    # --------------------------------------------------------

    noise_failed = find_failed_criteria(
        row,
        thresholds["noise"],
    )

    failed_noise_criteria.append(
        "; ".join(noise_failed)
    )


    # --------------------------------------------------------
    # MUA criteria
    #
    # Bombcell only evaluates MUA after a unit survives
    # the noise stage.
    #
    # Therefore, for Bombcell-noise units we leave this blank.
    # --------------------------------------------------------

    bombcell_label = row["bombcell_label"]

    if bombcell_label == "noise":

        failed_mua_criteria.append("")

    else:

        mua_failed = find_failed_criteria(
            row,
            thresholds["mua"],
        )

        failed_mua_criteria.append(
            "; ".join(mua_failed)
        )


result["failed_noise_criteria"] = (
    failed_noise_criteria
)

result["failed_mua_criteria"] = (
    failed_mua_criteria
)


# ============================================================
# START FINAL LABEL FROM BOMBCELL LABEL
# ============================================================

result["final_label"] = (
    result["bombcell_label"]
)


# ============================================================
# CUSTOM FAILED CRITERIA
# ============================================================

# We will collect ALL custom failures here.

result["custom_failed_criteria"] = ""


# ============================================================
# CUSTOM GOOD -> MUA DOWNGRADE
# ============================================================
# Only units that are still "good" are evaluated here.
#
# Rules:
#
# 1. pc_quality must be "pass".
#    Otherwise -> MUA immediately.
#
# 2. If pc_quality == "pass":
#    evaluate the 6 PC isolation metrics.
#
# 3. At least MIN_PC_METRICS_PASS metrics must pass.
#    Otherwise -> MUA.
#
# NaN values count as NOT passing that metric.
# ============================================================

custom_mua = pd.Series(
    False,
    index=result.index
)

pc_metrics_pass_count = pd.Series(
    0,
    index=result.index,
    dtype=int
)


for unit_id, row in result.iterrows():

    # --------------------------------------------------------
    # Only modify units currently classified as good
    # --------------------------------------------------------

    if row["final_label"] not in ["good", "non_soma_good"]:
        continue    


    failed = []


    # ========================================================
    # STEP 1: PC quality
    # ========================================================

    pc_quality = row.get("pc_quality")

    if (
        pd.isna(pc_quality)
        or str(pc_quality).lower() != "pass"
    ):

        custom_mua.loc[unit_id] = True

        failed.append(
            f"MUA:pc_quality={pc_quality}"
        )

        result.loc[
            unit_id,
            "custom_failed_criteria"
        ] = "; ".join(failed)

        continue


    # ========================================================
    # STEP 2: Check 6 PC metrics
    # ========================================================

    pass_count = 0

    for metric_name, (
        comparison,
        threshold
    ) in PC_METRIC_THRESHOLDS.items():

        value = row.get(metric_name)

        # NaN / missing -> does not pass
        if pd.isna(value):

            failed.append(
                f"{metric_name}=NaN"
            )

            continue


        # ----------------------------------------------------
        # Minimum threshold:
        # value >= threshold -> PASS
        # ----------------------------------------------------

        if comparison == "greater":

            passed = (
                value >= threshold
            )

            if not passed:
                failed.append(
                    f"{metric_name}="
                    f"{value:.6g} "
                    f"< {threshold}"
                )


        # ----------------------------------------------------
        # Maximum threshold:
        # value <= threshold -> PASS
        # ----------------------------------------------------

        elif comparison == "less":

            passed = (
                value <= threshold
            )

            if not passed:
                failed.append(
                    f"{metric_name}="
                    f"{value:.6g} "
                    f"> {threshold}"
                )


        else:

            raise ValueError(
                f"Unknown comparison: "
                f"{comparison}"
            )


        if passed:
            pass_count += 1


    # Save number of passed PC metrics
    pc_metrics_pass_count.loc[
        unit_id
    ] = pass_count


    # ========================================================
    # STEP 3: Require at least X / 6 metrics to pass
    # ========================================================

    if pass_count < MIN_PC_METRICS_PASS:

        custom_mua.loc[unit_id] = True

        failed.insert(
            0,
            f"MUA:PC_metrics_pass="
            f"{pass_count}/"
            f"{len(PC_METRIC_THRESHOLDS)} "
            f"< {MIN_PC_METRICS_PASS}"
        )

    # ========================================================
    # OTHER PARALLEL MUA CRITERIA
    # ========================================================

    isi_violations_ratio = row.get("isi_violations_ratio")

    if (
        pd.isna(isi_violations_ratio)
        or isi_violations_ratio > MAX_ISI_VIOLATIONS_RATIO
    ):
        custom_mua.loc[unit_id] = True

        failed.append(
            f"MUA:isi_violations_ratio="
            f"{isi_violations_ratio} "
            f"> {MAX_ISI_VIOLATIONS_RATIO}"
        )


    # Save custom failure reasons
    result.loc[
        unit_id,
        "custom_failed_criteria"
    ] = "; ".join(failed)


# ============================================================
# SAVE PC METRIC PASS COUNT
# ============================================================

result["pc_metrics_pass_count"] = (
    pc_metrics_pass_count
)


# ============================================================
# APPLY CUSTOM MUA LABEL
# ============================================================

result.loc[
    custom_mua,
    "final_label"
] = "mua"


# ============================================================
# SAVE FULL RESULT
# ============================================================

result.to_csv(
    OUTPUT_CSV
)


# ============================================================
# SAVE CLASSIFICATION DETAILS ONLY
# ============================================================

classification_details = result[
    [
        "bombcell_label",
        "failed_noise_criteria",
        "failed_mua_criteria",
        "pc_metrics_pass_count",
        "custom_failed_criteria",
        "final_label",
    ]
].copy()


classification_details.index.name = "unit_id"


classification_details.to_csv(
    DETAIL_OUTPUT_CSV
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("FINAL LABEL COUNTS")
print("========================================")

print(
    result[
        "final_label"
    ].value_counts()
)


print("\n========================================")
print("CLASSIFICATION DETAILS")
print("========================================")

print(
    classification_details
)


print("\nFull result saved to:")
print(
    OUTPUT_CSV
)

print("\nClassification details saved to:")
print(
    DETAIL_OUTPUT_CSV
)