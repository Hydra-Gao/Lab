from pathlib import Path

import spikeinterface.full as si
import spikeinterface.curation as sc


ANALYZER_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output\sorting_analyzer"
)

OUTPUT_CSV = Path(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output\bombcell_labels_custom.csv"
)


# ============================================================
# LOAD EXISTING ANALYZER
# ============================================================

analyzer = si.load_sorting_analyzer(
    ANALYZER_FOLDER
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

# # Example: require SNR >= 6 instead of default 5
# thresholds["mua"]["snr"]["greater"] = 6

# ============================================================
# OPTIONAL: DISABLE A THRESHOLD
# ============================================================

# Example:
# Do not use drift for good/MUA classification
# thresholds["mua"]["drift_ptp"]["less"] = None
thresholds["noise"]["peak_to_trough_duration"]["less"] = None
thresholds["noise"]["peak_to_trough_duration"]["greater"] = None

# ============================================================
# OPTIONAL: MODIFY NOISE CRITERIA
# ============================================================

# Example:
# thresholds["noise"]["num_positive_peaks"]["less"] = ...
# thresholds["noise"]["num_negative_peaks"]["less"] = 2
# thresholds["noise"]["peak_to_trough_duration"]["less"] = 0.001401
# thresholds["noise"]["peak_to_trough_duration"]["greater"] = 0.000066
thresholds["noise"]["exp_decay"]["greater"] = 0.005
thresholds["noise"]["peak_after_to_trough_ratio"]["less"] = 1.0

# ============================================================
# OPTIONAL: MODIFY NON-SOMATIC CRITERIA
# ============================================================

# Example:
# thresholds["non-somatic"]["trough_width"]["less"] = ...


print("\n===== CUSTOM THRESHOLDS =====")
print(thresholds)


# ============================================================
# RUN BOMBCELL WITH CUSTOM THRESHOLDS
# ============================================================

labels = sc.bombcell_label_units(
    sorting_analyzer=analyzer,
    thresholds=thresholds,
    label_non_somatic=True,
    split_non_somatic_good_mua=True,
)


print("\n===== LABELS =====")
print(labels)

print("\n===== COUNTS =====")
print(
    labels["bombcell_label"].value_counts()
)

metrics = analyzer.get_metrics_extension_data()

result = metrics.join(labels)
result["final_label"] = result["bombcell_label"]

# ============================================================
# CUSTOM NOISE OVERRIDE
# ============================================================

custom_noise = (
    result["amplitude_median"].abs() < 30
)

result.loc[
    custom_noise,
    "final_label"
] = "noise"

# ============================================================
# CUSTOM GOOD -> MUA DOWNGRADE
# ============================================================

custom_mua = (
    (result["final_label"] == "good")
    &
    (
        (result["isolation_distance"] < 20)
        |
        (result["l_ratio"] > 0.3)
    )
)

result.loc[
    custom_mua,
    "final_label"
] = "mua"


result.to_csv(
    OUTPUT_CSV
)