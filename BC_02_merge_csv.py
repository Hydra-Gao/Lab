import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Lab\Processing\TG963_site1-3\bombcell_output_merged")

csv1 = pd.read_csv(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_filtered_raw\all_quality_metrics.csv"
)

csv2 = pd.read_csv(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened\all_quality_metrics.csv"
)

# NEW
csv3 = pd.read_csv(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened\unwhitened_KS_PC_isolation_metrics_with_covariance_diagnostics_and_channel_zero_fractions.csv"
)

key = "unit_id"


# ============================================================
# Columns to take from CSV1
# ============================================================

cols_from_csv1 = [
    "amplitude_median",
    "snr",
    "amplitude_cutoff",
    # "drift_ptp",
    # "drift_std",
    # "drift_mad",
    "spread",
    "exp_decay",
    "velocity_above",
    "velocity_below",
]


# ============================================================
# Columns to take from CSV3
# ============================================================

cols_from_csv3 = [
    # "isolation_distance",
    # "l_ratio",
    # "d_prime",
    # "silhouette",
    # "nn_hit_rate",
    # "nn_miss_rate",
    # "pc_quality",
    # "max_channel_zero_fraction"
]


# ============================================================
# Merge CSV1 into CSV2
# ============================================================

df1_selected = csv1[[key] + cols_from_csv1].copy()

cols_to_drop = [
    col for col in cols_from_csv1
    if col in csv2.columns
]

csv2 = csv2.drop(columns=cols_to_drop)

merged = csv2.merge(
    df1_selected,
    on=key,
    how="left",
)


# ============================================================
# Merge CSV3
# ============================================================

df3_selected = csv3[[key] + cols_from_csv3].copy()

cols_to_drop = [
    col for col in cols_from_csv3
    if col in merged.columns
]

merged = merged.drop(columns=cols_to_drop)

merged = merged.merge(
    df3_selected,
    on=key,
    how="left",
)

merged = merged[
    ["unit_id"] +
    [col for col in merged.columns if col != "unit_id"]
]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "SI-PC_Comprehensive_metrics_unwhitened_with_channel_zero_fractions.csv"

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"Saved to: {OUTPUT_FILE}")