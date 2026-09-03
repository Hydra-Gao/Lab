from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# MANUAL_LABEL_CSV = Path(r"C:\Lab\Processing\TG963_site1-2\cluster_group.tsv")
MANUAL_LABEL_CSV = Path(r"C:\Lab\Processing\TG963_site1-2\bombcell_output\unit_classification_details.csv")
METRICS_CSV = Path(r"C:\Lab\Processing\TG963_site1-2\bombcell_output\all_quality_metrics.csv")

OUTPUT_DIR = Path(r"C:\Lab\Processing\TG963_site1-2\bombcell_output\metric_distribution_plots_pcfeatures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# column names in your CSVs
# UNIT_ID_COL_MANUAL = "cluster_id"
# LABEL_COL_MANUAL = "group"
UNIT_ID_COL_MANUAL = "unit_id"
LABEL_COL_MANUAL = "bombcell_label"

UNIT_ID_COL_METRICS = "unit_id"

# labels expected in manual_label
GOOD_LABEL = "good"
MUA_LABEL = "mua"
NOISE_LABEL = "noise"
UNSORTED_LABEL = "unsorted"
NON_SOMATIC_MUA_LABEL = "non_soma_mua"
NON_SOMATIC_GOOD_LABEL = "non_soma_good"

# metrics to plot
METRICS_TO_PLOT = [
    # "snr",
    "amplitude_median",
    # "rp_contamination",
    # "presence_ratio",
    # "amplitude_cutoff",
    # "num_spikes",
    "isolation_distance",
    "l_ratio",
    "silhouette",
    "d_prime",
    "nn_hit_rate",
    "nn_miss_rate",
    # "drift_ptp",
    # "isi_violations_ratio",
    # "waveform_baseline_flatness",
    # "exp_decay",
    # "peak_before_to_trough_ratio",
    # "peak_after_to_trough_ratio",
    # "peak_before_to_peak_after_ratio",
    # "main_peak_to_trough_ratio",
]

# metrics where absolute value is more meaningful
USE_ABS_FOR = [
    "amplitude_median",
]

XLIM_MODE = "percentile"   # "none" or "percentile"
XLIM_LOWER = 0             # 1st percentile
XLIM_UPPER = 95            # 99th percentile


# ============================================================
# LOAD DATA
# ============================================================

# manual_df = pd.read_csv(MANUAL_LABEL_CSV, sep="\t")
manual_df = pd.read_csv(MANUAL_LABEL_CSV)
metrics_df = pd.read_csv(METRICS_CSV)

manual_df[UNIT_ID_COL_MANUAL] = manual_df[UNIT_ID_COL_MANUAL].astype(str)
metrics_df[UNIT_ID_COL_METRICS] = metrics_df[UNIT_ID_COL_METRICS].astype(str)

df = manual_df.merge(
    metrics_df,
    left_on=UNIT_ID_COL_MANUAL,
    right_on=UNIT_ID_COL_METRICS,
    how="inner",
)

print("Merged rows:", len(df))
print("Label counts:")
print(df[LABEL_COL_MANUAL].value_counts(dropna=False))


# ============================================================
# OPTIONAL CLEANING
# ============================================================

# Example: remove obviously invalid peak_to_trough if you later include it
# df.loc[df["peak_to_trough_duration"] <= 0, "peak_to_trough_duration"] = np.nan

# add abs-version columns when useful
for metric in USE_ABS_FOR:
    if metric in df.columns:
        df[f"{metric}_abs"] = df[metric].abs()


# ============================================================
# HELPER
# ============================================================

def get_plot_series(df, metric):
    """Return the series to plot."""
    if metric in USE_ABS_FOR and f"{metric}_abs" in df.columns:
        return f"{metric}_abs"
    return metric


def get_axis_limits(series, mode="percentile", lower=1, upper=99):
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) == 0:
        return None

    if mode == "none":
        return None

    if mode == "percentile":
        x_min = np.percentile(series, lower)
        x_max = np.percentile(series, upper)

        # 防止极端情况下上下界一样
        if x_min == x_max:
            return None

        return x_min, x_max

    return None


def save_histogram(df, metric, label_col, output_path):
    plot_metric = get_plot_series(df, metric)

    sub = df[[label_col, plot_metric]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    good_vals = sub.loc[sub[label_col] == GOOD_LABEL, plot_metric].values
    mua_vals = sub.loc[sub[label_col] == MUA_LABEL, plot_metric].values
    noise_vals = sub.loc[sub[label_col] == NOISE_LABEL, plot_metric].values
    non_somatic_mua_vals = sub.loc[sub[label_col] == NON_SOMATIC_MUA_LABEL, plot_metric].values
    non_somatic_good_vals = sub.loc[sub[label_col] == NON_SOMATIC_GOOD_LABEL, plot_metric].values

    all_vals = sub[plot_metric].values

    if len(all_vals) == 0:
        return

    xlim = get_axis_limits(
        all_vals,
        mode=XLIM_MODE,
        lower=XLIM_LOWER,
        upper=XLIM_UPPER,
    )

    plt.figure(figsize=(8, 5))

    bins = 40

    if len(good_vals) > 0:
        plt.hist(good_vals, bins=bins, alpha=0.5, density=True, label="good")

    if len(mua_vals) > 0:
        plt.hist(mua_vals, bins=bins, alpha=0.5, density=True, label="mua")

    if len(noise_vals) > 0:
        plt.hist(noise_vals, bins=bins, alpha=0.5, density=True, label="noise")

    if len(non_somatic_mua_vals) > 0:
        plt.hist(non_somatic_mua_vals, bins=bins, alpha=0.5, density=True, label="non-somatic_mua")

    if len(non_somatic_good_vals) > 0:
        plt.hist(non_somatic_good_vals, bins=bins, alpha=0.5, density=True, label="non-somatic_good")

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel(plot_metric)
    plt.ylabel("Density")
    plt.title(f"Distribution of {plot_metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_boxplot(df, metric, label_col, output_path):
    plot_metric = get_plot_series(df, metric)

    sub = df[[label_col, plot_metric]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    data = []
    labels = []

    # for lbl in [GOOD_LABEL, MUA_LABEL, NOISE_LABEL, UNSORTED_LABEL]:
    for lbl in [GOOD_LABEL, MUA_LABEL, NOISE_LABEL, NON_SOMATIC_MUA_LABEL, NON_SOMATIC_GOOD_LABEL]:
        vals = sub.loc[sub[label_col] == lbl, plot_metric].values
        if len(vals) > 0:
            data.append(vals)
            labels.append(lbl)

    if len(data) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel(plot_metric)
    plt.title(f"Boxplot of {plot_metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_good_vs_nongood_hist(df, metric, label_col, output_path):
    plot_metric = get_plot_series(df, metric)

    sub = df[[label_col, plot_metric]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    sub["binary_label"] = np.where(
        sub[label_col].isin([GOOD_LABEL, NON_SOMATIC_GOOD_LABEL]),
        "good",
        "non_good"
    )

    good_vals = sub.loc[sub["binary_label"] == "good", plot_metric].values
    nongood_vals = sub.loc[sub["binary_label"] == "non_good", plot_metric].values
    all_vals = sub[plot_metric].values

    if len(all_vals) == 0:
        return

    xlim = get_axis_limits(
        all_vals,
        mode=XLIM_MODE,
        lower=XLIM_LOWER,
        upper=XLIM_UPPER,
    )

    plt.figure(figsize=(8, 5))

    bins = 40

    if len(good_vals) > 0:
        plt.hist(good_vals, bins=bins, alpha=0.5, density=True, label="good")

    if len(nongood_vals) > 0:
        plt.hist(nongood_vals, bins=bins, alpha=0.5, density=True, label="non_good")

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel(plot_metric)
    plt.ylabel("Density")
    plt.title(f"Good vs Non-good: {plot_metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []

for metric in METRICS_TO_PLOT:
    plot_metric = get_plot_series(df, metric)

    if plot_metric not in df.columns:
        print(f"Skipping missing metric: {metric}")
        continue

    # for lbl in [GOOD_LABEL, MUA_LABEL, NOISE_LABEL, UNSORTED_LABEL]:
    for lbl in [GOOD_LABEL, MUA_LABEL, NOISE_LABEL, NON_SOMATIC_MUA_LABEL, NON_SOMATIC_GOOD_LABEL]:
        vals = (
            df.loc[df[LABEL_COL_MANUAL] == lbl, plot_metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(vals) == 0:
            continue

        summary_rows.append({
            "metric": plot_metric,
            "label": lbl,
            "n": len(vals),
            "mean": vals.mean(),
            "std": vals.std(),
            "median": vals.median(),
            "q25": vals.quantile(0.25),
            "q75": vals.quantile(0.75),
            "min": vals.min(),
            "max": vals.max(),
        })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "metric_summary_by_label.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"Saved summary table to: {summary_csv}")


# ============================================================
# PLOTTING
# ============================================================

for metric in METRICS_TO_PLOT:

    plot_metric = get_plot_series(df, metric)

    if plot_metric not in df.columns:
        continue

    hist_path = OUTPUT_DIR / f"{plot_metric}_hist_4class.png"
    box_path = OUTPUT_DIR / f"{plot_metric}_box_4class.png"
    binary_hist_path = OUTPUT_DIR / f"{plot_metric}_hist_good_vs_nongood.png"

    save_histogram(df, metric, LABEL_COL_MANUAL, hist_path)
    save_boxplot(df, metric, LABEL_COL_MANUAL, box_path)
    save_good_vs_nongood_hist(df, metric, LABEL_COL_MANUAL, binary_hist_path)

print("Done.")