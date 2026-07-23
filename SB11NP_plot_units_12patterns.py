import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR

PLOT_MODE = "all"
SIGNIFICANCE_COLUMNS = [
    "is_motion_specific_global",
    "is_responsive_global_two_sided_gate",
    "is_suppressed_global_two_sided_gate",
    "is_motion_specific_within_pattern",
    "is_responsive_within_pattern_two_sided_gate",
    "is_suppressed_within_pattern_two_sided_gate",
]
TIME_RANGE = (-2.0, 4.0)
BIN_WIDTH = 0.1

PATTERN_ORDER = [
    "VAr", "VAl",
    "HA_leftcorner_clockwise", "HA_leftcorner_anticlockwise",
    "HA_rightcorner_clockwise", "HA_rightcorner_anticlockwise",
    "Ascent", "Descent",
    "EXPANSION_l", "EXPANSION_r", "CONTRACTION_left", "CONTRACTION_right",
]
PATTERN_LABELS_SHORT = {
    "VAr": "VAr", "VAl": "VAl",
    "HA_leftcorner_clockwise": "HA L-cw",
    "HA_leftcorner_anticlockwise": "HA L-acw",
    "HA_rightcorner_clockwise": "HA R-cw",
    "HA_rightcorner_anticlockwise": "HA R-acw",
    "Ascent": "Ascent", "Descent": "Descent",
    "EXPANSION_l": "Exp L", "EXPANSION_r": "Exp R",
    "CONTRACTION_left": "Con L", "CONTRACTION_right": "Con R",
}


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def sem(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def fmt_p(value):
    value = safe_float(value)
    if np.isnan(value):
        return "nan"
    if value < 0.0001:
        return "<1e-4"
    return f"{value:.4f}"


def short_pattern_name(pattern):
    return PATTERN_LABELS_SHORT.get(pattern, str(pattern))


def available_pattern_order(df):
    observed = [p for p in df["pattern"].dropna().unique().tolist()]
    ordered = [p for p in PATTERN_ORDER if p in observed]
    extras = sorted([p for p in observed if p not in PATTERN_ORDER])
    return ordered + extras


def add_epoch_background(ax):
    ax.axvspan(TIME_RANGE[0], 0, alpha=0.08)
    ax.axvspan(0, TIME_RANGE[1], alpha=0.08)
    ax.axvline(0, linestyle='--', linewidth=1)


def get_units_to_plot(units, sig):
    if PLOT_MODE == "all":
        return units["unit_id"].tolist()
    if sig is None:
        raise FileNotFoundError("unit_pattern_significance.csv not found, but PLOT_MODE='significant'.")
    mask = np.zeros(len(sig), dtype=bool)
    for col in SIGNIFICANCE_COLUMNS:
        if col in sig.columns:
            mask |= sig[col].fillna(False).astype(bool).to_numpy()
    return sorted(sig.loc[mask, "unit_id"].dropna().unique().tolist())


def plot_pattern_response_summary(ax, trial_unit, pattern_unit, pattern_order):
    df_trial = trial_unit[["pattern", "trial_id", "moving_minus_baseline"]].dropna(subset=["pattern", "moving_minus_baseline"]).copy()
    x = np.arange(len(pattern_order))
    rng = np.random.default_rng(42)
    for i, pattern in enumerate(pattern_order):
        vals = df_trial.loc[df_trial["pattern"] == pattern, "moving_minus_baseline"].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.7)
    mean_vals, sem_vals = [], []
    for pattern in pattern_order:
        row = pattern_unit[pattern_unit["pattern"] == pattern]
        if len(row) == 0:
            vals = df_trial.loc[df_trial["pattern"] == pattern, "moving_minus_baseline"].to_numpy(dtype=float)
            mean_vals.append(np.nanmean(vals) if len(vals) > 0 else np.nan)
            sem_vals.append(sem(vals) if len(vals) > 0 else np.nan)
        else:
            r = row.iloc[0]
            mean_vals.append(safe_float(r.get("moving_minus_baseline_mean", np.nan)))
            sem_vals.append(safe_float(r.get("moving_minus_baseline_sem", np.nan)))
    ax.errorbar(x, mean_vals, yerr=sem_vals, fmt='-o', capsize=4, linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_pattern_name(p) for p in pattern_order], rotation=45, ha='right')
    ax.set_ylabel("Moving - baseline FR")
    ax.set_title("Pattern response summary")


def plot_baseline_moving_fr_summary(ax, trial_unit, pattern_unit, pattern_order):
    df_trial = trial_unit[["pattern", "trial_id", "moving_fr"]].dropna(subset=["pattern", "moving_fr"]).copy()
    x = np.arange(len(pattern_order))
    rng = np.random.default_rng(42)
    for i, pattern in enumerate(pattern_order):
        vals = df_trial.loc[df_trial["pattern"] == pattern, "moving_fr"].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=0.65)
    baseline_mean, baseline_sem, moving_mean, moving_sem = [], [], [], []
    for pattern in pattern_order:
        row = pattern_unit[pattern_unit["pattern"] == pattern]
        if len(row) == 0:
            baseline_mean.append(np.nan); baseline_sem.append(np.nan); moving_mean.append(np.nan); moving_sem.append(np.nan)
        else:
            r = row.iloc[0]
            baseline_mean.append(safe_float(r.get("baseline_fr_mean", np.nan)))
            baseline_sem.append(safe_float(r.get("baseline_fr_sem", np.nan)))
            moving_mean.append(safe_float(r.get("moving_fr_mean", np.nan)))
            moving_sem.append(safe_float(r.get("moving_fr_sem", np.nan)))
    ax.errorbar(x, baseline_mean, yerr=baseline_sem, fmt='-o', capsize=4, linewidth=1.5, label='baseline FR')
    ax.errorbar(x, moving_mean, yerr=moving_sem, fmt='-o', capsize=4, linewidth=1.5, label='moving FR')
    ax.set_xticks(x)
    ax.set_xticklabels([short_pattern_name(p) for p in pattern_order], rotation=45, ha='right')
    ax.set_ylabel("FR (spikes/s)")
    ax.set_title("Baseline & moving FR")
    ax.legend(fontsize=8)


def draw_stats_table(ax, sig_unit, pattern_order):
    ax.axis('off')
    row_labels = [
        'mean',
        'p_perm_2s',
        'p_ttest_2s',
        'q_perm_within',
        'q_ttest_within',
    ]
    cell_text = []
    for rlabel in row_labels:
        row = []
        for pattern in pattern_order:
            sub = sig_unit[sig_unit['pattern'] == pattern]
            if sub.empty:
                row.append('nan')
                continue
            s = sub.iloc[0]
            if rlabel == 'mean':
                row.append(f"{safe_float(s.get('mean_moving_minus_baseline', np.nan)):.2f}" if pd.notna(s.get('mean_moving_minus_baseline', np.nan)) else 'nan')
            elif rlabel == 'p_perm_2s':
                row.append(fmt_p(s.get('p_motion_specific_two_sided', np.nan)))
            elif rlabel == 'p_ttest_2s':
                row.append(fmt_p(s.get('p_ttest_two_sided', np.nan)))
            elif rlabel == 'q_perm_within':
                row.append(fmt_p(s.get('q_motion_specific_within_pattern', np.nan)))
            elif rlabel == 'q_ttest_within':
                row.append(fmt_p(s.get('q_ttest_two_sided_within_pattern', np.nan)))
        cell_text.append(row)
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[short_pattern_name(p) for p in pattern_order],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)


def plot_page1(pdf, unit_id, trial_unit, pattern_unit, sig_unit, pattern_order):
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[1.0, 0.95], wspace=0.25, hspace=0.35)
    ax_response = fig.add_subplot(gs[0, 0])
    plot_pattern_response_summary(ax_response, trial_unit, pattern_unit, pattern_order)
    ax_fr = fig.add_subplot(gs[0, 1])
    plot_baseline_moving_fr_summary(ax_fr, trial_unit, pattern_unit, pattern_order)
    ax_table = fig.add_subplot(gs[1, :])
    draw_stats_table(ax_table, sig_unit, pattern_order)
    fig.suptitle(f"Unit {unit_id} | 12-pattern response summary and statistics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def plot_page2_raster(pdf, unit_id, labeled_unit, pattern_order):
    fig, ax = plt.subplots(figsize=(14, 16))
    y = 0
    yticks, ylabels = [], []
    for pattern in pattern_order:
        df_pat = labeled_unit[labeled_unit['pattern'] == pattern].copy()
        trial_ids = sorted(df_pat['trial_id'].dropna().unique())
        if len(trial_ids) == 0:
            continue
        start_y = y
        for trial_id in trial_ids:
            df_trial = df_pat[df_pat['trial_id'] == trial_id]
            spike_times = df_trial['time_from_moving_onset'].to_numpy(dtype=float)
            spike_times = spike_times[(spike_times >= TIME_RANGE[0]) & (spike_times <= TIME_RANGE[1])]
            ax.vlines(spike_times, y - 0.4, y + 0.4, linewidth=0.5)
            y += 1
        end_y = y - 1
        yticks.append((start_y + end_y) / 2)
        ylabels.append(short_pattern_name(pattern))
        ax.axhline(y - 0.5, linewidth=0.5, alpha=0.4)
        y += 1
    add_epoch_background(ax)
    ax.set_xlim(TIME_RANGE)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel('Time from moving onset (s)')
    ax.set_ylabel('Pattern / trials')
    ax.set_title(f'Unit {unit_id} | 12-pattern raster')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_psth_one_pattern(ax, labeled_unit, pattern):
    df_pat = labeled_unit[labeled_unit['pattern'] == pattern].copy()
    bins = np.arange(TIME_RANGE[0], TIME_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    centers = bins[:-1] + BIN_WIDTH / 2
    n_trials = df_pat['trial_id'].nunique()
    if n_trials > 0:
        spike_times = df_pat['time_from_moving_onset'].to_numpy(dtype=float)
        spike_times = spike_times[(spike_times >= TIME_RANGE[0]) & (spike_times <= TIME_RANGE[1])]
        counts, _ = np.histogram(spike_times, bins=bins)
        fr = counts / n_trials / BIN_WIDTH
        ax.plot(centers, fr, linewidth=1.2)
    add_epoch_background(ax)
    ax.set_xlim(TIME_RANGE)
    ax.set_title(short_pattern_name(pattern), fontsize=9)
    if n_trials == 0:
        ax.text(0.5, 0.5, 'No trials', ha='center', va='center', transform=ax.transAxes, fontsize=8)


def plot_page3_psths(pdf, unit_id, labeled_unit):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for ax, pattern in zip(axes.flat, PATTERN_ORDER):
        plot_psth_one_pattern(ax, labeled_unit, pattern)
    for ax in axes.flat[len(PATTERN_ORDER):]:
        ax.axis('off')
    for ax in axes[-1, :]:
        ax.set_xlabel('Time from moving onset (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('FR (spikes/s)')
    fig.suptitle(f'Unit {unit_id} | 12-pattern PSTHs', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def plot_one_unit(unit_id, labeled, trial_summary, pattern_summary, sig, out_dir):
    labeled_unit = labeled[labeled['unit_id'] == unit_id].copy()
    trial_unit = trial_summary[trial_summary['unit_id'] == unit_id].copy()
    pattern_unit = pattern_summary[pattern_summary['unit_id'] == unit_id].copy()
    sig_unit = sig[sig['unit_id'] == unit_id].copy() if sig is not None else pd.DataFrame()
    if trial_unit.empty:
        return
    pattern_order = available_pattern_order(trial_unit)
    out_path = out_dir / f"unit_{unit_id}_12pattern_summary.pdf"
    with PdfPages(out_path) as pdf:
        plot_page1(pdf, unit_id, trial_unit, pattern_unit, sig_unit, pattern_order)
        plot_page2_raster(pdf, unit_id, labeled_unit, pattern_order)
        plot_page3_psths(pdf, unit_id, labeled_unit)
    print(f"Saved: {out_path}")


def main():
    print('===== Plot NP 12-pattern unit summaries =====')
    plot_dir = ANALYSIS_OUTPUT_DIR / 'plots' / 'units_12patterns_np'
    plot_dir.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / 'labeled_spikes.csv')
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / 'curated_units.csv')
    trial_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / 'unit_trial_summary.csv')
    pattern_summary = pd.read_csv(ANALYSIS_OUTPUT_DIR / 'unit_pattern_summary.csv')
    sig_path = ANALYSIS_OUTPUT_DIR / 'unit_pattern_significance.csv'
    sig = pd.read_csv(sig_path) if sig_path.exists() else None
    if 'trial_kind' in labeled.columns:
        labeled = labeled[labeled['trial_kind'] == 'optimal_3screen_12pattern'].copy()
    if 'trial_kind' in trial_summary.columns:
        trial_summary = trial_summary[trial_summary['trial_kind'] == 'optimal_3screen_12pattern'].copy()
    require_columns(labeled, ['unit_id', 'trial_id', 'pattern', 'time_from_moving_onset'], 'labeled_spikes.csv')
    require_columns(trial_summary, ['unit_id', 'trial_id', 'pattern', 'moving_fr', 'moving_minus_baseline'], 'unit_trial_summary.csv')
    require_columns(pattern_summary, ['unit_id', 'pattern'], 'unit_pattern_summary.csv')
    units_to_plot = get_units_to_plot(units, sig)
    print(f'Units to plot: {len(units_to_plot)}')
    for unit_id in units_to_plot:
        plot_one_unit(unit_id, labeled, trial_summary, pattern_summary, sig, plot_dir)
    print(plot_dir)


if __name__ == '__main__':
    main()
