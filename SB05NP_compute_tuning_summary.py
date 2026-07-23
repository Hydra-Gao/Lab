"""
Handles two trial types in the same trial_table/labeled_spikes files:
    1) single_screen_8direction
    2) optimal_3screen_12pattern

Main outputs:
    unit_trial_summary.csv
    unit_condition_summary.csv          # 8-direction only
    unit_tuning_summary.csv             # 8-direction only
    unit_speed_tuning_summary.csv       # alias of unit_tuning_summary.csv
    unit_pattern_summary.csv            # 12-pattern only

Baseline modes
--------------
"auto":
    8-direction -> screen_axis_static
    12-pattern  -> pattern_pool_static

"global_static":
    8-direction: Pool static FR within unit × active_screen_role × speed.
    12-pattern: Pool static FR within unit × speed across all 12 patterns.

"trial_window_specific":
    Use each trial's firing rate inside BASELINE_WINDOW.

"screen_axis_static":
    8-direction only. Pool static FR within unit × screen × direction axis
    (opposite directions share one axis, e.g. 45/225).
    For 12-pattern trials, falls back to trial_window_specific.

"screen_global_static":
    8-direction only. Pool static FR within unit × screen.
    For 12-pattern trials, falls back to trial_window_specific.

"pattern_pool_static":
    12-pattern only. Pool static FR within unit × speed × predefined pattern pool.
    For 8-direction trials, falls back to trial_window_specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from SB0_config_analysis import (
    ANALYSIS_OUTPUT_DIR,
    BASELINE_WINDOW,
    EARLY_WINDOW,
    SUSTAINED_RESPONSE_WINDOW,
    MOVING_WINDOW,
)


# =====================
# User-adjustable settings
# =====================

# BASELINE_POOLING_MODE = "auto"
BASELINE_POOLING_MODE = "global_static"
# BASELINE_POOLING_MODE = "trial_window_specific"
# BASELINE_POOLING_MODE = "screen_axis_static"
# BASELINE_POOLING_MODE = "screen_global_static"
# BASELINE_POOLING_MODE = "pattern_pool_static"

# Small tolerance used only for excitation/suppression classification.
EPS = 0.0  # spikes/s

SINGLE_SCREEN_TRIAL_KIND = "single_screen_8direction"
PATTERN_TRIAL_KIND = "optimal_3screen_12pattern"

# Pool A from the later 12-pattern analysis. The remaining six patterns form pool B.
POOLED_BASELINE_POOL_A_PATTERNS = {
    "VAr",
    "VAl",
    "EXPANSION_l",
    "EXPANSION_r",
    "CONTRACTION_left",
    "CONTRACTION_right",
}

VALID_BASELINE_MODES = {
    "auto",
    "global_static",
    "trial_window_specific",
    "screen_axis_static",
    "screen_global_static",
    "pattern_pool_static",
}


# =====================
# Helpers
# =====================


def sem(x):
    """Standard error of the mean."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def firing_rate(count, window):
    """Convert a spike count to spikes/s."""
    duration = float(window[1]) - float(window[0])
    if duration <= 0:
        return np.nan
    return float(count) / duration


def build_spike_time_lookup(labeled):
    """Create {(unit_id, trial_id): sorted relative spike-time array}."""
    lookup = {}
    grouped = labeled.groupby(["unit_id", "trial_id"], sort=False)
    for key, df in grouped:
        lookup[key] = np.sort(
            df["time_from_moving_onset"].dropna().to_numpy(dtype=float)
        )
    return lookup


def count_lookup(lookup, unit_id, trial_id, t0, t1):
    """Count spikes in [t0, t1) using binary search."""
    times = lookup.get((unit_id, trial_id))
    if times is None or len(times) == 0:
        return 0
    left = np.searchsorted(times, t0, side="left")
    right = np.searchsorted(times, t1, side="left")
    return int(right - left)


def get_first_nonnull(series, default=np.nan):
    s = pd.Series(series).dropna()
    return default if len(s) == 0 else s.iloc[0]


def get_pattern_baseline_pool(pattern):
    if pattern in POOLED_BASELINE_POOL_A_PATTERNS:
        return "VA_EXP_CON_pool"
    return "OTHER_6_pool"


def resolve_row_baseline_mode(trial_kind):
    """Resolve auto/fallback behavior separately for the two trial types."""
    if BASELINE_POOLING_MODE == "auto":
        if trial_kind == SINGLE_SCREEN_TRIAL_KIND:
            return "screen_axis_static"
        if trial_kind == PATTERN_TRIAL_KIND:
            return "pattern_pool_static"
        return "trial_window_specific"

    if BASELINE_POOLING_MODE == "global_static":
        if trial_kind == SINGLE_SCREEN_TRIAL_KIND:
            return "screen_global_static"
        if trial_kind == PATTERN_TRIAL_KIND:
            return "pattern_global_static"
        return "trial_window_specific"

    if trial_kind == SINGLE_SCREEN_TRIAL_KIND:
        if BASELINE_POOLING_MODE in {
            "trial_window_specific",
            "screen_axis_static",
            "screen_global_static",
        }:
            return BASELINE_POOLING_MODE
        return "trial_window_specific"

    if trial_kind == PATTERN_TRIAL_KIND:
        if BASELINE_POOLING_MODE in {
            "trial_window_specific",
            "pattern_pool_static",
        }:
            return BASELINE_POOLING_MODE
        return "trial_window_specific"

    return "trial_window_specific"


def compute_vector_strength(direction_response):
    """Return vector-sum direction and normalized vector strength."""
    df = direction_response.dropna(subset=["direction", "response"]).copy()
    if df.empty:
        return np.nan, np.nan

    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["response"])
    if df.empty or df["response"].sum() <= 0:
        return np.nan, np.nan

    angles = np.deg2rad(df["direction"].astype(float).to_numpy())
    responses = df["response"].astype(float).to_numpy()
    vec = np.sum(responses * np.exp(1j * angles))
    total = np.sum(responses)
    return float(np.rad2deg(np.angle(vec)) % 360), float(np.abs(vec) / total)


def compute_dsi_details(direction_response):
    """Return preferred/opposite details and DSI for nonnegative responses."""
    df = direction_response.dropna(subset=["direction", "response"]).copy()
    if df.empty:
        return (np.nan,) * 5

    df["direction"] = df["direction"].astype(float) % 360
    df["response"] = df["response"].astype(float)
    if df["response"].sum() <= 0:
        return (np.nan,) * 5

    pref_row = df.loc[df["response"].idxmax()]
    pref_dir = float(pref_row["direction"])
    r_pref = float(pref_row["response"])

    nominal_opp = (pref_dir + 180) % 360
    df["opp_distance"] = np.abs(
        ((df["direction"] - nominal_opp + 180) % 360) - 180
    )
    opp_row = df.loc[df["opp_distance"].idxmin()]
    opp_dir = float(opp_row["direction"])
    r_opp = float(opp_row["response"])
    dsi = np.nan if r_pref + r_opp <= 0 else (r_pref - r_opp) / (r_pref + r_opp)
    return pref_dir, r_pref, opp_dir, r_opp, float(dsi)


def classify_signed_response(df_condition_unit, eps=EPS):
    r = df_condition_unit["moving_minus_baseline"].dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return "no_data", 0, 0

    n_pos = int(np.sum(r > eps))
    n_neg = int(np.sum(r < -eps))
    if n_pos > 0 and n_neg == 0:
        return "all_excited_or_nonnegative", n_pos, n_neg
    if n_neg > 0 and n_pos == 0:
        return "all_suppressed_or_nonpositive", n_pos, n_neg
    if n_pos > 0 and n_neg > 0:
        return "mixed_excited_and_suppressed", n_pos, n_neg
    return "flat_or_zero", n_pos, n_neg


def add_pooled_baselines(unit_trial_summary):
    """Calculate all supported pooled baselines, then select the requested one."""
    df = unit_trial_summary.copy()

    df["direction_float"] = pd.to_numeric(df["direction"], errors="coerce") % 360
    df["direction_axis"] = df["direction_float"] % 180
    df["pattern_baseline_pool"] = df["pattern"].map(get_pattern_baseline_pool)
    df["baseline_pooling_mode"] = df["trial_kind"].map(resolve_row_baseline_mode)

    # Start with the per-trial baseline-window result.
    df["pooled_baseline_fr"] = df["baseline_window_fr"]
    df["pooled_baseline_fr_sem"] = np.nan
    df["pooled_baseline_n_trials"] = 1

    # 8-direction: unit × active screen × direction axis.
    mask = df["baseline_pooling_mode"].eq("screen_axis_static")
    if mask.any():
        keys = ["unit_id", "active_screen_role", "direction_axis"]
        stats = (
            df.loc[mask]
            .groupby(keys, dropna=False)
            .agg(
                pooled=("static_fr", "mean"),
                pooled_sem=("static_fr", sem),
                pooled_n=("trial_id", "nunique"),
            )
            .reset_index()
        )
        merged = df.loc[mask, keys].merge(stats, on=keys, how="left")
        df.loc[mask, "pooled_baseline_fr"] = merged["pooled"].to_numpy()
        df.loc[mask, "pooled_baseline_fr_sem"] = merged["pooled_sem"].to_numpy()
        df.loc[mask, "pooled_baseline_n_trials"] = merged["pooled_n"].to_numpy()

    # 8-direction: unit × active screen global static baseline.
    mask = df["baseline_pooling_mode"].eq("screen_global_static")
    if mask.any():
        keys = ["unit_id", "active_screen_role"]
        stats = (
            df.loc[mask]
            .groupby(keys, dropna=False)
            .agg(
                pooled=("static_fr", "mean"),
                pooled_sem=("static_fr", sem),
                pooled_n=("trial_id", "nunique"),
            )
            .reset_index()
        )
        merged = df.loc[mask, keys].merge(stats, on=keys, how="left")
        df.loc[mask, "pooled_baseline_fr"] = merged["pooled"].to_numpy()
        df.loc[mask, "pooled_baseline_fr_sem"] = merged["pooled_sem"].to_numpy()
        df.loc[mask, "pooled_baseline_n_trials"] = merged["pooled_n"].to_numpy()

    # 12-pattern: unit × speed × predefined pattern pool.
    mask = df["baseline_pooling_mode"].eq("pattern_pool_static")
    if mask.any():
        keys = ["unit_id", "speed", "pattern_baseline_pool"]
        stats = (
            df.loc[mask]
            .groupby(keys, dropna=False)
            .agg(
                pooled=("static_fr", "mean"),
                pooled_sem=("static_fr", sem),
                pooled_n=("trial_id", "nunique"),
            )
            .reset_index()
        )
        merged = df.loc[mask, keys].merge(stats, on=keys, how="left")
        df.loc[mask, "pooled_baseline_fr"] = merged["pooled"].to_numpy()
        df.loc[mask, "pooled_baseline_fr_sem"] = merged["pooled_sem"].to_numpy()
        df.loc[mask, "pooled_baseline_n_trials"] = merged["pooled_n"].to_numpy()

    # 12-pattern: unit × speed global static baseline.
    mask = df["baseline_pooling_mode"].eq("pattern_global_static")
    if mask.any():
        keys = ["unit_id", "speed"]
        stats = (
            df.loc[mask]
            .groupby(keys, dropna=False)
            .agg(
                pooled=("static_fr", "mean"),
                pooled_sem=("static_fr", sem),
                pooled_n=("trial_id", "nunique"),
            )
            .reset_index()
        )
        merged = (df.loc[mask, keys].merge(stats, on=keys, how="left"))
        df.loc[mask, "pooled_baseline_fr"] = merged["pooled"].to_numpy()
        df.loc[mask, "pooled_baseline_fr_sem"] = merged["pooled_sem"].to_numpy()
        df.loc[mask, "pooled_baseline_n_trials"] = merged["pooled_n"].to_numpy()


    df["baseline_fr"] = df["pooled_baseline_fr"]
    df["baseline_fr_sem"] = df["pooled_baseline_fr_sem"]
    df["moving_minus_baseline"] = df["moving_fr"] - df["baseline_fr"]
    df["early_minus_baseline"] = df["early_fr"] - df["baseline_fr"]
    df["sustained_minus_baseline"] = df["sustained_fr"] - df["baseline_fr"]

    return df


# =====================
# Main
# =====================


def main():
    if BASELINE_POOLING_MODE not in VALID_BASELINE_MODES:
        raise ValueError(
            f"Invalid BASELINE_POOLING_MODE={BASELINE_POOLING_MODE!r}. "
            f"Choose from {sorted(VALID_BASELINE_MODES)}"
        )

    print("===== Compute unified Neuropixels tuning/pattern summary =====")
    print(f"BASELINE_POOLING_MODE = {BASELINE_POOLING_MODE}")

    labeled = pd.read_csv(ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv")
    trials = pd.read_csv(ANALYSIS_OUTPUT_DIR / "trial_table.csv")
    units = pd.read_csv(ANALYSIS_OUTPUT_DIR / "curated_units.csv")

    required_labeled = {"unit_id", "trial_id", "time_from_moving_onset"}
    required_trials = {
        "trial_id",
        "trial_kind",
        "static_start_sec",
        "static_end_sec",
        "moving_start_sec",
        "moving_end_sec",
    }
    missing_labeled = required_labeled - set(labeled.columns)
    missing_trials = required_trials - set(trials.columns)
    if missing_labeled:
        raise ValueError(f"labeled_spikes.csv missing: {sorted(missing_labeled)}")
    if missing_trials:
        raise ValueError(f"trial_table.csv missing: {sorted(missing_trials)}")

    # Normalize fields used in grouping.
    for col in ["trial_kind", "active_screen_role", "speed_label", "pattern"]:
        if col in trials.columns:
            trials[col] = trials[col].astype("string").str.strip()

    unit_ids = units["unit_id"].tolist()
    spike_lookup = build_spike_time_lookup(labeled)

    print(f"Units: {len(unit_ids)}")
    print(f"Trials: {len(trials)}")
    print(f"Labeled spikes: {len(labeled)}")
    print("\nTrial kinds:")
    print(trials["trial_kind"].value_counts(dropna=False))

    rows = []

    for unit_id in unit_ids:
        for _, tr in trials.iterrows():
            trial_id = tr["trial_id"]

            static_window = (
                float(tr["static_start_sec"] - tr["moving_start_sec"]),
                float(tr["static_end_sec"] - tr["moving_start_sec"]),
            )

            baseline_count = count_lookup(
                spike_lookup, unit_id, trial_id, BASELINE_WINDOW[0], BASELINE_WINDOW[1]
            )
            static_count = count_lookup(
                spike_lookup, unit_id, trial_id, static_window[0], static_window[1]
            )
            early_count = count_lookup(
                spike_lookup, unit_id, trial_id, EARLY_WINDOW[0], EARLY_WINDOW[1]
            )
            sustained_count = count_lookup(
                spike_lookup,
                unit_id,
                trial_id,
                SUSTAINED_RESPONSE_WINDOW[0],
                SUSTAINED_RESPONSE_WINDOW[1],
            )
            moving_count = count_lookup(
                spike_lookup, unit_id, trial_id, MOVING_WINDOW[0], MOVING_WINDOW[1]
            )

            baseline_window_fr = firing_rate(baseline_count, BASELINE_WINDOW)
            static_fr = firing_rate(static_count, static_window)
            early_fr = firing_rate(early_count, EARLY_WINDOW)
            sustained_fr = firing_rate(sustained_count, SUSTAINED_RESPONSE_WINDOW)
            moving_fr = firing_rate(moving_count, MOVING_WINDOW)

            rows.append(
                {
                    "unit_id": unit_id,
                    "trial_id": trial_id,
                    "trial_number_overall": tr.get("trial_number_overall", np.nan),
                    "replicate": tr.get("replicate", np.nan),
                    "condition_order": tr.get("condition_order", np.nan),
                    "condition_name": tr.get("condition_name", np.nan),
                    "trial_kind": tr.get("trial_kind", np.nan),
                    "trial_within_condition": tr.get("trial_within_condition", np.nan),
                    "active_screen_role": tr.get("active_screen_role", np.nan),
                    "direction": tr.get("direction", np.nan),
                    "orientation": tr.get("orientation", np.nan),
                    "pattern": tr.get("pattern", np.nan),
                    "biological_label": tr.get("biological_label", np.nan),
                    "speed": tr.get("speed", np.nan),
                    "speed_label": tr.get("speed_label", np.nan),
                    "recording_site_side": tr.get("recording_site_side", np.nan),
                    "ipsilateral_screen_role": tr.get("ipsilateral_screen_role", np.nan),
                    "contralateral_screen_role": tr.get("contralateral_screen_role", np.nan),
                    "left_movement": tr.get("left_movement", np.nan),
                    "front_movement": tr.get("front_movement", np.nan),
                    "right_movement": tr.get("right_movement", np.nan),
                    "alignment_residual_ms": tr.get("alignment_residual_ms", np.nan),
                    "baseline_count": baseline_count,
                    "static_count": static_count,
                    "early_count": early_count,
                    "sustained_count": sustained_count,
                    "moving_count": moving_count,
                    "baseline_window_fr": baseline_window_fr,
                    "static_fr": static_fr,
                    "early_fr": early_fr,
                    "sustained_fr": sustained_fr,
                    "moving_fr": moving_fr,
                    "moving_minus_static": moving_fr - static_fr,
                }
            )

    unit_trial_summary = pd.DataFrame(rows)
    unit_trial_summary = add_pooled_baselines(unit_trial_summary)

    # -----------------------------
    # 8-direction outputs
    # -----------------------------

    single = unit_trial_summary.loc[
        unit_trial_summary["trial_kind"].eq(SINGLE_SCREEN_TRIAL_KIND)
    ].copy()

    condition_group_cols = [
        "unit_id",
        "active_screen_role",
        "speed",
        "speed_label",
        "direction",
        "orientation",
        "pattern",
    ]

    if not single.empty:
        unit_condition_summary = (
            single.groupby(condition_group_cols, dropna=False)
            .agg(
                n_trials=("trial_id", "nunique"),
                baseline_pooling_mode=("baseline_pooling_mode", "first"),
                pattern_baseline_pool=("pattern_baseline_pool", "first"),
                direction_axis=("direction_axis", "first"),
                baseline_fr=("baseline_fr", "mean"),
                baseline_fr_sem=("baseline_fr_sem", "first"),
                baseline_window_fr=("baseline_window_fr", "mean"),
                static_fr=("static_fr", "mean"),
                early_fr=("early_fr", "mean"),
                sustained_fr=("sustained_fr", "mean"),
                moving_fr=("moving_fr", "mean"),
                moving_minus_baseline=("moving_minus_baseline", "mean"),
                moving_minus_static=("moving_minus_static", "mean"),
                early_minus_baseline=("early_minus_baseline", "mean"),
                sustained_minus_baseline=("sustained_minus_baseline", "mean"),
                pooled_baseline_n_trials=("pooled_baseline_n_trials", "first"),
            )
            .reset_index()
        )

        unit_condition_summary["motion_baseline_positive"] = (
            unit_condition_summary["moving_minus_baseline"].clip(lower=0)
        )
        unit_condition_summary["motion_baseline_negative_strength"] = (
            -unit_condition_summary["moving_minus_baseline"].clip(upper=0)
        )

        tuning_rows = []
        tuning_groups = ["unit_id", "active_screen_role", "speed", "speed_label"]

        for keys, dfc in unit_condition_summary.groupby(tuning_groups, dropna=False):
            unit_id, screen_role, speed, speed_label = keys
            dfc = dfc.sort_values("direction").copy()
            response_class, n_pos, n_neg = classify_signed_response(dfc)

            moving_resp = dfc[["direction", "moving_fr"]].rename(
                columns={"moving_fr": "response"}
            )
            excitatory_resp = dfc[["direction", "motion_baseline_positive"]].rename(
                columns={"motion_baseline_positive": "response"}
            )
            suppressive_resp = dfc[
                ["direction", "motion_baseline_negative_strength"]
            ].rename(columns={"motion_baseline_negative_strength": "response"})

            if response_class == "all_excited_or_nonnegative":
                pd_method = "motion_minus_baseline_positive"
                primary_resp = excitatory_resp
            elif response_class == "all_suppressed_or_nonpositive":
                pd_method = "suppression_strength"
                primary_resp = suppressive_resp
            elif response_class == "mixed_excited_and_suppressed":
                pd_method = "pure_moving_fr_for_mixed"
                primary_resp = moving_resp
            else:
                pd_method = "none"
                primary_resp = pd.DataFrame(columns=["direction", "response"])

            pref, pref_r, opp, opp_r, dsi = compute_dsi_details(primary_resp)
            vec_dir, vec_strength = compute_vector_strength(primary_resp)

            moving_metrics = compute_dsi_details(moving_resp)
            moving_vec = compute_vector_strength(moving_resp)
            excitation_metrics = compute_dsi_details(excitatory_resp)
            excitation_vec = compute_vector_strength(excitatory_resp)
            suppression_metrics = compute_dsi_details(suppressive_resp)
            suppression_vec = compute_vector_strength(suppressive_resp)

            tuning_rows.append(
                {
                    "unit_id": unit_id,
                    "active_screen_role": screen_role,
                    "speed": speed,
                    "speed_label": speed_label,
                    "baseline_pooling_mode": get_first_nonnull(
                        dfc["baseline_pooling_mode"]
                    ),
                    "mean_baseline_fr": dfc["baseline_fr"].mean(),
                    "mean_static_fr": dfc["static_fr"].mean(),
                    "mean_moving_fr": dfc["moving_fr"].mean(),
                    "mean_moving_minus_baseline": dfc[
                        "moving_minus_baseline"
                    ].mean(),
                    "mean_moving_minus_static": dfc["moving_minus_static"].mean(),
                    "response_class": response_class,
                    "n_positive_directions": n_pos,
                    "n_negative_directions": n_neg,
                    "classification_eps_spikes_per_sec": EPS,
                    "pd_method": pd_method,
                    "preferred_direction": pref,
                    "preferred_response_used_for_dsi": pref_r,
                    "opposite_direction": opp,
                    "opposite_response_used_for_dsi": opp_r,
                    "dsi": dsi,
                    "vector_sum_direction": vec_dir,
                    "vector_strength": vec_strength,
                    "moving_fr_preferred_direction": moving_metrics[0],
                    "moving_fr_preferred_response": moving_metrics[1],
                    "moving_fr_opposite_direction": moving_metrics[2],
                    "moving_fr_opposite_response": moving_metrics[3],
                    "moving_fr_dsi": moving_metrics[4],
                    "moving_fr_vector_direction": moving_vec[0],
                    "moving_fr_vector_strength": moving_vec[1],
                    "excitation_preferred_direction": excitation_metrics[0],
                    "excitation_preferred_response": excitation_metrics[1],
                    "excitation_opposite_direction": excitation_metrics[2],
                    "excitation_opposite_response": excitation_metrics[3],
                    "excitation_dsi": excitation_metrics[4],
                    "excitation_vector_direction": excitation_vec[0],
                    "excitation_vector_strength": excitation_vec[1],
                    "suppression_preferred_direction": suppression_metrics[0],
                    "suppression_preferred_response": suppression_metrics[1],
                    "suppression_opposite_direction": suppression_metrics[2],
                    "suppression_opposite_response": suppression_metrics[3],
                    "suppression_dsi": suppression_metrics[4],
                    "suppression_vector_direction": suppression_vec[0],
                    "suppression_vector_strength": suppression_vec[1],
                }
            )

        unit_tuning_summary = pd.DataFrame(tuning_rows).merge(
            units, on="unit_id", how="left"
        )
    else:
        unit_condition_summary = pd.DataFrame()
        unit_tuning_summary = pd.DataFrame()

    # -----------------------------
    # 12-pattern output
    # -----------------------------

    pattern_trials = unit_trial_summary.loc[
        unit_trial_summary["trial_kind"].eq(PATTERN_TRIAL_KIND)
    ].copy()

    pattern_group_cols = [
        "unit_id",
        "speed",
        "speed_label",
        "pattern",
        "pattern_baseline_pool",
        "biological_label",
        "left_movement",
        "front_movement",
        "right_movement",
    ]

    if not pattern_trials.empty:
        unit_pattern_summary = (
            pattern_trials.groupby(pattern_group_cols, dropna=False)
            .agg(
                n_trials=("trial_id", "nunique"),
                baseline_pooling_mode=("baseline_pooling_mode", "first"),
                baseline_fr_mean=("baseline_fr", "mean"),
                baseline_fr_sem=("baseline_fr_sem", "first"),
                baseline_window_fr_mean=("baseline_window_fr", "mean"),
                baseline_window_fr_sem=("baseline_window_fr", sem),
                static_fr_mean=("static_fr", "mean"),
                static_fr_sem=("static_fr", sem),
                pooled_baseline_n_trials=("pooled_baseline_n_trials", "first"),
                early_fr_mean=("early_fr", "mean"),
                early_fr_sem=("early_fr", sem),
                sustained_fr_mean=("sustained_fr", "mean"),
                sustained_fr_sem=("sustained_fr", sem),
                moving_fr_mean=("moving_fr", "mean"),
                moving_fr_sem=("moving_fr", sem),
                moving_minus_baseline_mean=("moving_minus_baseline", "mean"),
                moving_minus_baseline_sem=("moving_minus_baseline", sem),
                moving_minus_static_mean=("moving_minus_static", "mean"),
                moving_minus_static_sem=("moving_minus_static", sem),
                early_minus_baseline_mean=("early_minus_baseline", "mean"),
                early_minus_baseline_sem=("early_minus_baseline", sem),
                sustained_minus_baseline_mean=("sustained_minus_baseline", "mean"),
                sustained_minus_baseline_sem=("sustained_minus_baseline", sem),
            )
            .reset_index()
            .merge(units, on="unit_id", how="left")
        )
    else:
        unit_pattern_summary = pd.DataFrame()

    # -----------------------------
    # Save
    # -----------------------------

    outputs = {
        "unit_trial_summary.csv": unit_trial_summary,
        "unit_condition_summary.csv": unit_condition_summary,
        "unit_tuning_summary.csv": unit_tuning_summary,
        "unit_speed_tuning_summary.csv": unit_tuning_summary,
        "unit_pattern_summary.csv": unit_pattern_summary,
    }

    print("\n===== Saved =====")
    for filename, df_out in outputs.items():
        path = ANALYSIS_OUTPUT_DIR / filename
        df_out.to_csv(path, index=False)
        print(f"{path}  rows={len(df_out)}")

    print("\nBaseline modes actually used:")
    print(unit_trial_summary["baseline_pooling_mode"].value_counts(dropna=False))

    if not unit_condition_summary.empty:
        print("\n8-direction summary counts:")
        print(
            unit_condition_summary.groupby(
                ["active_screen_role", "speed_label"], dropna=False
            ).size()
        )

    if not unit_pattern_summary.empty:
        print("\n12-pattern summary counts:")
        print(unit_pattern_summary.groupby(["speed_label"], dropna=False).size())


if __name__ == "__main__":
    main()
