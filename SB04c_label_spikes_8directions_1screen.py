# 04_label_spikes.py

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


def get_available_states(trials):
    """
    Detect which stimulus states exist in trial_table.

    For three-screen old stimulus:
        blank, static, moving

    For single-screen new stimulus:
        static, moving
    """
    possible_states = ["blank", "static", "moving"]
    states = []

    for state in possible_states:
        start_col = f"{state}_start_sec"
        end_col = f"{state}_end_sec"

        if start_col in trials.columns and end_col in trials.columns:
            states.append(state)

    if len(states) == 0:
        raise ValueError(
            "No stimulus state timing columns found in trial_table. "
            "Expected columns like static_start_sec/static_end_sec "
            "or moving_start_sec/moving_end_sec."
        )

    return states


def label_one_state(spikes, trials, state):
    """Label spikes falling inside one stimulus state."""

    start_col = f"{state}_start_sec"
    end_col = f"{state}_end_sec"

    rows = []

    for _, tr in trials.iterrows():

        # Skip malformed trials if timing is missing.
        if pd.isna(tr[start_col]) or pd.isna(tr[end_col]):
            continue

        mask = (
            (spikes["spike_time_sec"] >= tr[start_col])
            & (spikes["spike_time_sec"] < tr[end_col])
        )

        sp = spikes.loc[mask].copy()

        if sp.empty:
            continue

        sp["trial_id"] = tr["trial_id"]
        sp["stimulus_state"] = state

        # Core stimulus metadata.
        # These columns should exist in trial_table.
        sp["direction"] = tr.get("direction", np.nan)
        sp["orientation"] = tr.get("orientation", np.nan)
        sp["pattern"] = tr.get("pattern", np.nan)
        sp["speed"] = tr.get("speed", np.nan)

        # Extra single-screen metadata if available.
        sp["speed_label"] = tr.get("speed_label", np.nan)
        sp["speed_deg_per_sec"] = tr.get("speed_deg_per_sec", np.nan)
        sp["tf_hz"] = tr.get("tf_hz", np.nan)
        sp["sf_cpd"] = tr.get("sf_cpd", np.nan)

        # Useful for checking which screen was used.
        sp["active_monitor_label"] = tr.get("active_monitor_label", np.nan)
        sp["active_screen_number"] = tr.get("active_screen_number", np.nan)

        # Time relative to this state onset.
        sp["time_from_state_onset"] = sp["spike_time_sec"] - tr[start_col]

        # Main alignment variable used by later analysis.
        # For static spikes this will be negative, because static occurs before moving onset.
        sp["time_from_moving_onset"] = sp["spike_time_sec"] - tr["moving_start_sec"]

        rows.append(sp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main():
    print("===== Label spikes by trial / stimulus state =====")

    spikes_path = ANALYSIS_OUTPUT_DIR / "spikes.csv"
    trial_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"

    spikes = pd.read_csv(spikes_path)
    trials = pd.read_csv(trial_path)

    print(f"Spikes: {len(spikes)}")
    print(f"Trials: {len(trials)}")

    states = get_available_states(trials)

    print("Detected states in trial_table:")
    print(states)

    labeled_parts = []

    for state in states:
        labeled = label_one_state(spikes, trials, state)
        print(f"{state}: {len(labeled)} spikes")

        if len(labeled) > 0:
            labeled_parts.append(labeled)

    if len(labeled_parts) == 0:
        print("Warning: no spikes were labeled.")
        labeled_spikes = pd.DataFrame()
    else:
        labeled_spikes = pd.concat(labeled_parts, ignore_index=True)

        labeled_spikes = labeled_spikes.sort_values(
            ["unit_id", "trial_id", "spike_time_sec"]
        ).reset_index(drop=True)

    out_path = ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv"
    labeled_spikes.to_csv(out_path, index=False)

    print("\n===== Saved =====")
    print(out_path)

    print("\nFirst few labeled spikes:")
    print(labeled_spikes.head())


if __name__ == "__main__":
    main()