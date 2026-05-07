# 04_label_spikes.py

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


def label_one_state(spikes, trials, state):
    """Label spikes falling inside one stimulus state."""

    start_col = f"{state}_start_sec"
    end_col = f"{state}_end_sec"

    rows = []

    for _, tr in trials.iterrows():
        mask = (
            (spikes["spike_time_sec"] >= tr[start_col])
            & (spikes["spike_time_sec"] < tr[end_col])
        )

        sp = spikes.loc[mask].copy()

        if sp.empty:
            continue

        sp["trial_id"] = tr["trial_id"]
        sp["stimulus_state"] = state

        sp["direction"] = tr["direction"]
        sp["orientation"] = tr["orientation"]
        sp["pattern"] = tr["pattern"]
        sp["speed"] = tr["speed"]

        sp["time_from_state_onset"] = sp["spike_time_sec"] - tr[start_col]
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

    labeled_parts = []

    for state in ["blank", "static", "moving"]:
        labeled = label_one_state(spikes, trials, state)
        print(f"{state}: {len(labeled)} spikes")
        labeled_parts.append(labeled)

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
    