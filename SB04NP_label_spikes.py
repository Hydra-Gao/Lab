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

        optional_metadata_cols = [
            "trial_number_overall",
            "replicate",
            "condition_order",
            "condition_name",
            "trial_kind",
            "trial_within_condition",
            "active_screen_role",
            "biological_label",
            "speed_label",
            "recording_site_side",
            "ipsilateral_screen_role",
            "contralateral_screen_role",
            "left_movement",
            "front_movement",
            "right_movement",
            "alignment_residual_ms",
        ]

        for col in optional_metadata_cols:
            if col in trials.columns:
                sp[col] = tr[col]

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
        start_col = f"{state}_start_sec"
        end_col = f"{state}_end_sec"

        # Skip states that are absent from this experiment.
        if start_col not in trials.columns or end_col not in trials.columns:
            print(f"{state}: skipped because timing columns are missing")
            continue

        if not (
            trials[start_col].notna().any()
            and trials[end_col].notna().any()
        ):
            print(f"{state}: skipped because all timing values are NaN")
            continue

        labeled = label_one_state(spikes, trials, state)
        print(f"{state}: {len(labeled)} spikes")

        if not labeled.empty:
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
    