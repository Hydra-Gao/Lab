# SB04b_label_spikes_vbc_3screen.py

import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


STATES_TO_LABEL = ["static", "moving"]
# blank duration is 0 in the VbC three-screen script,
# so we usually do not need to label blank spikes.


def require_columns(df, cols, name):
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def label_one_state(spikes, trials, state):
    """
    Label spikes falling inside one stimulus state.

    For each spike, keep:
        - unit / spike timing
        - trial id
        - state label
        - VbC three-screen pattern metadata
        - time relative to current state onset
        - time relative to moving onset
    """

    start_col = f"{state}_start_sec"
    end_col = f"{state}_end_sec"

    if start_col not in trials.columns or end_col not in trials.columns:
        raise ValueError(
            f"trial_table.csv does not contain {start_col} / {end_col}"
        )

    rows = []

    for _, tr in trials.iterrows():

        start_time = float(tr[start_col])
        end_time = float(tr[end_col])

        # Skip zero-duration or invalid states.
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            continue

        if end_time <= start_time:
            continue

        mask = (
            (spikes["spike_time_sec"] >= start_time)
            & (spikes["spike_time_sec"] < end_time)
        )

        sp = spikes.loc[mask].copy()

        if sp.empty:
            continue

        # -----------------------------
        # Trial / state identity
        # -----------------------------

        sp["trial_id"] = tr["trial_id"]
        sp["stimulus_state"] = state

        # -----------------------------
        # VbC condition labels
        # -----------------------------

        sp["trial_number_randomized"] = tr["trial_number_randomized"]
        sp["replicate"] = tr["replicate"]

        sp["pattern"] = tr["pattern"]
        sp["biological_label"] = tr["biological_label"]

        sp["left_movement"] = tr["left_movement"]
        sp["front_movement"] = tr["front_movement"]
        sp["right_movement"] = tr["right_movement"]

        # -----------------------------
        # Speed / grating metadata
        # -----------------------------

        sp["tf_hz"] = tr["tf_hz"]
        sp["sf_cpd"] = tr["sf_cpd"]
        sp["phase_step"] = tr["phase_step"]
        sp["speed_deg_per_sec"] = tr["speed_deg_per_sec"]

        # -----------------------------
        # Timing relative to trial
        # -----------------------------

        sp["time_from_state_onset"] = sp["spike_time_sec"] - start_time
        sp["time_from_moving_onset"] = (
            sp["spike_time_sec"] - tr["moving_start_sec"]
        )

        rows.append(sp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main():
    print("===== Label spikes by VbC three-screen trial / stimulus state =====")

    spikes_path = ANALYSIS_OUTPUT_DIR / "spikes.csv"
    trial_path = ANALYSIS_OUTPUT_DIR / "trial_table.csv"

    spikes = pd.read_csv(spikes_path)
    trials = pd.read_csv(trial_path)

    print(f"Spikes: {len(spikes)}")
    print(f"Trials: {len(trials)}")

    # -----------------------------
    # Required input columns
    # -----------------------------

    require_columns(
        spikes,
        [
            "unit_id",
            "spike_time_sec",
        ],
        "spikes.csv",
    )

    require_columns(
        trials,
        [
            "trial_id",
            "trial_number_randomized",
            "replicate",

            "pattern",
            "biological_label",
            "left_movement",
            "front_movement",
            "right_movement",

            "tf_hz",
            "sf_cpd",
            "phase_step",
            "speed_deg_per_sec",

            "static_start_sec",
            "static_end_sec",
            "moving_start_sec",
            "moving_end_sec",
        ],
        "trial_table.csv",
    )

    # -----------------------------
    # Label spikes
    # -----------------------------

    labeled_parts = []

    for state in STATES_TO_LABEL:
        labeled = label_one_state(spikes, trials, state)
        print(f"{state}: {len(labeled)} spikes")
        labeled_parts.append(labeled)

    if len(labeled_parts) == 0:
        labeled_spikes = pd.DataFrame()
    else:
        labeled_parts = [
            df for df in labeled_parts
            if df is not None and not df.empty
        ]

        if len(labeled_parts) == 0:
            labeled_spikes = pd.DataFrame()
        else:
            labeled_spikes = pd.concat(labeled_parts, ignore_index=True)

    if labeled_spikes.empty:
        print("Warning: no spikes were labeled.")
    else:
        labeled_spikes = labeled_spikes.sort_values(
            ["unit_id", "trial_id", "spike_time_sec"]
        ).reset_index(drop=True)

    # -----------------------------
    # Save
    # -----------------------------

    out_path = ANALYSIS_OUTPUT_DIR / "labeled_spikes.csv"
    labeled_spikes.to_csv(out_path, index=False)

    print("\n===== Saved =====")
    print(out_path)

    print("\nFirst few labeled spikes:")
    print(labeled_spikes.head())


if __name__ == "__main__":
    main()