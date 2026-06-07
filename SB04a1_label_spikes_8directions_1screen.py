import numpy as np
import pandas as pd

from SB0_config_analysis import ANALYSIS_OUTPUT_DIR


def get_available_states(trials):
    """
    Detect which stimulus states exist in trial_table.

    For single-screen 8-direction stimulus:
        static, moving

    If blank columns exist, it will also include blank.
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


def get_trial_value(trial_row, col, default=np.nan):
    """Safely get one value from trial row."""
    if col in trial_row.index:
        return trial_row[col]
    return default


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

        # -----------------------------
        # Trial identity
        # -----------------------------

        sp["trial_id"] = tr["trial_id"]
        sp["stimulus_state"] = state

        # -----------------------------
        # Recording / segment identity
        # -----------------------------

        sp["original_segment_index"] = get_trial_value(
            tr, "original_segment_index"
        )
        sp["ttl_index_global"] = get_trial_value(
            tr, "ttl_index_global"
        )

        # -----------------------------
        # Screen identity
        # -----------------------------

        sp["screen_role"] = get_trial_value(
            tr, "screen_role", "unknown"
        )
        sp["active_monitor_config_index"] = get_trial_value(
            tr, "active_monitor_config_index"
        )
        sp["active_monitor_label"] = get_trial_value(
            tr, "active_monitor_label"
        )
        sp["active_screen_number"] = get_trial_value(
            tr, "active_screen_number"
        )

        # -----------------------------
        # Core stimulus metadata
        # -----------------------------

        sp["direction"] = get_trial_value(tr, "direction")
        sp["orientation"] = get_trial_value(tr, "orientation")
        sp["pattern"] = get_trial_value(tr, "pattern")
        sp["speed"] = get_trial_value(tr, "speed")

        # Extra single-screen metadata.
        sp["speed_label"] = get_trial_value(tr, "speed_label")
        sp["speed_deg_per_sec"] = get_trial_value(tr, "speed_deg_per_sec")
        sp["tf_hz"] = get_trial_value(tr, "tf_hz")
        sp["sf_cpd"] = get_trial_value(tr, "sf_cpd")

        # -----------------------------
        # Timing relative to state / moving onset
        # -----------------------------

        sp["time_from_state_onset"] = sp["spike_time_sec"] - tr[start_col]

        # Main alignment variable used by later analysis.
        # For static spikes this should be negative.
        sp["time_from_moving_onset"] = (
            sp["spike_time_sec"] - tr["moving_start_sec"]
        )

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

    if "screen_role" not in trials.columns:
        print(
            "\nWarning: trial_table.csv has no screen_role column. "
            "All labeled spikes will use screen_role='unknown'. "
            "Run the updated SB02c first if you want front/left/right separation.\n"
        )
        trials["screen_role"] = "unknown"

    states = get_available_states(trials)

    print("Detected states in trial_table:")
    print(states)

    print("\nTrial counts by screen_role:")
    print(trials["screen_role"].value_counts(dropna=False))

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

    if len(labeled_spikes) > 0:
        print("\nLabeled spikes by screen_role:")
        print(labeled_spikes["screen_role"].value_counts(dropna=False))

        print("\nLabeled spikes by stimulus_state and screen_role:")
        print(
            labeled_spikes
            .groupby(["stimulus_state", "screen_role"], dropna=False)
            .size()
        )

        print("\nFirst few labeled spikes:")
        print(labeled_spikes.head())
    else:
        print("\nNo labeled spikes to show.")


if __name__ == "__main__":
    main()