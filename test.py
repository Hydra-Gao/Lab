# import spikeinterface.full as si
# from config_local import WORKING_DIR, SEGMENT_INDEX_TO_USE

# # 这里按你的实际 sorter 改
# SORTER_NAME = "kilosort4"

# recording = si.load(WORKING_DIR / "preprocessed_M12")

# # 和你 sorting 时保持一致：你 SS2 里用了 select_segments([1])
# recording_seg = recording.select_segments([SEGMENT_INDEX_TO_USE])

# sorting = si.load(WORKING_DIR / f"{SORTER_NAME}_M12_test_output")

# n_frames = recording_seg.get_num_frames(segment_index=0)

# print("Recording frames:", n_frames)
# print("Recording duration sec:", recording_seg.get_total_duration())

# total_excess = 0

# for unit_id in sorting.get_unit_ids():
#     spikes = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0)

#     excess_mask = spikes >= n_frames
#     n_excess = excess_mask.sum()

#     if n_excess > 0:
#         total_excess += n_excess
#         print(
#             f"Unit {unit_id}: {n_excess} excess spikes "
#             f"/ {len(spikes)} total, "
#             f"max_frame={spikes.max()}"
#         )

# print("\nTotal excess spikes:", total_excess)

import pandas as pd

DIR = r"G:\Lab\Processing\Output_dir_almost\analysis_TG915_2026-05-27_16-32-17_kilosort4"

# # after SB01
# ttl = pd.read_csv(DIR + "/ttl_events_global.csv")
# print(ttl.groupby("original_segment_index").size())

# # after SB02
# trials = pd.read_csv(DIR + "/trial_table.csv")
# print(trials.groupby(["original_segment_index", "screen_role"]).size())
# print(trials.groupby(["screen_role", "speed", "direction"]).size())
# print((trials["moving_start_sec"] - trials["static_start_sec"]).describe())
# print((trials["ttl_time_sec"] - trials["moving_start_sec"]).describe())

# # after SB04
# labeled = pd.read_csv(DIR + "/labeled_spikes.csv")
# print(labeled.groupby(["stimulus_state", "screen_role", "speed"]).size())
# trials = pd.read_csv(DIR + "/trial_table.csv")
# print(trials.groupby(["screen_role", "speed"]).size())
# print(trials.groupby(["screen_role", "speed", "direction"]).size())
# trials["static_duration"] = trials["static_end_sec"] - trials["static_start_sec"]
# trials["moving_duration"] = trials["moving_end_sec"] - trials["moving_start_sec"]
# print(trials.groupby(["screen_role", "speed"])[["static_duration", "moving_duration"]].describe())
# dup = labeled.duplicated(
#     subset=["unit_id", "spike_time_sec", "stimulus_state"],
#     keep=False
# )
# print("Duplicated labeled spikes:", dup.sum())

# after SB05
# cond = pd.read_csv(DIR + "/unit_condition_summary.csv")
# print(cond.groupby(["screen_role", "speed"]).size())
# print(cond.groupby(["screen_role", "speed", "direction"])["n_trials"].describe())
trial = pd.read_csv(DIR + "/unit_trial_summary.csv")
# print(trial.groupby(["screen_role", "speed"]).size())
# print(trial.groupby(["screen_role", "speed", "direction"]).size())
# cols = [
#     "baseline_fr",
#     "static_fr",
#     "moving_fr",
#     "early_fr",
#     "sustained_fr",
#     "moving_minus_baseline",
#     "moving_minus_static",
# ]
# print(trial[cols].isna().sum())
# print(trial[cols].describe())

print(trial.groupby(["screen_role", "speed", "direction"])["pooled_baseline_fr"].mean())
# 情况 1：
# 同一个 unit × screen_role 下 pooled_baseline_fr 应该完全一样
# 情况 2：
# 同一个 unit × screen_role × direction_axis 下 pooled_baseline_fr 应该完全一样
print(
    trial.groupby(["unit_id", "screen_role", "direction_axis"])["pooled_baseline_fr"]
    .nunique()
    .describe()
)


# # after SB06
# sig = pd.read_csv(DIR + "/unit_significance_summary.csv")
# print(sig.groupby(["screen_role", "speed"])[
#     ["is_motion_baseline_responsive", "is_motion_baseline_suppressed", "is_direction_tuned_motion_baseline"]
# ].sum())