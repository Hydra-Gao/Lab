import numpy as np
import spikeinterface.extractors as se
import spikeinterface as si
from pathlib import Path
from spikeinterface.core import BinaryFolderRecording

RAW_FOLDER = r"G:\Lab\Raw_data\TG964\2026-06-24_13-48-39"
OUT_ROOT = Path(r"G:\Lab\Processing\Working_dir_processing\split_experiments")

EXP1_SEGMENTS = [0, 1]
EXP2_SEGMENT = 3
EXP3_SEGMENT = 4

EXP2_CUT_TIME_SEC = 1864.0
EXP3_CUT_TIME_SEC = 1768.0

raw_rec = se.read_neuralynx(RAW_FOLDER)

print("Raw data read")

exp1 = BinaryFolderRecording(OUT_ROOT / "experiment_1")
exp2 = BinaryFolderRecording(OUT_ROOT / "experiment_2")
exp3 = BinaryFolderRecording(OUT_ROOT / "experiment_3")

fs = raw_rec.get_sampling_frequency()

exp2_cut_frame = int(round(EXP2_CUT_TIME_SEC * fs))
exp3_cut_frame = int(round(EXP3_CUT_TIME_SEC * fs))

print("Cut frames calculated")

def compare_trace_block(
    raw_recording,
    saved_recording,
    raw_segment_index,
    saved_segment_index,
    raw_start_frame,
    saved_start_frame,
    num_frames=32000,
    channel_ids=None,
    label=""
):
    if channel_ids is None:
        channel_ids = raw_recording.channel_ids

    raw_traces = raw_recording.get_traces(
        segment_index=raw_segment_index,
        start_frame=raw_start_frame,
        end_frame=raw_start_frame + num_frames,
        channel_ids=channel_ids,
        return_scaled=False,
    )

    saved_traces = saved_recording.get_traces(
        segment_index=saved_segment_index,
        start_frame=saved_start_frame,
        end_frame=saved_start_frame + num_frames,
        channel_ids=channel_ids,
        return_scaled=False,
    )

    same = np.array_equal(raw_traces, saved_traces)
    max_abs_diff = np.max(np.abs(raw_traces.astype("float64") - saved_traces.astype("float64")))

    print(f"\n{label}")
    print("  same:", same)
    print("  max_abs_diff:", max_abs_diff)


# Experiment 1:
# saved exp1 segment 0 应该等于 raw segment EXP1_SEGMENTS[0]
compare_trace_block(
    raw_rec, exp1,
    raw_segment_index=EXP1_SEGMENTS[0],
    saved_segment_index=0,
    raw_start_frame=10000,
    saved_start_frame=10000,
    label="Exp1 segment 0 check"
)

# saved exp1 segment 1 应该等于 raw segment EXP1_SEGMENTS[1]
compare_trace_block(
    raw_rec, exp1,
    raw_segment_index=EXP1_SEGMENTS[1],
    saved_segment_index=1,
    raw_start_frame=10000,
    saved_start_frame=10000,
    label="Exp1 segment 1 check"
)

print("\nComparison 1 done.")

# Experiment 2:
# saved exp2 segment 0 = raw EXP2_SEGMENT 从 0 到 cut_frame
compare_trace_block(
    raw_rec, exp2,
    raw_segment_index=EXP2_SEGMENT,
    saved_segment_index=0,
    raw_start_frame=10000,
    saved_start_frame=10000,
    label="Exp2 segment 0 check"
)

# saved exp2 segment 1 = raw EXP2_SEGMENT 从 cut_frame 之后开始
compare_trace_block(
    raw_rec, exp2,
    raw_segment_index=EXP2_SEGMENT,
    saved_segment_index=1,
    raw_start_frame=exp2_cut_frame + 10000,
    saved_start_frame=10000,
    label="Exp2 segment 1 check"
)

print("\nComparison 2 done.")

# Experiment 3:
compare_trace_block(
    raw_rec, exp3,
    raw_segment_index=EXP3_SEGMENT,
    saved_segment_index=0,
    raw_start_frame=10000,
    saved_start_frame=10000,
    label="Exp3 segment 0 check"
)

compare_trace_block(
    raw_rec, exp3,
    raw_segment_index=EXP3_SEGMENT,
    saved_segment_index=1,
    raw_start_frame=exp3_cut_frame + 10000,
    saved_start_frame=10000,
    label="Exp3 segment 1 check"
)
print("\nComparison 3 done.")