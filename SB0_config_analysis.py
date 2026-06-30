from pathlib import Path
from config_local import RAW_DATA, WORKING_DIR, OUTPUT_DIR, SEGMENT_INDEX_TO_USE

BIRD = "TG964"
SESSION = "2026-06-24_13-48-39"
SORTER_NAME = "kilosort4" 
# SORTER_NAME = "mountainsort5"

# path settings
RAW_NLX_FOLDER = RAW_DATA / BIRD / SESSION

EVENTS_NEV_PATH = RAW_NLX_FOLDER / "Events.nev"

# STIMLOG_PATH = Path(r"F:\Work\UBC\Lab\Data\TG884\stimulus_log\2026-04-27_TG884_19.02_1st_spatemp_0x20c9.csv")
# STIMLOG_PATH = Path(r"E:\Lab\_analyzed\Section_1-1\2026-04-27_TG884_19.02_1st_spatemp_0xf12d.csv")
# STIMLOG_PATH = Path(r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_op_s__2026_05_27_VbC_12patterns_3screens_001.csv")
STIMLOG_PATH = Path(r"G:\Lab\Raw_data\TG964\stimulus_logs\2026-06-24_TG964_VbC_7speeds_4directions_site2_L_2026_06_24_front_single_screen_4directions_7speeds_0x35d8.csv")
# STIMLOG_PATH = Path(r"E:\Lab\Data\stimulus_log\2026-04-27_TG884_19.02_1st_spatemp_0xf12d.csv")

ORIGINAL_RECORDING_SEGMENT_INDEX = 3

TEST_START_SEC_WITHIN_ORIGINAL_SEGMENT = 1864.0
TEST_END_SEC_WITHIN_ORIGINAL_SEGMENT = None

# =====================
# Stimlog runs for concatenated single-screen recordings
# =====================

# STIMLOG_RUNS = [
#     {
#         "stimlog_path": Path(
#             r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_F_correct_2026_05_27_screen0_8dir_2speeds_001.csv"
#         ),
#         "original_segment_index": 2,
#         "screen_role": "front",
#     },
#     {
#         "stimlog_path": Path(
#             r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_L_2026_05_27_screen1_8dir_2speeds_001.csv"
#         ),
#         "original_segment_index": 3,
#         "screen_role": "left",
#     },
#     {
#         "stimlog_path": Path(
#             r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_R_2026_05_27_screen3_8dir_2speeds_001.csv"
#         ),
#         "original_segment_index": 4,
#         "screen_role": "right",
#     },
# ]


PREPROCESSED_FOLDER = WORKING_DIR / "preprocessed_M12"
CURATED_SORTING_FOLDER = WORKING_DIR / f"sorting_M12_{SORTER_NAME}_curated"
PHY_FOLDER = OUTPUT_DIR / f"phy_M12_{SORTER_NAME}_curated"
ANALYZER_FOLDER = WORKING_DIR / f"analyzer_M12_{SORTER_NAME}_curated"

ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / f"analysis_{BIRD}_{SESSION}_{SORTER_NAME}"


# Recording segment selection
# This MUST match the segment used in SS2_Sorting.py and SS4_Export_phy.py.
RECORDING_SEGMENT_INDEX = SEGMENT_INDEX_TO_USE

# If None, 01_extract_events.py will infer segment boundaries from .ncs timestamps.
SEGMENT_START_TIMESTAMP_US = None
SEGMENT_END_TIMESTAMP_US = None


# Stimulus / alignment settings
# TTL is sent during Stimulus_state == "moving" in the Psychopy script.

# # TG884(OCb):
# MOTION_STATE = "moving"
# BASELINE_WINDOW = (-4.0, 0.0)
# EARLY_WINDOW = (0.0, 1.0)
# SUSTAINED_RESPONSE_WINDOW = (1.0, 6.0)
# MOVING_WINDOW = (0.0, 6.0)
# SAMPLING_FREQUENCY = 32000.0

# TG915(VbC):
MOTION_STATE = "moving"
BASELINE_WINDOW = (-2.0, 0.0)
EARLY_WINDOW = (0.0, 0.5)
SUSTAINED_RESPONSE_WINDOW = (0.5, 4.0)
MOVING_WINDOW = (0.0, 4.0)
SAMPLING_FREQUENCY = 32000.0


# Optional sanity check.
# TG884(OCb): 4 directions × 5 replicates × 1 speed = 20 moving epochs.
# EXPECTED_MOTION_TTL_COUNT = 20

# TG915(VbC): 12 patterns × 6 replicates × 1 speed = 72 moving epochs.
# EXPECTED_MOTION_TTL_COUNT = 72

# TG915(VbC): 8 directions × 6 replicates × 2 speed = 96 moving epochs.
# EXPECTED_MOTION_TTL_COUNT = 288

# TG964(VbC_speed): 4 directions × 8 or 6 replicates × 6 speed = 192 or 144 moving epochs.
EXPECTED_MOTION_TTL_COUNT = 144