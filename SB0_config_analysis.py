from pathlib import Path
from config_local import RAW_DATA, WORKING_DIR, OUTPUT_DIR, SEGMENT_INDEX_TO_USE

BIRD = "TG915"
SESSION = "2026-05-27_19-14-21"
SORTER_NAME = "kilosort4" 
# SORTER_NAME = "mountainsort5"

# path settings
RAW_NLX_FOLDER = RAW_DATA / BIRD / SESSION

EVENTS_NEV_PATH = RAW_NLX_FOLDER / "Events.nev"

# STIMLOG_PATH = Path(r"F:\Work\UBC\Lab\Data\TG884\stimulus_log\2026-04-27_TG884_19.02_1st_spatemp_0x20c9.csv")
# STIMLOG_PATH = Path(r"E:\Lab\_analyzed\Section_1-1\2026-04-27_TG884_19.02_1st_spatemp_0xf12d.csv")
# STIMLOG_PATH = Path(r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_op_s__2026_05_27_VbC_12patterns_3screens_001.csv")
STIMLOG_PATH = Path(r"G:\Lab\Raw_data\TG915\stimulus_logs\2026-05-27_TG915_VbC_Site2_R_2026_05_27_screen3_8dir_2speeds_001.csv")

# STIMLOG_PATH = Path(r"E:\Lab\Data\stimulus_log\2026-04-27_TG884_19.02_1st_spatemp_0xf12d.csv")

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
BASELINE_WINDOW = (-3.0, 0.0)
EARLY_WINDOW = (0.0, 1.0)
SUSTAINED_RESPONSE_WINDOW = (1.0, 5.0)
MOVING_WINDOW = (0.0, 5.0)
SAMPLING_FREQUENCY = 32000.0


# Optional sanity check.
# TG884(OCb): 4 directions × 5 replicates × 1 speed = 20 moving epochs.
# EXPECTED_MOTION_TTL_COUNT = 20

# # TG915(VbC): 12 patterns × 6 replicates × 1 speed = 72 moving epochs.
# EXPECTED_MOTION_TTL_COUNT = 72

# TG915(VbC): 8 directions × 6 replicates × 2 speed = 96 moving epochs.
EXPECTED_MOTION_TTL_COUNT = 96