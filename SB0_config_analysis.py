from pathlib import Path
from config_local import RAW_DATA, WORKING_DIR, OUTPUT_DIR

BIRD = "TG884"
SESSION = "2026-04-27_18-54-07"
SORTER_NAME = "kilosort4" 
# SORTER_NAME = "mountainsort5"

# path settings
RAW_NLX_FOLDER = RAW_DATA / BIRD / SESSION

EVENTS_NEV_PATH = RAW_NLX_FOLDER / "Events.nev"
STIMLOG_PATH = Path(
    r"F:\Work\UBC\Lab\Data\TG884\stimulus_log\2026-04-27_TG884_19.02_1st_spatemp_0xf12d.csv"
)

PREPROCESSED_FOLDER = WORKING_DIR / "preprocessed_M12"
CURATED_SORTING_FOLDER = WORKING_DIR / f"sorting_M12_{SORTER_NAME}_curated"
PHY_FOLDER = OUTPUT_DIR / f"phy_M12_{SORTER_NAME}_curated"

ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / f"analysis_{BIRD}_{SESSION}_{SORTER_NAME}"


# Recording segment selection
# This MUST match the segment used in SS2_Sorting.py and SS4_Export_phy.py.
RECORDING_SEGMENT_INDEX = 1

# If None, 01_extract_events.py will infer segment boundaries from .ncs timestamps.
SEGMENT_START_TIMESTAMP_US = None
SEGMENT_END_TIMESTAMP_US = None


# Stimulus / alignment settings
# TTL is sent during Stimulus_state == "moving" in the Psychopy script.
MOTION_STATE = "moving"
BASELINE_WINDOW = (-4.0, 0.0)
EARLY_WINDOW = (0.0, 1.0)
SUSTAINED_RESPONSE_WINDOW = (1.0, 6.0)
MOVING_WINDOW = (0.0, 6.0)
SAMPLING_FREQUENCY = 32000.0

# Optional sanity check.
# Current design: 4 directions × 5 replicates × 1 speed = 20 moving epochs.
EXPECTED_MOTION_TTL_COUNT = 20