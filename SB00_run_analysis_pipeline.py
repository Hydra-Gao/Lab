from pathlib import Path
import subprocess
import sys

# scripts = [
#     "SB01_extract_events.py",
#     "SB02_build_trial_table.py",
#     "SB03_export_curated_spikes.py",
#     "SB04_label_spikes.py",
#     "SB05_compute_tuning_summary.py",
#     "SB06_compute_significance.py",
# ]

# scripts = [
#     "SB01_extract_events_dedup_12patterns.py",
#     "SB02b_build_trial_table_12patterns_3screen.py",
#     "SB03_export_curated_spikes.py",
#     "SB04b_label_spikes_12patterns_3screen.py",
#     "SB05b_compute_12patterns_summary.py",
#     "SB06b_compute_12patterns_significance.py",
# ]

scripts = [
    # "SB01_extract_events_dedup.py",
    "SB01_extract_events_for_cut_test.py",
    "SB02c_build_trial_table_8directions_1screen.py",
    "SB03_export_curated_spikes.py",
    "SB04c_label_spikes_8directions_1screen.py",
    "SB05c_compute_8directions_summary.py",
    "SB06c_compute_8directions_significance.py",
]

# scripts = [
#     "SB01a_extract_events_concatenated.py",
#     "SB02a1_build_trial_table_8directions_1screen.py",
#     "SB03_export_curated_spikes.py",
#     "SB04a1_label_spikes_8directions_1screen.py",
#     "SB05a1_compute_8directions_summary.py",
#     "SB06a1_compute_8directions_significance.py",
# ]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)