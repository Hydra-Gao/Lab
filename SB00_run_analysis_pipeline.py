from pathlib import Path
import subprocess
import sys

scripts = [
    "SB01_extract_events.py",
    "SB02_build_trial_table.py",
    "SB03_export_curated_spikes.py",
    "SB04_label_spikes.py",
    "SB05_compute_tuning_summary.py",
    "SB05b_compute_significance.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)