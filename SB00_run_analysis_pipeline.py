from pathlib import Path
import subprocess
import sys

scripts = [
    "01_extract_events.py",
    "02_build_trial_table.py",
    "03_export_curated_spikes.py",
    "04_label_spikes.py",
    "05_compute_tuning_summary.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)