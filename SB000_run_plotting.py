from pathlib import Path
import subprocess
import sys

scripts = [
    "SB06_plot_units.py",
    "SB07_plot_population.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)