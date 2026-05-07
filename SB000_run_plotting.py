from pathlib import Path
import subprocess
import sys

scripts = [
    "06_plot_units.py",
    "07_plot_population.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)