from pathlib import Path
import subprocess
import sys

# scripts = [
#     "SB11_plot_units.py",
#     "SB12_plot_population.py",
# ]

scripts = [
    "SB11b_plot_units_12patterns.py",
    #"SB12_plot_population.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    subprocess.run([sys.executable, script], check=True)