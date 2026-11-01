import numpy as np
import spikeinterface as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

analyzer = si.load_sorting_analyzer(
    r"C:\Lab\Processing\TG963_site1-3\bombcell_output_whitened\sorting_analyzer"
)

for unit_id in analyzer.unit_ids[:20]:
    channel_ids = analyzer.sparsity.unit_id_to_channel_ids[unit_id]
    print(unit_id, len(channel_ids), channel_ids)