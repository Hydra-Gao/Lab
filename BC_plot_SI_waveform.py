import numpy as np
import spikeinterface as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt

analyzer = si.load_sorting_analyzer(
    r"C:\Lab\Processing\TG963_site1-2\bombcell_output\sorting_analyzer"
)

unit_ids = analyzer.unit_ids[[20, 21, 65]]

for unit_id in unit_ids:
    templates = analyzer.get_extension("templates").get_data()
    unit_index = analyzer.sorting.id_to_index(unit_id)
    template = templates[unit_index]

    # best channel
    ptp = np.ptp(template, axis=0)
    best_channel_index = np.argmax(ptp)

    # spatially nearest channels
    channel_locations = analyzer.get_channel_locations()
    best_location = channel_locations[best_channel_index]

    distances = np.linalg.norm(
        channel_locations - best_location,
        axis=1
    )

    closest_indices = np.argsort(distances)[:10]
    channels_to_plot = analyzer.channel_ids[closest_indices]

    print(
        f"Unit {unit_id}, "
        f"best channel = {analyzer.channel_ids[best_channel_index]}"
    )

    sw.plot_unit_templates(
        analyzer,
        unit_ids=[unit_id],
        channel_ids=channels_to_plot,
        abs_y_scale=0.05,
        scalebar=True,
    )

    sw.plot_unit_waveforms(
        analyzer,
        unit_ids=[unit_id],
        channel_ids=channels_to_plot,
        abs_y_scale=0.05,
        scalebar=True,
    )

    
plt.show()