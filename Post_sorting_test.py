import spikeinterface.full as si

recording_saved = si.load("preprocessed_M12")
recording_seg0 = recording_saved.select_segments([0])
recording_test = recording_seg0.frame_slice(
    start_frame=0,
    end_frame=recording_seg0.get_num_frames(segment_index=0)
)

from spikeinterface.curation import CurationSorting, remove_duplicated_spikes

sorting_test = si.read_sorter_folder("kilosort4_M12_test_output")

print("Original sorting:")
print(sorting_test)
print("Original spike counts:")
print(sorting_test.count_num_spikes_per_unit())

""" cs = CurationSorting(parent_sorting=sorting_test)

cs.merge(unit_ids=[3, 7], new_unit_id=307)

sorting_test = cs.sorting """

sorting_clean = remove_duplicated_spikes(
    sorting_test,
    censored_period_ms=0.25,
    method="keep_first"
)

print("After removing duplicated spikes:")
print(sorting_clean)
print("Cleaned spike counts:")
print(sorting_clean.count_num_spikes_per_unit())

analyzer = si.create_sorting_analyzer(
    sorting=sorting_clean,
    recording=recording_test,
    format="binary_folder",
    folder="analyzer_M12_test_dedup",
    overwrite=True
)

analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
analyzer.compute("noise_levels")
analyzer.compute("quality_metrics", metric_names=[
    "snr",
    "isi_violation",
    "firing_rate",
    "presence_ratio"
])

metrics = analyzer.get_extension("quality_metrics").get_data()
print(metrics)

unit_ids = sorting_test.get_unit_ids()

si.plot_unit_waveforms(analyzer, unit_ids=unit_ids[:5])
si.plot_unit_templates(analyzer, unit_ids=unit_ids[:5])
si.plot_autocorrelograms(sorting_test, unit_ids=unit_ids[:10])
si.plot_rasters(sorting_test)

si.export_to_phy(
    analyzer,
    output_folder="phy_M12_test_dedup",
    remove_if_exists=True
)