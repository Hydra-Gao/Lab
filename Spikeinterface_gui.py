import spikeinterface as si
from spikeinterface_gui import run_mainwindow

analyzer_path = r"C:\Lab\Processing\TG963_site1-2\bombcell_output_whitened\sorting_analyzer"

# reload the SortingAnalyzer
sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
# open and run the Qt app
run_mainwindow(sorting_analyzer, mode="desktop", curation=False)