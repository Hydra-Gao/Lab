import spikeinterface.sorters as ss

params = ss.get_default_sorter_params("kilosort4")
for k, v in params.items():
    print(k, "=", v)