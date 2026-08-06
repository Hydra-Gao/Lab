from pathlib import Path
import numpy as np
import spikeinterface.full as si
from kilosort.io import save_probe

data_folder = Path(r"D:\Lab\Raw_data\TG963_Cb\Cb_2026_07_14_1site_3_g0\Cb_2026_07_14_1site_3_g0_imec0")

recording = si.read_spikeglx(
    data_folder,
    stream_name="imec0.ap"
)

# SpikeInterface自动从SpikeGLX metadata恢复probe
si_probe = recording.get_probe()

positions = recording.get_channel_locations()

# 对普通单shank Neuropixels，通常全部为0
if si_probe.shank_ids is None:
    kcoords = np.zeros(recording.get_num_channels(), dtype=np.int32)
else:
    # 将shank标签转换成0, 1, 2...整数
    _, kcoords = np.unique(si_probe.shank_ids, return_inverse=True)
    kcoords = kcoords.astype(np.int32)

ks_probe = {
    "chanMap": np.arange(
        recording.get_num_channels(),
        dtype=np.int32
    ),
    "xc": positions[:, 0].astype(np.float32),
    "yc": positions[:, 1].astype(np.float32),
    "kcoords": kcoords,
    "n_chan": recording.get_num_channels(),
}

save_probe(
    ks_probe,
    r"D:\Lab\Raw_data\TG963_Cb\Cb_2026_07_14_1site_3_g0\Cb_2026_07_14_1site_3_g0_imec0\probe_from_meta.json"
)

print("Saved Kilosort probe file.")