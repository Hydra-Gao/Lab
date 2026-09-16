from pathlib import Path
import numpy as np
import torch


KS_FOLDER = Path(
    r"C:\Lab\Processing\TG963_site1-2"
)

ops = np.load(
    KS_FOLDER / "ops.npy",
    allow_pickle=True
).item()

iU = ops["iU"]
iCC = ops["iCC"]
xc = np.asarray(ops["xc"])
yc = np.asarray(ops["yc"])

if torch.is_tensor(iU):
    iU = iU.cpu().numpy()

if torch.is_tensor(iCC):
    iCC = iCC.cpu().numpy()

iU = np.asarray(iU).astype(int)
iCC = np.asarray(iCC).astype(int)


# ============================================================
# Find learned templates relevant to channels 88/90/92
# ============================================================

rows = []

for tid in range(len(iU)):

    main_ch = int(iU[tid])
    footprint = np.asarray(
        iCC[:, main_ch],
        dtype=int,
    )

    has88 = 88 in footprint
    has90 = 90 in footprint
    has92 = 92 in footprint

    # Only show templates spatially relevant to this region
    if has88 or has90 or has92:

        rows.append(
            (
                tid,
                main_ch,
                xc[main_ch],
                yc[main_ch],
                has88,
                has90,
                has92,
                ",".join(map(str, footprint)),
            )
        )


print(
    "template  main_ch    x      y    "
    "has88  has90  has92   footprint"
)

for row in rows:
    print(
        f"{row[0]:8d} "
        f"{row[1]:8d} "
        f"{row[2]:6.1f} "
        f"{row[3]:6.1f} "
        f"{str(row[4]):>6} "
        f"{str(row[5]):>6} "
        f"{str(row[6]):>6}   "
        f"{row[7]}"
    )