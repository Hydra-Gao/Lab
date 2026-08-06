import numpy as np
import pandas as pd

npy_path = r"C:\Lab\Processing\TG963_site1-3\similar_templates.npy"
csv_path = r"C:\Lab\Processing\TG963_site1-3\similar_templates.csv"

data = np.load(npy_path)

template_ids = np.arange(data.shape[0])

df = pd.DataFrame(
    data,
    index=template_ids,
    columns=template_ids
)

df.index.name = "template_id"

df.to_csv(csv_path, float_format="%.6f")

print("Saved to:", csv_path)
print("Shape:", data.shape)