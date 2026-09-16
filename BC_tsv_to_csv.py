import pandas as pd

df = pd.read_csv(r"C:\Lab\Processing\TG963_site1-3\cluster_group.tsv", sep="\t")
df.to_csv(r"C:\Lab\Processing\TG963_site1-3\cluster_group.csv", index=False)