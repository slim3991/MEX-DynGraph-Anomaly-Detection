import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CICIDS2017
df = pd.read_csv(
    "./raw_data/GeneratedLabelledFlows/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


nt = df[" Timestamp"].unique().__len__()
ns = df[" Source IP"].unique().__len__()
nr = df[" Destination IP"].unique().__len__()

df["Weight"] = df["Total Length of Fwd Packets"] + df[" Total Length of Bwd Packets"]
df["Is_Anomaly"] = (df[" Label"] != "BENIGN").astype(int)

df["s_idx"] = df[" Source IP"].astype("category").cat.codes
df["r_idx"] = df[" Destination IP"].astype("category").cat.codes
df["t_idx"] = df[" Timestamp"].astype("category").cat.codes

agg_df = (
    df.groupby(["s_idx", "r_idx", "t_idx"])
    .agg({"Weight": "sum", "Is_Anomaly": "max"})
    .reset_index()
)

T = np.zeros((ns, nr, nt))
L = np.zeros((ns, nr, nt), dtype=int)

T[agg_df["s_idx"], agg_df["r_idx"], agg_df["t_idx"]] = agg_df["Weight"]
L[agg_df["s_idx"], agg_df["r_idx"], agg_df["t_idx"]] = agg_df["Is_Anomaly"]
np.save("data/DDos_data.npy", T)
np.save("data/DDos_labels", L)
exit()
plt.ion()
fig, ax = plt.subplots()
T = np.log1p(T)
vmax = T.max()  # or a fixed value you choose
vmin = T.min()  # often 0
# vmax = 2000  # or a fixed value you choose
# vmin = abiline.min()  # often 0
# length = abiline.shape[2]
im = ax.imshow(T[:, :, 0], vmin=vmin, vmax=1, origin="lower")
fig.colorbar(im, ax=ax)

for i in range(200):
    plt.title(i)
    im.set_data(T[:, :, i])
    plt.pause(0.1)
