import pandas as pd

df = pd.read_csv("raw_data/SKAB/valve1/0.csv", delimiter=";")


print(df.columns)

df["max_acc"] = df[["Accelerometer1RMS", "Accelerometer2RMS"]].max(axis=1)
df["power"] = df[["Current", "Voltage"]].prod(axis=1)


klkj
