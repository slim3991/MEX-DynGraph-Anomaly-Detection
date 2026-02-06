import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import savemat, loadmat


def parse_xml(path):
    tree = ET.parse(path)
    return tree.getroot()


files = sorted(glob.glob("./raw_data/abilene/*.xml"))
root1 = parse_xml(files[0])

# Use wildcard for both levels
nodes = root1.findall("{*}networkStructure/{*}nodes/{*}node")
Nnodes = len(nodes)

node_dict = {}
for i, node in enumerate(nodes):
    # Changed from "idAttribute" to "id" to match your XML snippet
    node_id = node.attrib["id"]
    node_dict[node_id] = i


T = np.zeros((Nnodes, Nnodes, len(files)), dtype=np.float32)

for k, file_path in enumerate(files):
    root = parse_xml(file_path)

    # Use wildcard for demands and children
    # We use .// to find them anywhere, or explicit path with {*}
    demands = root.findall(".//{*}demand")

    for demand in demands:
        # Note: If source/target also have namespaces, use {*} here too
        src = demand.find("{*}source").text
        tgt = demand.find("{*}target").text
        val = float(demand.find("{*}demandValue").text)

        if src in node_dict and tgt in node_dict:
            i = node_dict[src]
            j = node_dict[tgt]
            T[i, j, k] = val

    if (k + 1) % 50 == 0:
        print(f"Processed {k + 1} files...")

# --------------------------------------------------
# Save to .mat file (MATLAB-compatible)
# --------------------------------------------------
os.makedirs("data", exist_ok=True)

# --------------------------------------------------
# Clear + reload (like MATLAB)
# --------------------------------------------------
np.save("data/abiline_ten.npy", T)
del T

T = np.load("data/abiline_ten.npy")

# Equivalent of `whos T`
print("T shape:", T.shape)
print("T dtype:", T.dtype)
print("T size (MB):", T.nbytes / 1e6)
