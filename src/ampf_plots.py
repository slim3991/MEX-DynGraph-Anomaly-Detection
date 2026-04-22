import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp


with open("src/model_config.yaml") as f:
    m_conf = yaml.safe_load(f)
model_confs = m_conf["DDoS_configs"]

models = [
    MyGRTenDecomp(
        rank=20, tol=1e-4, laplacian_parameters=model_confs["GRRCP"]["laps_params"]
    ),
    MyGRTenDecomp(
        rank=20,
        tol=1e-4,
        local_threshold=0,
        laplacian_parameters=model_confs["GRRCP_no_robust"]["laps_params"],
    ),
    MyGRTuckerDecomp(
        rank=(20, 20, 20),
        local_threshold=None,
        tol=1e-4,
        laplacian_parameters=model_confs["GRRTucker"]["laps_params"],
    ),
    MyGRTuckerDecomp(
        rank=(20, 20, 20),
        local_threshold=0,
        tol=1e-4,
        laplacian_parameters=model_confs["GRRTucker_no_robust"]["laps_params"],
    ),
    MyTuckerTenDecomp(rank=(20, 20, 20), tol=1e-4),
    MyRHOOITenDecomp(rank=(20, 20, 20), tol=1e-4),
    MyCPTenDecomp(rank=20, tol=1e-4),
    MyTuckerTenDecomp(rank=(20, 20, 20), tol=1e4),
]

# Range of amplitude factors to test
amplitude_factors = [6, 7, 8, 9, 10, 11]

plt.figure(figsize=(8, 6))

for model in models:
    mean_aucs = []
    model.tol = 1e-3

    for ampf in amplitude_factors:
        aucs = []

        for i in range(2):
            print(f"{model.name} | ampf={ampf} | run={i}")

            T, L, _, _ = create_ddos_dataset_train(ampf)
            T_hat = model.fit_transform(T, L)
            resids = T - T_hat

            precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())

            # Sort recall before AUC
            recall, precision = zip(*sorted(zip(recall, precision)))

            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        # Average AUC for this amplitude
        mean_aucs.append(np.mean(aucs))
        print("aucs: ", aucs)

    # Plot AUC vs amplitude
    plt.plot(amplitude_factors, mean_aucs, marker="o", label=model.name)

# Final touches
plt.xlabel("Amplitude Factor")
plt.ylabel("Average PR AUC")
plt.title("Effect of Amplitude on PR AUC")
plt.legend()
plt.grid()
try:
    plt.savefig("./figures/ampfEffects.png")
except FileNotFoundError as e:
    print("path not found")


plt.show()
