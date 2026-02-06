import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


data = np.load("./data/abiline_ten.npy")

node = data[1, 4, :]

# plot_pacf(node, lags=180)
# plot_acf(node, lags=180)
# plt.show()
node = node[-600:]
print(node.shape)

model = ARIMA(node[-600:], order=(8, 1, 8))
model_fit = model.fit(method="")
print(model_fit.summary())
