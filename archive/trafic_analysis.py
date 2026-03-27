import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


data = np.load("./data/abiline_ten.npy")

node = data[1, 4, :]

data = node[-4000:]
data = np.sqrt(data)
# data = np.diff(data, n=1)
# plt.plot(data)
# plt.show()
# exit()

# plot_pacf(data, lags=12 * 24)
# plot_acf(data, lags=12 * 24)
# plt.show()
# exit()
# print(node.shape)

model = ARIMA(data, order=(10, 1, 10))
model_fit = model.fit()
print(model_fit.summary())
