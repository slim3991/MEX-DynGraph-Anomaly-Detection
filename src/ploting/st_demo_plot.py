import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Generate Data
np.random.seed(42)
x = np.linspace(-3, 3, 100)
# True cubic: y = x^3 - 2x + 1
y_true = x**3 - 2 * x + 1
noise = np.random.normal(0, 1.5, size=x.shape)
y_noisy = y_true + noise

# Add significant outliers
outlier_indices = np.random.choice(len(x), 8, replace=False)
y_noisy[outlier_indices] += np.random.choice([-15, 15], 8)


# 2. Fit Cubic Function
def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


popt, pcov = curve_fit(cubic_func, x, y_noisy)
y_fit = cubic_func(x, *popt)

# 3. Calculate Confidence Interval (Band)
# Standard deviation of the residuals
residuals = y_noisy - y_fit
sigma = np.std(residuals)
# Define the band (e.g., 2 standard deviations)
band_width = 1.96 * sigma
y_upper = y_fit + band_width
y_lower = y_fit - band_width

# 4. Identify Outliers
is_outlier = (y_noisy > y_upper) | (y_noisy < y_lower)

# 5. Prepare Second Dataset (Pulled outliers)
y_pulled = y_noisy.copy()
y_pulled[y_noisy > y_upper] = y_upper[y_noisy > y_upper]
y_pulled[y_noisy < y_lower] = y_lower[y_noisy < y_lower]

# 6. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Subplot 1: Original with Outliers highlighted
ax1.fill_between(x, y_lower, y_upper, color="gray", alpha=0.2, label="95% CI Band")
ax1.plot(x, y_fit, color="firebrick", lw=2, label="Cubic Fit")
ax1.scatter(x[~is_outlier], y_noisy[~is_outlier], s=20, alpha=0.7, label="Inliers")
ax1.scatter(
    x[is_outlier],
    y_noisy[is_outlier],
    color="orange",
    s=40,
    edgecolors="k",
    label="Outliers",
)
ax1.set_title("Original Data with Fit")
ax1.legend()

# Subplot 2: Outliers pulled to border
ax2.fill_between(x, y_lower, y_upper, color="gray", alpha=0.2, label="95% CI Band")
ax2.plot(x, y_fit, color="firebrick", lw=2, label="Cubic Fit")
ax2.scatter(x[~is_outlier], y_pulled[~is_outlier], s=20, alpha=0.7)
ax2.scatter(
    x[is_outlier],
    y_pulled[is_outlier],
    color="orange",
    s=40,
    edgecolors="k",
    label="Pulled Points",
)
ax2.set_title("Outliers Pulled to Band Border")

plt.tight_layout()
plt.savefig("cubic_fit_analysis.png")
plt.show()
