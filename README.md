# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
Date: 02/05/2026

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("timeseries_2year.csv")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])

Y = df["stock_price"].values
n = len(Y)
MAX_LAG = 35

print("       AUTO CORRELATION FUNCTION (ACF)")
print(f"  Dataset     : timeseries_2year.csv")
print(f"  Variable    : stock_price")
print(f"  Observations: {n}")
print(f"  Max Lag     : {MAX_LAG}")
print()

mean_Y    = np.sum(Y) / n
variance  = np.sum((Y - mean_Y) ** 2) / n
std_Y     = np.sqrt(variance)
Y_norm    = Y - mean_Y

print(f"  Mean      : {mean_Y:.4f}")
print(f"  Variance  : {variance:.4f}")
print(f"  Std Dev   : {std_Y:.4f}")
print()

acf_values = []

C0 = np.sum(Y_norm ** 2) / n

for k in range(0, MAX_LAG + 1):
    Ck = np.sum(Y_norm[:n - k] * Y_norm[k:]) / n
    rk = Ck / C0
    acf_values.append(rk)

acf_values = np.array(acf_values)

conf_bound = 1.96 / np.sqrt(n)

lags = np.arange(0, MAX_LAG + 1)

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Correlogram — ACF for First 35 Lags\nDataset: timeseries_2year.csv — Stock Price",
             fontsize=13, fontweight='bold')

colors = ['crimson' if abs(rk) > conf_bound else 'steelblue' for rk in acf_values]
for k, rk in enumerate(acf_values):
    ax.vlines(k, 0, rk, color=colors[k], linewidth=2.2)
    ax.plot(k, rk, 'o', color=colors[k], markersize=5)

ax.axhline(0,            color='black', linewidth=0.8)
ax.axhline( conf_bound,  color='green', linewidth=1.5, linestyle='--',
            label=f'+95% CI = +{conf_bound:.4f}')
ax.axhline(-conf_bound,  color='green', linewidth=1.5, linestyle='--',
            label=f'−95% CI = −{conf_bound:.4f}')
ax.fill_between(lags, -conf_bound, conf_bound, alpha=0.08, color='green')
ax.set_xlabel("Lag (k)")
ax.set_ylabel("ACF  r(k)")
ax.set_xticks(lags)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
### OUTPUT:
<img width="1249" height="709" alt="image" src="https://github.com/user-attachments/assets/253a8f4f-4f66-42cb-8711-285a0895cb3a" />

### RESULT:
Thus we have successfully implemented the auto correlation function in python.
