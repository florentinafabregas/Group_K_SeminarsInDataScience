import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

plt.style.use('fivethirtyeight')

# ==========================
# Load the data
# ==========================
df = pd.read_csv("regression_file.csv")

# Log-transform the variables
df["log_pop"] = np.log(df["population_2019"])
df["log_users"] = np.log(df["user_count"])

# ==========================
# Fit regression on ALL states
# ==========================
slope_all, intercept_all, r_value_all, p_all, std_err_all = linregress(
    df["log_pop"], df["log_users"])
df["pred_log_users"] = slope_all * df["log_pop"] + intercept_all
df["residuals"] = df["log_users"] - df["pred_log_users"]

# Compute mean and std of residuals
res_mean = df["residuals"].mean()
res_std = df["residuals"].std()

# Add this line to print the value
print(f"Residual Mean: {res_mean:.4f}")
print(f"Residual Std Dev: {res_std:.4f}")

# ==========================
# Identify outliers
# ==========================

# ==========================
# Identify outliers
# ==========================
# States with |residual - mean| > 1 std
outlier_mask = (np.abs(df["residuals"] - res_mean) > res_std)
df["outlier"] = outlier_mask

# Low user states (<1000)
low_user_mask = df["user_count"] < 1000

# Combine filters (residual ±1 std and low-user)
filtered_df = df.loc[~outlier_mask & ~low_user_mask].copy()


# Get outlier data as list of (state, residual) tuples
outlier_data = list(zip(df.loc[outlier_mask, "state_code"],
                        df.loc[outlier_mask, "residuals"]))

# Print the formatted list (rounding residual to 2 decimal places)
print("Removed due to residual > 1 std (State, Residual):",
      [(state, round(res, 2)) for state, res in outlier_data])


# print("Removed due to residual > 1 std:",
#     df.loc[outlier_mask, "state_code"].tolist())

print("Removed due to low user count:",
      df.loc[low_user_mask, "state_code"].tolist())
print("Final kept states:", len(filtered_df))

# ==========================
# Fit regression on filtered states
# ==========================
slope_f, intercept_f, r_value_f, p_f, std_err_f = linregress(
    filtered_df["log_pop"], filtered_df["log_users"])

# ==========================
# Plot
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---------- Panel 1: All states ----------
sns.regplot(
    x=df["log_pop"],
    y=df["log_users"],
    ax=axes[0],
    scatter=True,
    ci=95,
    line_kws={'color': 'blue', 'linewidth': 1.5}
)
for _, row in df.iterrows():
    axes[0].text(row["log_pop"], row["log_users"],
                 row["state_code"], fontsize=8)

axes[0].set_title("All states")
axes[0].set_xlabel("log(population)")
axes[0].set_ylabel("log(reddit users)")
axes[0].text(
    13.4, 11,
    f"#states = {len(df)}\n$R^2$ = {r_value_all**2:.2f}\nβ = {slope_all:.2f}",
    fontsize=10
)

# ---------- Panel 2: Filtered states ----------
sns.regplot(
    x=filtered_df["log_pop"],
    y=filtered_df["log_users"],
    ax=axes[1],
    scatter=True,
    ci=95,
    line_kws={'color': 'blue', 'linewidth': 1.5}
)
for _, row in filtered_df.iterrows():
    axes[1].text(row["log_pop"], row["log_users"],
                 row["state_code"], fontsize=8)

axes[1].set_title("Filtered states")
axes[1].set_xlabel("log(population)")
axes[1].set_ylabel("")
axes[1].text(
    13.4, 11,
    f"#states = {len(filtered_df)}\n$R^2$ = {r_value_f**2:.2f}\nβ = {slope_f:.2f}",
    fontsize=10
)

plt.tight_layout()
plt.show()
