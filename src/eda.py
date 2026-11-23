"""Generate EDA plots for README/Streamlit."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

from data import build_training_dataset

df = build_training_dataset()

# 1. Market value distribution (log)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(np.log1p(df["market_value_in_eur"]), bins=60, kde=True, ax=ax, color="#1f77b4")
ax.set_title("Distribution of log(Market Value)")
ax.set_xlabel("log(1 + market_value_eur)")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "value_distribution.png"), dpi=120)
plt.close(fig)

# 2. Age vs market value (box by age bucket)
fig, ax = plt.subplots(figsize=(9, 5))
tmp = df.copy()
tmp["age_bucket"] = pd.cut(tmp["age"], bins=[15, 20, 23, 26, 29, 32, 45], labels=["15-20", "21-23", "24-26", "27-29", "30-32", "33+"])
sns.boxplot(data=tmp, x="age_bucket", y=np.log1p(tmp["market_value_in_eur"]), ax=ax, palette="Blues")
ax.set_title("Market Value by Age Bucket")
ax.set_xlabel("Age bucket")
ax.set_ylabel("log(1 + market_value_eur)")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "age_vs_value.png"), dpi=120)
plt.close(fig)

# 3. Position boxplot
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="position", y=np.log1p(df["market_value_in_eur"]), ax=ax, palette="Set2")
ax.set_title("Market Value by Position")
ax.set_xlabel("Position")
ax.set_ylabel("log(1 + market_value_eur)")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "position_boxplot.png"), dpi=120)
plt.close(fig)

# 4. Top leagues
fig, ax = plt.subplots(figsize=(9, 5))
top = df["league"].value_counts()
top = top[top.index != "Unknown"].head(12)
sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
ax.set_title("Top Leagues by Valuation Records")
ax.set_xlabel("Count")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "top_leagues.png"), dpi=120)
plt.close(fig)

# 5. Feature importance
bundle = joblib.load(os.path.join(ROOT, "models", "lgbm_market_value.joblib"))
model = bundle["model"]
imp = pd.Series(model.feature_importances_, index=bundle["features"]).sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
imp.plot(kind="barh", ax=ax, color="#ff7f0e")
ax.set_title("Top 15 Feature Importances (LightGBM)")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "feature_importance.png"), dpi=120)
plt.close(fig)

print("Saved plots to", ASSETS)
