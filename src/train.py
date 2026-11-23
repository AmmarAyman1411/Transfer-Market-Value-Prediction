"""Train a tuned LightGBM regressor to predict log(market value)."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

from data import build_training_dataset

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "lgbm_market_value.joblib")

CAT_COLS = ["position", "sub_position", "foot", "country_of_citizenship", "league"]
NUM_COLS = ["age", "age_sq", "height_in_cm", "international_caps", "international_goals",
            "contract_months", "contract_short",
            "cum_matches", "cum_goals", "cum_assists", "cum_minutes",
            "cum_yellows", "cum_reds",
            "recent_matches", "recent_goals", "recent_assists", "recent_minutes",
            "goals_per_90", "assists_per_90", "minutes_per_match",
            "num_transfers", "sum_transfer_fee", "max_transfer_mv", "max_transfer_fee",
            "has_transfer", "top_league", "is_prime", "is_young",
            "has_international", "goals_per_cap",
            "prev_value", "peak_so_far", "days_since_prev", "player_highest_mv",
            "log_prev_value", "log_peak_so_far", "prev_peak_ratio"]
FEATURES = NUM_COLS + CAT_COLS


def main():
    df = build_training_dataset()

    train_mask = df["date"] < pd.Timestamp("2023-01-01")
    val_mask = (df["date"] >= pd.Timestamp("2023-01-01")) & (df["date"] < pd.Timestamp("2024-01-01"))
    test_mask = df["date"] >= pd.Timestamp("2024-01-01")

    X_train = df.loc[train_mask, FEATURES].copy()
    y_train = df.loc[train_mask, "log_value"].copy()
    X_val = df.loc[val_mask, FEATURES].copy()
    y_val = df.loc[val_mask, "log_value"].copy()
    X_test = df.loc[test_mask, FEATURES].copy()
    y_test = df.loc[test_mask, "log_value"].copy()

    for c in CAT_COLS:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print("train/val/test", len(X_train), len(X_val), len(X_test))

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.00941910455780081,
        num_leaves=464,
        min_child_samples=12,
        subsample=0.7677353186469493,
        colsample_bytree=0.6032117013417799,
        reg_alpha=0.003142801325387677,
        reg_lambda=0.001013552854225303,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    pred_log = model.predict(X_test)
    pred_val = np.expm1(pred_log)
    true_val = np.expm1(y_test)
    rmse = np.sqrt(mean_squared_error(true_val, pred_val))
    mae = mean_absolute_error(true_val, pred_val)
    r2 = r2_score(true_val, pred_val)
    medae = np.median(np.abs(true_val - pred_val))
    mape = np.mean(np.abs((true_val - pred_val) / true_val)) * 100
    print(f"R2={r2:.4f} RMSE={rmse:,.0f} MAE={mae:,.0f} MedAE={medae:,.0f} MAPE={mape:.2f}")
    for thr in [0.1, 0.2, 0.3, 0.5]:
        print(f"within {int(thr*100)}%: {np.mean(np.abs((true_val-pred_val)/true_val)<=thr)*100:.2f}%")

    joblib.dump({"model": model, "features": FEATURES, "cat_cols": CAT_COLS,
                 "num_cols": NUM_COLS}, MODEL_PATH)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
