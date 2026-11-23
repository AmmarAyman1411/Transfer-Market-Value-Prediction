import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
from data import build_training_dataset

df = build_training_dataset()
CATS = ["position", "sub_position", "foot", "country_of_citizenship", "league"]
NUMS = [c for c in df.columns if c not in ["player_id","date","market_value_in_eur","current_club_name","current_club_id","player_club_domestic_competition_id","date_of_birth","position","sub_position","foot","country_of_citizenship","league","contract_expiration_date","highest_market_value_in_eur","log_value"] and not c.startswith("lag_")]
FEATS = NUMS + CATS

train_mask = df["date"] < pd.Timestamp("2023-01-01")
val_mask = (df["date"] >= pd.Timestamp("2023-01-01")) & (df["date"] < pd.Timestamp("2024-01-01"))

Xtr = df.loc[train_mask, FEATS].copy(); ytr = df.loc[train_mask, "log_value"].copy()
Xva = df.loc[val_mask, FEATS].copy(); yva = df.loc[val_mask, "log_value"].copy()
for c in CATS:
    Xtr[c]=Xtr[c].astype("category"); Xva[c]=Xva[c].astype("category")

def objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": 42, "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva,yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict(Xva)
    return np.sqrt(mean_squared_error(yva, pred))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15, show_progress_bar=True)
print("Best RMSE(log):", study.best_value)
print("Best params:", study.best_params)
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_params.json"), "w") as f:
    json.dump(study.best_params, f, indent=2)
