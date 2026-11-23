"""Build a training dataset for market-value prediction."""
import os

import numpy as np
import pandas as pd
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PLAYERS_PARQUET = os.path.join(DATA_DIR, "players.parquet")
VALS_PARQUET = os.path.join(DATA_DIR, "player_valuations.parquet")
APPS_PARQUET = os.path.join(DATA_DIR, "appearances.parquet")
TRANSFERS_PARQUET = os.path.join(DATA_DIR, "transfers.parquet")
CLUBS_PARQUET = os.path.join(DATA_DIR, "clubs.parquet")
DATASET_PARQUET = os.path.join(DATA_DIR, "training.parquet")


def _load(name, file):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    ds = load_dataset("ngeorgea/transfermarkt-player-scores", data_files={"train": file}, split="train")
    df = ds.to_pandas()
    df.to_parquet(path, index=False)
    return df


def _load_players():
    return _load("players.parquet", "players.csv")


def _load_valuations():
    return _load("player_valuations.parquet", "player_valuations.csv")


def _load_appearances():
    return _load("appearances.parquet", "appearances.csv")


def _load_transfers():
    return _load("transfers.parquet", "transfers.csv")


def _load_clubs():
    return _load("clubs.parquet", "clubs.csv")


def build_training_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_PARQUET):
        return pd.read_parquet(DATASET_PARQUET)

    players = _load_players()
    vals = _load_valuations()

    pcols = ["player_id", "date_of_birth", "position", "sub_position", "foot",
             "height_in_cm", "country_of_citizenship", "international_caps",
             "international_goals", "contract_expiration_date",
             "highest_market_value_in_eur"]
    players = players[pcols].copy()

    df = vals.merge(players, on="player_id", how="left", validate="many_to_one")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = (df["date"] - df["date_of_birth"]).dt.days / 365.25

    df["contract_expiration_date"] = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
    df["contract_months"] = (df["contract_expiration_date"] - df["date"]).dt.days / 30.44

    # Valuation history features
    v = vals.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.sort_values(["player_id", "date"])
    grp = v.groupby("player_id", group_keys=False)
    v["prev_value"] = grp["market_value_in_eur"].shift(1)
    v["prev_date"] = grp["date"].shift(1)
    v["peak_so_far"] = grp["market_value_in_eur"].transform(lambda s: s.expanding().max().shift(1))
    v["days_since_prev"] = (v["date"] - v["prev_date"]).dt.days
    hist = v[["player_id", "date", "prev_value", "peak_so_far", "days_since_prev"]]
    df = df.merge(hist, on=["player_id", "date"], how="left", validate="many_to_one")

    # Appearances cumulative
    apps = _load_appearances().copy()
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps.sort_values(["player_id", "date"])
    for c in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
        apps[c] = pd.to_numeric(apps[c], errors="coerce").fillna(0)
    g = apps.groupby("player_id", group_keys=False)
    apps["cum_matches"] = g.cumcount() + 1
    apps["cum_goals"] = g["goals"].cumsum()
    apps["cum_assists"] = g["assists"].cumsum()
    apps["cum_minutes"] = g["minutes_played"].cumsum()
    apps["cum_yellows"] = g["yellow_cards"].cumsum()
    apps["cum_reds"] = g["red_cards"].cumsum()
    apps_cur = apps[["player_id", "date", "cum_matches", "cum_goals", "cum_assists",
                     "cum_minutes", "cum_yellows", "cum_reds"]].copy()

    # lagged cumulative (12 months prior)
    apps_lag = apps_cur.copy()
    apps_lag = apps_lag.rename(columns={
        "date": "lag_app_date",
        "cum_matches": "lag_matches",
        "cum_goals": "lag_goals",
        "cum_assists": "lag_assists",
        "cum_minutes": "lag_minutes",
        "cum_yellows": "lag_yellows",
        "cum_reds": "lag_reds",
    })

    df = df.sort_values("date")
    df = pd.merge_asof(
        df, apps_cur.sort_values("date"),
        by="player_id", left_on="date", right_on="date",
        direction="backward",
    )
    df["lag_date"] = df["date"] - pd.Timedelta(days=365)
    df = df.sort_values("lag_date")
    df = pd.merge_asof(
        df, apps_lag.sort_values("lag_app_date"),
        by="player_id", left_on="lag_date", right_on="lag_app_date",
        direction="backward",
    )

    for c in ["cum_matches", "cum_goals", "cum_assists", "cum_minutes", "cum_yellows", "cum_reds",
              "lag_matches", "lag_goals", "lag_assists", "lag_minutes", "lag_yellows", "lag_reds"]:
        df[c] = df[c].fillna(0)
    # recent 12-month form
    df["recent_matches"] = (df["cum_matches"] - df["lag_matches"]).clip(lower=0)
    df["recent_goals"] = (df["cum_goals"] - df["lag_goals"]).clip(lower=0)
    df["recent_assists"] = (df["cum_assists"] - df["lag_assists"]).clip(lower=0)
    df["recent_minutes"] = (df["cum_minutes"] - df["lag_minutes"]).clip(lower=0)

    # ratios
    df["goals_per_90"] = np.where(df["cum_minutes"] > 0, df["cum_goals"] * 90 / df["cum_minutes"], 0)
    df["assists_per_90"] = np.where(df["cum_minutes"] > 0, df["cum_assists"] * 90 / df["cum_minutes"], 0)
    df["minutes_per_match"] = np.where(df["cum_matches"] > 0, df["cum_minutes"] / df["cum_matches"], 0)

    # Transfer aggregates
    tr = _load_transfers().copy()
    tr_agg = tr.groupby("player_id").agg(
        num_transfers=("transfer_date", "count"),
        sum_transfer_fee=("transfer_fee", "sum"),
        max_transfer_mv=("market_value_in_eur", "max"),
        max_transfer_fee=("transfer_fee", "max"),
    ).reset_index()
    df = df.merge(tr_agg, on="player_id", how="left", validate="many_to_one")
    for c in ["num_transfers", "sum_transfer_fee", "max_transfer_mv", "max_transfer_fee"]:
        df[c] = df[c].fillna(0)

    # Clean
    df = df.dropna(subset=["market_value_in_eur", "age"])
    df = df[df["market_value_in_eur"] > 0]
    df = df[(df["age"] >= 15) & (df["age"] <= 45)]

    df["log_value"] = np.log1p(df["market_value_in_eur"])

    for c in ["position", "sub_position", "foot", "country_of_citizenship",
              "player_club_domestic_competition_id"]:
        df[c] = df[c].fillna("Unknown").astype(str)
    for c in ["height_in_cm", "international_caps", "international_goals",
              "contract_months"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    df["league"] = df["player_club_domestic_competition_id"].fillna("Unknown").astype(str)

    # age features
    df["age_sq"] = df["age"] ** 2
    df["is_prime"] = ((df["age"] >= 23) & (df["age"] <= 28)).astype(int)
    df["is_young"] = (df["age"] < 23).astype(int)

    # top league
    top_leagues = {"GB1", "L1", "IT1", "ES1", "FR1"}
    df["top_league"] = df["league"].isin(top_leagues).astype(int)

    # national team
    df["has_international"] = (df["international_caps"].fillna(0) > 0).astype(int)
    df["goals_per_cap"] = np.where(df["international_caps"].fillna(0) > 0,
                                   df["international_goals"].fillna(0) / df["international_caps"].fillna(0), 0)

    # contract short
    df["contract_short"] = (df["contract_months"] < 12).astype(int)

    # has transfer
    df["has_transfer"] = (df["num_transfers"] > 0).astype(int)

    # prev_value / peak fill
    df["prev_value"] = pd.to_numeric(df["prev_value"], errors="coerce").fillna(0)
    df["peak_so_far"] = pd.to_numeric(df["peak_so_far"], errors="coerce").fillna(0)
    df["days_since_prev"] = pd.to_numeric(df["days_since_prev"], errors="coerce").fillna(9999)
    df["player_highest_mv"] = pd.to_numeric(df["highest_market_value_in_eur"], errors="coerce").fillna(0)
    df["log_prev_value"] = np.log1p(df["prev_value"])
    df["log_peak_so_far"] = np.log1p(df["peak_so_far"])
    df["prev_peak_ratio"] = df["prev_value"] / (df["peak_so_far"] + 1)

    df["contract_months"] = df["contract_months"].clip(-12, 120)

    # drop helper
    for c in ["lag_date", "lag_app_date"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df.to_parquet(DATASET_PARQUET, index=False)
    return df


if __name__ == "__main__":
    d = build_training_dataset()
    print(d.shape)
    print(d.columns.tolist())
