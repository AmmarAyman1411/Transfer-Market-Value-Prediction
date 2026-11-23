"""Predict market value for one or more players."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))

def load():
    import os
    path = os.getenv("MODEL_PATH")
    if path and os.path.exists(path):
        import joblib
        return joblib.load(path)
    from registry import get_model_path, load_model
    version = os.getenv("MODEL_VERSION")
    try:
        p, _ = get_model_path(version)
    except Exception:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lgbm_market_value.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Model file not found. Set MODEL_URL secret in Streamlit or place "
            "models/lgbm_market_value.joblib in the repo."
        )
    bundle, _ = load_model(version)
    return bundle


def add_derived(df):
    d = df.copy()
    if "age" in d.columns:
        d["age_sq"] = pd.to_numeric(d["age"], errors="coerce").fillna(0) ** 2
    if "contract_months" in d.columns:
        d["contract_short"] = (pd.to_numeric(d["contract_months"], errors="coerce").fillna(12) < 12).astype(int)
    for c in ["cum_yellows", "cum_reds", "recent_matches", "recent_goals", "recent_assists",
              "recent_minutes", "goals_per_90", "assists_per_90", "minutes_per_match",
              "sum_transfer_fee", "max_transfer_mv", "max_transfer_fee"]:
        if c not in d.columns:
            d[c] = 0.0
    if "num_transfers" in d.columns:
        d["has_transfer"] = (pd.to_numeric(d["num_transfers"], errors="coerce").fillna(0) > 0).astype(int)
    if "league" in d.columns:
        d["top_league"] = d["league"].isin(["GB1", "L1", "IT1", "ES1", "FR1"]).astype(int)
    if "age" in d.columns:
        a = pd.to_numeric(d["age"], errors="coerce").fillna(0)
        d["is_prime"] = a.between(23, 28).astype(int)
        d["is_young"] = (a < 23).astype(int)
    if "international_caps" in d.columns and "international_goals" in d.columns:
        caps = pd.to_numeric(d["international_caps"], errors="coerce").fillna(0)
        gls = pd.to_numeric(d["international_goals"], errors="coerce").fillna(0)
        d["has_international"] = (caps > 0).astype(int)
        d["goals_per_cap"] = [ (g / c) if c > 0 else 0.0 for g, c in zip(gls, caps) ]
    if "prev_value" in d.columns:
        pv = pd.to_numeric(d["prev_value"], errors="coerce").fillna(0)
        d["log_prev_value"] = np.log1p(pv)
    if "peak_so_far" in d.columns:
        ps = pd.to_numeric(d["peak_so_far"], errors="coerce").fillna(0)
        d["log_peak_so_far"] = np.log1p(ps)
    if "prev_value" in d.columns and "peak_so_far" in d.columns:
        pv = pd.to_numeric(d["prev_value"], errors="coerce").fillna(0)
        ps = pd.to_numeric(d["peak_so_far"], errors="coerce").fillna(0)
        d["prev_peak_ratio"] = pv / (ps + 1)
    return d


def predict_df(df):
    bundle = load()
    model = bundle["model"]
    features = bundle["features"]
    cat_cols = bundle["cat_cols"]
    X = add_derived(df.copy())
    for c in features:
        if c not in X.columns:
            X[c] = 0 if c not in cat_cols else "Unknown"
    X = X[features].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    log_pred = model.predict(X)
    return np.expm1(log_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--height", type=float, default=180.0)
    ap.add_argument("--caps", type=float, default=0.0)
    ap.add_argument("--goals", type=float, default=0.0)
    ap.add_argument("--position", default="Attack")
    ap.add_argument("--sub_position", default="Centre-Forward")
    ap.add_argument("--foot", default="right")
    ap.add_argument("--country", default="Unknown")
    ap.add_argument("--league", default="Unknown")
    ap.add_argument("--contract_months", type=float, default=12.0)
    ap.add_argument("--cum_matches", type=float, default=0.0)
    ap.add_argument("--cum_goals", type=float, default=0.0)
    ap.add_argument("--cum_assists", type=float, default=0.0)
    ap.add_argument("--cum_minutes", type=float, default=0.0)
    ap.add_argument("--num_transfers", type=float, default=0.0)
    ap.add_argument("--prev_value", type=float, default=0.0)
    ap.add_argument("--peak_so_far", type=float, default=0.0)
    ap.add_argument("--days_since_prev", type=float, default=9999.0)
    ap.add_argument("--player_highest_mv", type=float, default=0.0)
    ap.add_argument("--csv", help="Input CSV with feature columns; writes predictions.csv")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        preds = predict_df(df)
        out = df.copy()
        out["predicted_market_value_eur"] = preds
        out.to_csv("predictions.csv", index=False)
        print("Wrote predictions.csv")
        return

    row = pd.DataFrame([{
        "age": args.age,
        "height_in_cm": args.height,
        "international_caps": args.caps,
        "international_goals": args.goals,
        "position": args.position,
        "sub_position": args.sub_position,
        "foot": args.foot,
        "country_of_citizenship": args.country,
        "league": args.league,
        "contract_months": args.contract_months,
        "cum_matches": args.cum_matches,
        "cum_goals": args.cum_goals,
        "cum_assists": args.cum_assists,
        "cum_minutes": args.cum_minutes,
        "num_transfers": args.num_transfers,
        "prev_value": args.prev_value,
        "peak_so_far": args.peak_so_far,
        "days_since_prev": args.days_since_prev,
        "player_highest_mv": args.player_highest_mv,
    }])
    # derived features
    r = row
    r["age_sq"] = r["age"] ** 2
    r["contract_short"] = (r["contract_months"] < 12).astype(int)
    r["cum_yellows"] = 0.0
    r["cum_reds"] = 0.0
    r["recent_matches"] = 0.0
    r["recent_goals"] = 0.0
    r["recent_assists"] = 0.0
    r["recent_minutes"] = 0.0
    r["goals_per_90"] = 0.0
    r["assists_per_90"] = 0.0
    r["minutes_per_match"] = 0.0
    r["sum_transfer_fee"] = 0.0
    r["max_transfer_mv"] = 0.0
    r["max_transfer_fee"] = 0.0
    r["has_transfer"] = (r["num_transfers"] > 0).astype(int)
    r["top_league"] = r["league"].isin(["GB1", "L1", "IT1", "ES1", "FR1"]).astype(int)
    r["is_prime"] = r["age"].between(23, 28).astype(int)
    r["is_young"] = (r["age"] < 23).astype(int)
    r["has_international"] = (r["international_caps"] > 0).astype(int)
    r["goals_per_cap"] = r.apply(lambda x: (x["international_goals"] / x["international_caps"]) if x["international_caps"] > 0 else 0.0, axis=1)
    print(f"Predicted market value: EUR {predict_df(r)[0]:,.0f}")


if __name__ == "__main__":
    main()
