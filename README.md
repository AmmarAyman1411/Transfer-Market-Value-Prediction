# Football Player Transfer Market Value Prediction

Predicts a football player's transfer market value (EUR) from profile + history features.

## Data
Transfermarkt data via HuggingFace mirror `ngeorgea/transfermarkt-player-scores`
(players, historical valuations, appearances, transfers, clubs).

## Features
Age, height, position/sub-position/foot, nationality, league, international caps/goals,
contract months remaining, cumulative appearances (matches/goals/assists/minutes/cards),
recent 12-month form, transfer history (count, fees, max value), value history
(prev/peak/days since), age curve, league tier, derived ratios.

Train/test split is time-based (train <2023, val 2023, test 2024+).

## Metrics (test, 2024+ valuations, tuned LightGBM)
R2=0.975, RMSE=EUR1.38M, MAE=EUR420k, MedAE=EUR84k, MAPE=20.0%.
Within 10%: 38.7%, within 20%: 67.6%, within 50%: 92.7%.
Cold-start (no history): ~R2=0.61.

## Setup
```
pip install -r requirements.txt
```

## Train
```
python src/train.py
```
Saves `models/lgbm_market_value.joblib`. Uses Optuna-tuned params.

Optional hyperparameter search:
```
python src/tune.py
```

## Predict (CLI)
```
python src/predict.py --age 24 --position Attack --sub_position "Centre-Forward" --foot right --country Germany --league GB1 --caps 30 --goals 15
```
Or batch:
```
python src/predict.py --csv input.csv
```

## Streamlit app
```
python -m streamlit run streamlit_app.py
```
Set secrets `MODEL_URL` (hosted model) and optional `APP_PASSWORD`.

## FastAPI
```
uvicorn api:app --reload
```
POST to http://localhost:8000/predict with JSON body.
Env: `API_KEY`, `RATE_LIMIT`, `CORS_ORIGINS`, `MODEL_PATH`, `MODEL_VERSION`, `PORT`.

## Model registry
`models/registry.json` + `src/registry.py`. Endpoints: `/models`, `/models/{version}`.

## Docker
```
docker build -t mv-predictor .
docker run -p 8000:8000 mv-predictor
```

## Refresh data + retrain
```
python refresh.py
```

## Gradio demo
```
python app.py
```

## Tests
```
pytest
```