import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from predict import predict_df


def test_predict_returns_positive():
    row = pd.DataFrame([{
        "age": 24, "height_in_cm": 180, "international_caps": 10,
        "international_goals": 3, "position": "Attack",
        "sub_position": "Centre-Forward", "foot": "right",
        "country_of_citizenship": "Germany", "league": "GB1",
    }])
    val = predict_df(row)[0]
    assert val > 0
    assert val < 1e9
