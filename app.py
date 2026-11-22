"""Gradio demo for market-value prediction."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import pandas as pd
from predict import predict_df

POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]
FEET = ["right", "left", "both"]


def predict(age, height, caps, goals, position, sub_position, foot, country, league,
            contract_months, cum_matches, cum_goals, cum_assists, cum_minutes,
            num_transfers, prev_value, peak_so_far, days_since_prev, player_highest_mv):
    row = pd.DataFrame([{
        "age": age, "height_in_cm": height, "international_caps": caps,
        "international_goals": goals, "position": position,
        "sub_position": sub_position, "foot": foot,
        "country_of_citizenship": country, "league": league,
        "contract_months": contract_months,
        "cum_matches": cum_matches, "cum_goals": cum_goals,
        "cum_assists": cum_assists, "cum_minutes": cum_minutes,
        "num_transfers": num_transfers,
        "prev_value": prev_value, "peak_so_far": peak_so_far,
        "days_since_prev": days_since_prev, "player_highest_mv": player_highest_mv,
    }])
    val = predict_df(row)[0]
    return f"EUR {val:,.0f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age", value=24),
        gr.Number(label="Height (cm)", value=180),
        gr.Number(label="International caps", value=0),
        gr.Number(label="International goals", value=0),
        gr.Dropdown(POSITIONS, label="Position", value="Attack"),
        gr.Textbox(label="Sub-position", value="Centre-Forward"),
        gr.Dropdown(FEET, label="Foot", value="right"),
        gr.Textbox(label="Country of citizenship", value="Germany"),
        gr.Textbox(label="League (competition id)", value="GB1"),
        gr.Number(label="Contract months remaining", value=12),
        gr.Number(label="Cumulative matches", value=0),
        gr.Number(label="Cumulative goals", value=0),
        gr.Number(label="Cumulative assists", value=0),
        gr.Number(label="Cumulative minutes", value=0),
        gr.Number(label="Number of transfers", value=0),
        gr.Number(label="Previous market value (EUR)", value=0),
        gr.Number(label="Peak market value so far (EUR)", value=0),
        gr.Number(label="Days since previous valuation", value=9999),
        gr.Number(label="Player highest market value (EUR)", value=0),
    ],
    outputs=gr.Textbox(label="Predicted market value"),
    title="Football Player Market Value Predictor",
)

if __name__ == "__main__":
    demo.launch()
