"""
Trains XGBoost on full dataset and extracts feature importance.
Answers RQ2: which modalities contribute most to forecast accuracy?
"""
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "model_features.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENTIMENT_FEATURES = ["sentiment_score", "article_count", "daily_sentiment_lag_48", "article_count_lag_48"]

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("datetime_local").reset_index(drop=True)
    print(f"Loaded: {df.shape}")

    exclude_cols = ["datetime_local", "settlementDate", "systemSellPrice", "systemBuyPrice"]
    features = [c for c in df.columns if c not in exclude_cols]
    y = df["systemSellPrice"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print(f"Saved feature importance to feature_importance.csv")
    print(f"\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    modality_map = {}
    for f in features:
        if f in SENTIMENT_FEATURES:
            modality_map[f] = "NLP Sentiment"
        elif f in ["temperature_2m", "wind_speed_10m", "cloud_cover", "solar_generation_mw"]:
            modality_map[f] = "Weather"
        elif f in ["gas_price_close", "gas_price_high", "gas_price_low",
                    "gas_price_close_lag_48", "gas_price_close_lag_96", "gas_price_day_change"]:
            modality_map[f] = "Gas"
        elif f in ["initialDemandOutturn", "initialTransmissionSystemDemandOutturn",
                    "total_interconnector_imports"] or f.startswith("INT") or f in ["BIOMASS","CCGT","COAL","INTELE","INTELEC","INTEW","INTFR","INTGRNL","INTIFA2","INTIRL","INTNED","INTNEM","INTNSL","INTVKL","NPSHYD","NUCLEAR","OCGT","OIL","OTHER","PS","WIND"]:
            modality_map[f] = "Grid / Generation"
        elif f in ["hour", "day_of_week", "month", "is_weekend"]:
            modality_map[f] = "Calendar"
        elif "lag" in f or "rolling" in f or f in ["systemSellPrice_lag_1","systemBuyPrice_lag_1","systemSellPrice_lag_2","systemSellPrice_lag_48","systemSellPrice_rolling_24h_mean","systemSellPrice_rolling_24h_std","systemSellPrice_rolling_7d_mean"]:
            modality_map[f] = "Price History"
        else:
            modality_map[f] = "Grid / Generation"

    importance["modality"] = importance["feature"].map(modality_map)
    modality_importance = importance.groupby("modality")["importance"].sum().sort_values(ascending=False)
    print(f"\n=== MODALITY IMPORTANCE ===")
    for m, v in modality_importance.items():
        print(f"  {m:25s} {v:.2%}")

    modality_importance.to_csv(os.path.join(OUTPUT_DIR, "modality_importance.csv"))
    print(f"\nSaved modality importance to modality_importance.csv")

if __name__ == "__main__":
    main()
