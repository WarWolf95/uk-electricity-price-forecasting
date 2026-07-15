"""
Recalculates model metrics using sMAPE instead of MAPE.
Reads saved predictions from forecast_predictions.csv.
No retraining needed — just metric recomputation.
"""
import os
import pandas as pd
import numpy as np

PROJECT_DIR = r"C:\Dissertation"
PRED_PATH = os.path.join(PROJECT_DIR, "data", "processed", "forecast_predictions.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "processed", "model_comparison_smape.csv")

TEST_FOLD_SIZE = 1440

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    r2 = 1.0 - (np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "sMAPE": round(smape(yt, yp), 2), "R2": round(r2, 4)}

def main():
    df = pd.read_csv(PRED_PATH)
    print(f"Loaded predictions: {df.shape}")

    y_true = df["systemSellPrice"].values

    model_cols = [c for c in df.columns if c.startswith("pred_")]

    results = []
    for fold in range(5):
        start = fold * TEST_FOLD_SIZE
        end = start + TEST_FOLD_SIZE
        yt = y_true[start:end]

        for col in model_cols:
            yp = df[col].values[start:end]

            if col == "pred_ARIMA":
                model_name = "ARIMA"
                config = "Price-Only"
            else:
                parts = col.replace("pred_", "").split("_With_") if "_With_" in col else col.replace("pred_", "").split("_Without_")
                model_key = parts[0]
                if model_key == "TFT":
                    model_key = "GatedTransformer"
                config = "With Sentiment" if "_With_" in col else "Without Sentiment"
                model_name = model_key

            metrics = calculate_metrics(yt, yp)
            results.append({"Fold": fold + 1, "Model": model_name, "Config": config, **metrics})

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved per-fold metrics to {OUTPUT_PATH}")

    summary = result_df.groupby(["Model", "Config"])[["MAE", "RMSE", "sMAPE", "R2"]].mean().round(2).reset_index()
    print("\n=== AVERAGE ACROSS ALL FOLDS ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
