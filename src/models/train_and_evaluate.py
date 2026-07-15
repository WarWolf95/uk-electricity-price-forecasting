#!/usr/bin/env python
"""
train_and_evaluate.py — Unified Model Training & Evaluation Pipeline.

Modes:
  --mode crisis       (default) Train 2020-01–2022-01, test 2022-02–2022-10 (energy crisis)
  --mode walkforward  5-fold walk-forward validation (original)

Usage:
    python src/models/train_and_evaluate.py
    python src/models/train_and_evaluate.py --mode walkforward --epochs 30
    python src/models/train_and_evaluate.py --quick   # XGBoost + LSTM only
"""

import os
import sys
import time
import json
import argparse
import logging
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_and_evaluate")
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cpu":
    torch.set_num_threads(4)

PROJECT_DIR = r"C:\Dissertation"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "model_features.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 96
BATCH_SIZE = 256
SENTIMENT_FEATURES = ['sentiment_score', 'article_count', 'daily_sentiment_lag_48', 'article_count_lag_48']

# ─── Datasets ────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len]

# ─── Model Definitions ───────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.encoder_input = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder_input(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x).squeeze(-1)

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.linear(x)
        val, gate = torch.chunk(x, 2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.linear_1(x)
        x = self.elu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(x + residual)

class GatedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=8, dropout=0.1):
        super().__init__()
        self.grn_input = GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead,
                                          batch_first=True, dropout=dropout)
        self.grn_post_attn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.grn_input(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.grn_post_attn(self.dropout(attn_out[:, -1, :]))
        return self.fc(x).squeeze(-1)

# ─── Training Helpers ────────────────────────────────────────────────

def train_pytorch_model(model, train_loader, val_loader=None, epochs=30):
    """Train with validation split, early stopping, and LR scheduling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)

        train_loss /= len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(model(bx), by)
                    val_loss += loss.item() * len(bx)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1:2d}/{epochs}  train: {train_loss:.6f}  val: {val_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e}  patience: {patience_counter}/{5}")

            if patience_counter >= 5:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1:2d}/{epochs}  train: {train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  Restored best model (val_loss: {best_val_loss:.6f})")

    return model


def predict_pytorch_model(model, X_scaled, seq_len=SEQ_LEN):
    model.eval()
    dataset = TimeSeriesDataset(X_scaled, np.zeros(len(X_scaled)), seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            preds.extend(model(bx.to(device)).cpu().numpy())
    return np.array([np.nan] * seq_len + preds)


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2 = float(1.0 - (np.sum((yt - yp) ** 2) / max(np.sum((yt - np.mean(yt)) ** 2), 1e-12)))
    return OrderedDict([("MAE", round(mae, 2)), ("RMSE", round(rmse, 2)),
                        ("sMAPE", round(smape(yt, yp), 2)), ("R2", round(r2, 4))])


def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    mask = ~np.isnan(y_pred1) & ~np.isnan(y_pred2)
    yt, yp1, yp2 = y_true[mask], y_pred1[mask], y_pred2[mask]
    e1, e2 = yt - yp1, yt - yp2
    d = e1**2 - e2**2
    mean_d, n = float(np.mean(d)), len(d)
    gamma = np.zeros(h)
    gamma[0] = float(np.var(d))
    for lag in range(1, h):
        gamma[lag] = float(np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d)))
    var_d = gamma[0] + 2 * np.sum(gamma[1:])
    if var_d <= 0:
        return 0.0, 1.0
    from scipy.stats import norm
    dm_stat = float(mean_d / np.sqrt(var_d / n))
    p_value = float(2 * (1.0 - norm.cdf(np.abs(dm_stat))))
    return round(dm_stat, 4), round(p_value, 4)


def get_modality_map(features):
    m = {}
    for f in features:
        if f in SENTIMENT_FEATURES:
            m[f] = "NLP Sentiment"
        elif f in ["temperature_2m", "wind_speed_10m", "cloud_cover", "solar_generation_mw"]:
            m[f] = "Weather"
        elif f in ["gas_price_close", "gas_price_high", "gas_price_low",
                    "gas_price_close_lag_48", "gas_price_close_lag_96", "gas_price_day_change"]:
            m[f] = "Gas Prices"
        elif f in ["hour", "day_of_week", "month", "is_weekend"]:
            m[f] = "Calendar"
        elif "lag" in f or "rolling" in f or f in ["systemSellPrice_lag_1", "systemBuyPrice_lag_1",
                                                     "systemSellPrice_lag_2", "systemSellPrice_lag_48",
                                                     "systemSellPrice_rolling_24h_mean",
                                                     "systemSellPrice_rolling_24h_std",
                                                     "systemSellPrice_rolling_7d_mean"]:
            m[f] = "Price History"
        else:
            m[f] = "Grid / Generation"
    return m


def run_single_split(train_df, test_df, configs, fold_label="1"):
    """Run all models on a single train/test split. Returns results list + pred_df slice."""
    results_list = []

    N_test = len(test_df)

    pred_df = test_df[['datetime_local', 'settlementDate', 'systemSellPrice']].copy()

    # ── ARIMA ──────────────────────────────────────────────────────
    logger.info("Training ARIMA(2,1,2) baseline...")
    arima_train = train_df['systemSellPrice'].values[-2000:]
    try:
        arima_model = ARIMA(arima_train, order=(2, 1, 2)).fit()
        arima_pred = arima_model.forecast(steps=N_test)
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}. Using naive seasonal baseline.")
        arima_pred = train_df['systemSellPrice'].values[-N_test:]

    arima_metrics = calculate_metrics(test_df['systemSellPrice'].values, arima_pred)
    logger.info(f"ARIMA: MAE={arima_metrics['MAE']}  R2={arima_metrics['R2']}")
    results_list.append({"Fold": fold_label, "Model": "ARIMA", "Config": "Price-Only", **arima_metrics})
    pred_df["pred_ARIMA"] = arima_pred

    # ── XGBoost feature importance (trained once on full feature set) ──
    exclude_cols_full = ['datetime_local', 'settlementDate', 'systemSellPrice', 'systemBuyPrice']
    full_features = [c for c in train_df.columns if c not in exclude_cols_full]
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(train_df[full_features])
    X_test_full = scaler_full.transform(test_df[full_features])
    y_train = train_df['systemSellPrice'].values
    y_test = test_df['systemSellPrice'].values

    xgb_all = xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1, learning_rate=0.05)
    xgb_all.fit(X_train_full, y_train)
    imp_df = pd.DataFrame({"feature": full_features, "importance": xgb_all.feature_importances_})
    imp_df["modality"] = imp_df["feature"].map(get_modality_map(full_features))
    imp_df.to_csv(os.path.join(OUTPUT_DIR, f"feature_importance_{fold_label}.csv"), index=False)
    modality_pct = imp_df.groupby("modality")["importance"].sum().sort_values(ascending=False)
    logger.info(f"Modality importance:\n" + "\n".join(f"  {m:25s} {v:.2%}" for m, v in modality_pct.items()))

    # ── Per-config models (With/Without Sentiment) ─────────────────
    for config in configs:
        logger.info(f"--- Config: {config} ---")

        exclude_cols = ['datetime_local', 'settlementDate', 'systemSellPrice', 'systemBuyPrice']
        if config == "Without Sentiment":
            exclude_cols += SENTIMENT_FEATURES

        features = [c for c in train_df.columns if c not in exclude_cols]

        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(train_df[features])
        X_test = scaler_X.transform(test_df[features])

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Validation split for DL (last 14 days of training)
        val_size = min(14 * 48, len(X_train) - SEQ_LEN - 1)
        X_val_seq = X_train[-val_size:]
        y_val_seq = y_train_scaled[-val_size:]
        X_train_dl = X_train[:-val_size] if val_size > 0 else X_train
        y_train_dl = y_train_scaled[:-val_size] if val_size > 0 else y_train_scaled

        train_dataset = TimeSeriesDataset(X_train_dl, y_train_dl)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_loader = None
        if len(X_val_seq) > SEQ_LEN + 10:
            val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        combined_X = np.vstack([X_train[-SEQ_LEN:], X_test])
        config_label = config.replace(' ', '_')

        # XGBoost
        logger.info("  XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1, learning_rate=0.05)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        pred_df[f"pred_XGBoost_{config_label}"] = xgb_pred
        m = calculate_metrics(y_test, xgb_pred)
        logger.info(f"  XGBoost ({config}): MAE={m['MAE']}  sMAPE={m['sMAPE']}  R2={m['R2']}")
        results_list.append({"Fold": fold_label, "Model": "XGBoost", "Config": config, **m})

        # LSTM
        logger.info("  LSTM...")
        lstm = LSTMModel(input_dim=len(features))
        lstm = train_pytorch_model(lstm, train_loader, val_loader, epochs=ARGS.epochs)
        lstm_pred = scaler_y.inverse_transform(
            predict_pytorch_model(lstm, combined_X)[SEQ_LEN:].reshape(-1, 1)).flatten()
        pred_df[f"pred_LSTM_{config_label}"] = lstm_pred
        m = calculate_metrics(y_test, lstm_pred)
        logger.info(f"  LSTM ({config}): MAE={m['MAE']}  sMAPE={m['sMAPE']}  R2={m['R2']}")
        results_list.append({"Fold": fold_label, "Model": "LSTM", "Config": config, **m})

        # Transformer
        if not ARGS.quick:
            logger.info("  Transformer...")
            transformer = TransformerModel(input_dim=len(features))
            transformer = train_pytorch_model(transformer, train_loader, val_loader, epochs=ARGS.epochs)
            transformer_pred = scaler_y.inverse_transform(
                predict_pytorch_model(transformer, combined_X)[SEQ_LEN:].reshape(-1, 1)).flatten()
            pred_df[f"pred_Transformer_{config_label}"] = transformer_pred
            m = calculate_metrics(y_test, transformer_pred)
            logger.info(f"  Transformer ({config}): MAE={m['MAE']}  sMAPE={m['sMAPE']}  R2={m['R2']}")
            results_list.append({"Fold": fold_label, "Model": "Transformer", "Config": config, **m})

        # GatedTransformer
        if not ARGS.quick:
            logger.info("  GatedTransformer...")
            gt = GatedTransformer(input_dim=len(features))
            gt = train_pytorch_model(gt, train_loader, val_loader, epochs=ARGS.epochs)
            gt_pred = scaler_y.inverse_transform(
                predict_pytorch_model(gt, combined_X)[SEQ_LEN:].reshape(-1, 1)).flatten()
            pred_df[f"pred_GatedTransformer_{config_label}"] = gt_pred
            m = calculate_metrics(y_test, gt_pred)
            logger.info(f"  GatedTransformer ({config}): MAE={m['MAE']}  sMAPE={m['sMAPE']}  R2={m['R2']}")
            results_list.append({"Fold": fold_label, "Model": "GatedTransformer", "Config": config, **m})

    return results_list, pred_df


# ─── Main ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate electricity price forecasting models.")
    p.add_argument("--mode", choices=["crisis", "walkforward"], default="crisis",
                   help="Evaluation mode: 'crisis' (train pre-2022, test on 2022 crisis) or 'walkforward'")
    p.add_argument("--epochs", type=int, default=30, help="Max epochs for DL models")
    p.add_argument("--quick", action="store_true", help="Skip Transformer and GatedTransformer for fast iteration")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


ARGS = None


def main():
    global ARGS
    ARGS = parse_args()

    logger.info(f"Mode: {ARGS.mode} | Epochs: {ARGS.epochs} | Quick: {ARGS.quick}")
    logger.info(f"Device: {device}")

    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)

    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}!")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df['datetime_local'] = pd.to_datetime(df['datetime_local'], utc=True)
    df = df.sort_values('datetime_local').reset_index(drop=True)
    logger.info(f"Loaded data: {df.shape}  ({df['settlementDate'].min()} to {df['settlementDate'].max()})")

    configs = ["With Sentiment", "Without Sentiment"]
    all_results = []
    all_pred_parts = []

    if ARGS.mode == "crisis":
        # ── Crisis Mode ────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("CRISIS PERIOD EVALUATION")
        logger.info("Train: 2020-01 to 2022-01-31  |  Test: 2022-02-01 to 2022-10-31")
        logger.info("=" * 60)

        train_df = df[df['settlementDate'] < "2022-02-01"].copy()
        test_df = df[(df['settlementDate'] >= "2022-02-01") & (df['settlementDate'] <= "2022-10-31")].copy()
        logger.info(f"Train: {len(train_df)} rows ({train_df['settlementDate'].min()} to {train_df['settlementDate'].max()})")
        logger.info(f"Test:  {len(test_df)} rows ({test_df['settlementDate'].min()} to {test_df['settlementDate'].max()})")

        results, pred_df = run_single_split(train_df, test_df, configs, fold_label="crisis")
        all_results.extend(results)
        all_pred_parts.append(pred_df)
        suffix = "crisis"

    else:
        # ── Walk-Forward Mode ──────────────────────────────────────
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION (5 folds)")
        logger.info("=" * 60)

        test_fold_size = 1440
        num_folds = 5
        N = len(df)

        for fold in range(num_folds):
            fold_idx = fold + 1
            test_start = N - (num_folds - fold) * test_fold_size
            test_end = test_start + test_fold_size

            train_df = df.iloc[:test_start].copy()
            test_df = df.iloc[test_start:test_end].copy()

            logger.info(f"--- Fold {fold_idx}/{num_folds} ---")
            logger.info(f"  Train: {train_df['settlementDate'].min()} to {train_df['settlementDate'].max()} ({len(train_df)} rows)")
            logger.info(f"  Test:  {test_df['settlementDate'].min()} to {test_df['settlementDate'].max()} ({len(test_df)} rows)")

            results, pred_df = run_single_split(train_df, test_df, configs, fold_label=str(fold_idx))
            all_results.extend(results)
            all_pred_parts.append(pred_df)

        suffix = "walkforward"

    # ── Save Results ───────────────────────────────────────────────
    combined_pred = pd.concat(all_pred_parts, ignore_index=True)
    pred_path = os.path.join(OUTPUT_DIR, f"forecast_predictions_{suffix}.csv")
    combined_pred.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions: {pred_path}")

    metrics_df = pd.DataFrame(all_results)
    metrics_path = os.path.join(OUTPUT_DIR, f"model_comparison_{suffix}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics: {metrics_path}")

    # ── Summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (averaged across folds)")
    logger.info("=" * 60)
    summary = metrics_df.groupby(["Model", "Config"])[["MAE", "RMSE", "sMAPE", "R2"]].mean().round(2).reset_index()
    print(summary.to_string(index=False))

    # ── Diebold-Mariano Tests ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DIEBOLD-MARIANO TESTS (Sentiment: Without vs With)")
    logger.info("=" * 60)

    dm_models = ["XGBoost", "LSTM"]
    if not ARGS.quick:
        dm_models += ["Transformer", "GatedTransformer"]

    for model_name in dm_models:
        col_with = f"pred_{model_name}_With_Sentiment"
        col_without = f"pred_{model_name}_Without_Sentiment"

        if col_with not in combined_pred.columns or col_without not in combined_pred.columns:
            continue

        yt = combined_pred['systemSellPrice'].values
        stat, pval = diebold_mariano_test(yt, combined_pred[col_without].values, combined_pred[col_with].values)
        sig = "significant" if pval < 0.05 else "not significant"
        direction = "sentiment improved" if stat > 0 else "sentiment degraded"
        logger.info(f"{model_name:20s}  DM={stat:+7.4f}  p={pval:.4f}  ({direction}, {sig})")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
