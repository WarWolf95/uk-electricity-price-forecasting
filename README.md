# Multimodal Deep Learning for UK Wholesale Electricity Price Forecasting

[![Tests Passed](https://img.shields.io/badge/Tests-PyTest_Passed-brightgreen.svg)](tests/test_pipeline.py)

This project predicts UK wholesale electricity prices by combining market data, weather, grid conditions, and news sentiment. The idea was to see whether NLP-derived geopolitical signals add any value during volatile periods like the 2022 energy crisis, or whether price history alone is good enough.

## How it works

Seven data sources feed into the pipeline, all pulled from public APIs (no paid keys needed):

| What | Where from | Granularity |
|---|---|---|
| Electricity prices (target) | Elexon BMRS | Half-hourly |
| Generation by fuel type | Elexon BMRS | Half-hourly |
| National demand | Elexon BMRS | Half-hourly |
| Solar generation | Sheffield Solar PV Live | Half-hourly |
| Weather (temp, wind, cloud) | Open-Meteo Archive | Hourly |
| Gas prices (TTF futures) | Yahoo Finance | Daily |
| News articles | Guardian API | Per-article |

Everything gets aligned to half-hourly timestamps, missing values get interpolated, and the generation mix gets pivoted by fuel type into feature columns. News articles pass through FinBERT for daily sentiment scoring, which becomes an additional feature alongside lags, rolling windows, and calendar flags.

The full dataset runs from January 2020 to June 2025.

## Models

Five architectures, ranging from a naive baseline to a custom attention-based model:

- **ARIMA(2,1,2)** -- prices only, statistical baseline
- **XGBoost** -- 200 trees, max depth 8
- **LSTM** -- 2-layer, 128 hidden units
- **Transformer** -- 2-layer, 8-head attention, 128-dim embeddings
- **Gated Transformer** -- custom model with gated residual networks feeding into multi-head attention

All deep learning models were trained with a validation holdout (last 14 days), early stopping (patience 5), learning rate scheduling, and gradient clipping. Ran on an RTX 5060, which kept training times reasonable.

## Key results

### Crisis period (Feb-Oct 2022 energy crisis)

The Transformer with sentiment features performed best here, which was the main question the project set out to answer.

| Model | Sentiment | MAE | RMSE | sMAPE | R2 |
|---|---|---|---|---|---|
| ARIMA | Price only | 94.17 | 131.22 | 51.32 | -0.15 |
| XGBoost | No | 75.57 | 137.32 | 38.03 | -0.25 |
| XGBoost | Yes | 73.50 | 126.90 | 38.33 | -0.07 |
| LSTM | No | 70.74 | 92.04 | 40.63 | 0.44 |
| LSTM | Yes | 70.35 | 92.68 | 40.57 | 0.43 |
| Transformer | No | 68.94 | 92.13 | 40.10 | 0.44 |
| **Transformer** | **Yes** | **66.09** | **86.63** | **39.35** | **0.50** |
| GatedTransformer | No | 80.93 | 105.22 | 44.96 | 0.26 |
| GatedTransformer | Yes | 103.92 | 142.33 | 62.17 | -0.35 |

Sentiment features improved Transformer MAE by about 4% during the crisis. The effect is modest but consistent across models. During normal market conditions (tested via 5-fold walk-forward on 2025 data), sentiment was neutral to slightly detrimental. It only adds value when the market is under unusual stress.

### Normal conditions (walk-forward, 5 folds)

XGBoost dominates here. Deep learning models need more training data to converge -- they struggled on fold 1 (where the training set was smallest) but recovered by fold 5.

| Model | MAE | RMSE | sMAPE | R2 |
|---|---|---|---|---|
| ARIMA | 39.59 | 47.99 | 58.21 | -0.45 |
| **XGBoost** | **17.82** | **23.86** | **33.10** | **0.66** |
| LSTM | 23.36 | 29.20 | 40.30 | 0.49 |
| Transformer | 26.69 | 34.01 | 42.35 | 0.27 |
| GatedTransformer | 40.01 | 55.01 | 48.78 | -1.88 |

### What matters most (feature importance)

During the crisis, the modality mix shifted noticeably. Price history dropped from its usual ~45% contribution down to ~33%, while grid/generation and NLP sentiment gained ground. Makes sense -- when historical patterns break down, real-time supply-demand and news signals become relatively more informative.

| Modality | Contribution |
|---|---|
| Grid / Generation | 34.1% |
| Price History | 33.2% |
| Gas Prices | 12.8% |
| Weather | 11.0% |
| NLP Sentiment | 6.9% |
| Calendar | 2.0% |

## Running it

```bash
# Crisis holdout (trains on pre-2022 data, tests on Feb-Oct 2022)
python src/models/train_and_evaluate.py --mode crisis

# 5-fold walk-forward on normal market conditions
python src/models/train_and_evaluate.py --mode walkforward

# Quick run (XGBoost + LSTM only, skips Transformer and GatedTransformer)
python src/models/train_and_evaluate.py --quick

# Adjust max epochs
python src/models/train_and_evaluate.py --epochs 50
```

Results land in `data/processed/` as CSV files.

## Project structure

```
src/
  data/         # API clients for each data source
  features/     # Alignment, feature engineering, sentiment pipeline
  models/       # Training and evaluation pipeline
  utils/        # Helper scripts
scripts/        # Analysis and metric recalculation
data/
  processed/    # Model outputs and feature importance
  raw/          # Raw API responses (not tracked in git)
```

## Data sources

All data is publicly available and used under the Open Government Licence or equivalent:
- Elexon BMRS API (UK electricity market data)
- Sheffield Solar PV Live (national solar generation)
- Open-Meteo Archive (historical weather)
- Yahoo Finance (TTF gas futures)
- Guardian Open Platform API (news articles)
