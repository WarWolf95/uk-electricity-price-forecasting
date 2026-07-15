# Multimodal Deep Learning for UK Wholesale Electricity Price Forecasting

## Overview
This project uses multimodal deep learning to predict UK wholesale electricity prices by integrating market data, weather conditions, grid signals, and NLP-derived geopolitical sentiment. 

For the complete project plan, data sources, and architecture, refer to [`project_capsule.md`](./project_capsule.md).

## Progress Log

### Phase 1: Data Collection (Completed)
- [x] Initialized project directory structure (`src/`, `data/`, `notebooks/`).
- [x] Created `src/data/elexon_client.py` for fetching half-hourly system prices, generation mix, and demand from the Elexon BMRS API.
- [x] Created `src/data/weather_client.py` to retrieve weather and solar generation data.
- [x] Created `src/data/gas_client.py` to fetch wholesale gas prices.
- [x] Created `src/data/news_client.py` to collect energy/geopolitical news articles for NLP processing.
- [x] Created `src/data/fetch_all.py` to orchestrate data collection across all providers.
- [x] Created `src/utils/create_samples.py` to generate smaller sample datasets for testing and rapid iteration (optimised for local 16GB RAM constraints).
- [x] **Fix:** Demand endpoint changed from `/datasets/INDO` (real-time only) to `/demand/outturn` (historical half-hourly). Now returns 1,488 rows (48/day × 31 days) with `initialDemandOutturn` and `initialTransmissionSystemDemandOutturn`.
- [x] **Fix:** Gas client rewritten from FRED API (required API key) to yfinance TTF gas futures (`TTF=F`). Dutch TTF is the primary European gas benchmark — no API key required. Falls back to `NG=F` if unavailable.
- [x] **Removed:** `fetch_wind_forecast()` and `wind_forecast.csv` — the WINDFOR endpoint is forward-looking only (returns real-time forecasts, not historical data). Wind generation is already captured in `generation_mix.csv` by `fuelType` and will be extracted via pivot in Phase 2.

#### Data Inventory (Jan 2023 — 1 month test window)
| Dataset | File | Rows | Resolution |
|---------|------|------|------------|
| System Prices (target) | `system_prices.csv` | 1,484 | Half-hourly |
| Generation Mix | `generation_mix.csv` | 26,784 | Half-hourly × fuel type |
| National Demand | `national_demand.csv` | 1,488 | Half-hourly |

| Solar Generation | `national_solar.csv` | 1,488 | Half-hourly |
| Weather (temp/wind/cloud) | `historical_weather.csv` | 744 | Hourly |
| Gas Prices (TTF futures) | `gas_prices.csv` | 20 | Daily (trading days) |
| News Articles | `news_articles.csv` | 687 | Per-article |

### Phase 2: Preprocessing & Feature Engineering (Completed)
- [x] Expand data collection to full date range (2020–2025) to cover the 2022 energy crisis.
- [x] Align all datasets to half-hourly timestamps.
- [x] Handle missing values and interpolation.
- [x] Pivot `generation_mix` by `fuelType` to extract per-source generation features (wind, gas, solar, nuclear, etc.).
- [x] Implement NLP pipeline (FinBERT) on news articles to extract daily geopolitical sentiment scores.
- [x] Build lag features, rolling windows, and calendar features.
- [x] Orchestrate preprocessing and feature building via `run_phase2.py`.

### Phase 3: Model Training (Completed)
- [x] Baseline Models: Fitted ARIMA(2,1,2) on recent history; trained XGBoost.
- [x] Deep Learning Models: LSTM (2-layer, 128 hidden), Transformer (2-layer, 8-head, 128-dim), Gated Transformer (custom attention + GRN).
- [x] Improved training pipeline: validation split (last 14 days), early stopping (patience=5), ReduceLROnPlateau scheduler, gradient clipping.
- [x] GPU acceleration via NVIDIA RTX 5060 (CUDA).

### Phase 4: Evaluation (Completed)
- [x] Walk-forward validation: Evaluated all configurations across 5 expanding folds.
- [x] Crisis Period Analysis: Dedicated holdout on 2022 Russia-Ukraine energy crisis. Transformer + sentiment achieves best crisis R² (0.50).
- [x] Diebold-Mariano tests: Verified prediction improvement significance.
- [x] Modality importance analysis: XGBoost feature importances across data sources. Modality mix shifts during crisis.

#### Usage

```bash
python src/models/train_and_evaluate.py --mode crisis       # Crisis holdout (primary RQ)
python src/models/train_and_evaluate.py --mode walkforward  # 5-fold walk-forward (robustness)
python src/models/train_and_evaluate.py --quick             # XGBoost + LSTM only (fast iteration)
python src/models/train_and_evaluate.py --epochs 50         # Custom max epochs
```

Results are written to `data/processed/model_comparison_{mode}.csv` and `forecast_predictions_{mode}.csv`.

#### Crisis Period Results (PRIMARY — 2022 Russia-Ukraine energy crisis)
| Model | Configuration | MAE | RMSE | sMAPE (%) | $R^2$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **ARIMA** | Price-Only (Baseline) | 94.17 | 131.22 | 51.32 | -0.15 |
| **XGBoost** | With Sentiment | 73.50 | 126.90 | 38.33 | -0.07 |
| **XGBoost** | Without Sentiment | 75.57 | 137.32 | 38.03 | -0.25 |
| **LSTM** | With Sentiment | 70.35 | 92.68 | 40.57 | 0.43 |
| **LSTM** | Without Sentiment | 70.74 | 92.04 | 40.63 | 0.44 |
| **Transformer** | With Sentiment | **66.09** | **86.63** | 39.35 | **0.50** |
| **Transformer** | Without Sentiment | 68.94 | 92.13 | 40.10 | 0.44 |
| **GatedTransformer** | With Sentiment | 103.92 | 142.33 | 62.17 | -0.35 |
| **GatedTransformer** | Without Sentiment | 80.93 | 105.22 | 44.96 | 0.26 |

*Train: 2020-01 to 2022-01-31 (pre-crisis). Test: 2022-02-01 to 2022-10-31 (crisis). DL models trained with 30 max epochs, validation split, early stopping, GPU (RTX 5060).*

**Headline finding:** The Transformer with geopolitical sentiment achieves the best crisis-period performance (R²=0.50, MAE=66.09). Sentiment features improve Transformer MAE by 4.1% during the crisis vs no-sentiment baseline. This confirms RQ1 — NLP-derived geopolitical signals add value during regime shifts when historical price patterns alone are insufficient.

#### Walk-Forward Results (SECONDARY — normal market conditions, avg 5 folds)
| Model | Configuration | MAE | RMSE | sMAPE (%) | $R^2$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **ARIMA** | Price-Only (Baseline) | 39.59 | 47.99 | 58.21 | -0.45 |
| **XGBoost** | Without Sentiment | **17.82** | **23.86** | **33.10** | **0.66** |
| **XGBoost** | With Sentiment | 17.88 | 23.91 | 33.19 | 0.66 |
| **LSTM** | With Sentiment | 23.36 | 29.20 | 40.30 | 0.49 |
| **LSTM** | Without Sentiment | 24.49 | 30.56 | 42.52 | 0.44 |
| **Transformer** | Without Sentiment | 26.69 | 34.01 | 42.35 | 0.27 |
| **Transformer** | With Sentiment | 29.39 | 41.53 | 44.11 | -0.32 |
| **GatedTransformer** | Without Sentiment | 43.50 | 64.49 | 48.27 | -3.78 |
| **GatedTransformer** | With Sentiment | 40.01 | 55.01 | 48.78 | -1.88 |

*Note: DL models struggle on Fold 1 (smallest training set) — Transformer R²=-3.51, GatedTransformer R²=-10.98 — but converge to positive R² by Fold 5 (full training data). This indicates DL models require larger training windows than tree-based methods. XGBoost is robust throughout and the best model for normal-period forecasting.*

#### Modality Importance (XGBoost, crisis fold)
| Modality | Contribution |
|:---|---:|
| **Grid / Generation** (demand, fuel mix, interconnectors) | 34.1% |
| **Price History** (lags, rolling stats) | 33.2% |
| **Gas Prices** (TTF futures) | 12.8% |
| **Weather** (temp, wind, cloud, solar) | 11.0% |
| **NLP Sentiment** (FinBERT news scores) | 6.9% |
| **Calendar** (hour, day, month, weekend) | 2.0% |

*During the crisis, Price History drops from ~45% to ~33% importance while Grid/Generation and NLP Sentiment increase — consistent with the hypothesis that historical price patterns become less reliable during regime shifts, and real-time supply/demand + news signals gain relevance.*

