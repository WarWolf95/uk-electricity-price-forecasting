#!/usr/bin/env python
"""
build_features.py — Computes lags, rolling features, calendar flags, and consolidated grid metrics.
Merges with daily news sentiment scores and outputs the final model-ready dataset.
Ensures zero data leakage by shifting historical features correctly.
"""

import os
import logging
# pyrefly: ignore [missing-import]
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_features")

DB_PATH = "data/db/electricity.duckdb"
OUTPUT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "model_features.csv")

def main():
    logger.info("Initializing Feature Engineering Pipeline...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"DuckDB database not found at {DB_PATH}!")
        return

    conn = duckdb.connect(DB_PATH)
    
    # 1. Load aligned features
    logger.info("Loading aligned features from DuckDB...")
    df = conn.execute("SELECT * FROM aligned_features ORDER BY datetime_local").fetch_df()
    logger.info(f"Loaded aligned features. Shape: {df.shape}")
    
    # Convert datetime_local to datetime object and set as index
    df['datetime_local'] = pd.to_datetime(df['datetime_local'], utc=True).dt.tz_convert('Europe/London')
    df = df.sort_values('datetime_local').reset_index(drop=True)

    # 2. Check and load daily sentiment
    logger.info("Checking for daily sentiment data...")
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    
    if 'daily_sentiment' in tables:
        logger.info("Loading daily sentiment scores...")
        df_sent = conn.execute("SELECT * FROM daily_sentiment").fetch_df()
        logger.info(f"Loaded daily sentiment. Shape: {df_sent.shape}")
        
        # Merge sentiment scores into aligned features
        # daily_sentiment matches settlementDate (YYYY-MM-DD)
        df = pd.merge(df, df_sent, left_on='settlementDate', right_on='date', how='left')
        df = df.drop(columns=['date'])
        
        # Handle missing dates by forward-filling (sentiment) and zero-filling (article count)
        df['sentiment_score'] = df['sentiment_score'].ffill().bfill().fillna(0.0)
        df['article_count'] = df['article_count'].fillna(0.0)
    else:
        logger.warning("Table 'daily_sentiment' not found in database! Creating placeholder sentiment features...")
        df['sentiment_score'] = 0.0
        df['article_count'] = 0.0

    # 3. Compute Features
    logger.info("Computing lag, rolling, grid, and calendar features...")
    
    # A. Price Features (Target: systemSellPrice)
    # 1-period (30 min) lag
    df['systemSellPrice_lag_1'] = df['systemSellPrice'].shift(1)
    df['systemBuyPrice_lag_1'] = df['systemBuyPrice'].shift(1)
    
    # 2-period (1 hour) lag
    df['systemSellPrice_lag_2'] = df['systemSellPrice'].shift(2)
    
    # 48-period (24 hour) lag
    df['systemSellPrice_lag_48'] = df['systemSellPrice'].shift(48)
    
    # Rolling averages (using shifted values to prevent data leakage)
    df['systemSellPrice_rolling_24h_mean'] = df['systemSellPrice_lag_1'].rolling(48).mean()
    df['systemSellPrice_rolling_24h_std'] = df['systemSellPrice_lag_1'].rolling(48).std()
    df['systemSellPrice_rolling_7d_mean'] = df['systemSellPrice_lag_1'].rolling(336).mean()
    
    # B. Gas Price Features (Daily prices, lagged to avoid same-day leakage)
    # Lag gas prices by 48 periods (24 hours) to represent day-ahead availability
    df['gas_price_close_lag_48'] = df['gas_price_close'].shift(48)
    df['gas_price_close_lag_96'] = df['gas_price_close'].shift(96)
    df['gas_price_day_change'] = df['gas_price_close_lag_48'] - df['gas_price_close_lag_96']
    
    # C. Grid features
    # Consolidate interconnector imports (all columns starting with INT)
    int_cols = [c for c in df.columns if c.startswith('INT')]
    logger.info(f"Consolidating interconnector columns: {int_cols}")
    df['total_interconnector_imports'] = df[int_cols].sum(axis=1)
    
    # D. Geopolitical Sentiment Features
    # Lag sentiment by 48 periods (24 hours) so we forecast using news from the prior day
    df['daily_sentiment_lag_48'] = df['sentiment_score'].shift(48)
    df['article_count_lag_48'] = df['article_count'].shift(48)
    
    # E. Calendar Features
    df['hour'] = df['datetime_local'].dt.hour
    df['day_of_week'] = df['datetime_local'].dt.dayofweek
    df['month'] = df['datetime_local'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 4. Handle NaNs introduced by rolling features and lags
    # The 7-day rolling window introduces 336 NaN rows at the start.
    logger.info("Dropping initial rows with NaN values from lags/rolling windows...")
    df_clean = df.dropna().copy()
    logger.info(f"Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    
    # 5. Verification Checks
    logger.info("Verifying final features dataset...")
    null_counts = df_clean.isnull().sum()
    null_sum = null_counts.sum()
    if null_sum == 0:
        logger.info("Zero missing values in final dataset! Check passed.")
    else:
        logger.warning(f"Found {null_sum} missing values in final dataset:")
        for col, c in null_counts.items():
            if c > 0:
                logger.warning(f"  {col}: {c} nulls")
                
    # 6. Save to DuckDB and CSV
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    logger.info(f"Writing final feature set to CSV: {OUTPUT_CSV}...")
    df_clean.to_csv(OUTPUT_CSV, index=False)
    
    # Convert datetime_local to string for DuckDB compatibility
    df_db = df_clean.copy()
    df_db['datetime_local'] = df_db['datetime_local'].astype(str)
    
    logger.info("Writing final feature set to DuckDB table 'model_features'...")
    conn.execute("DROP TABLE IF EXISTS model_features")
    conn.register("df_temp", df_db)
    conn.execute("CREATE TABLE model_features AS SELECT * FROM df_temp")
    
    rows_written = conn.execute("SELECT count(*) FROM model_features").fetchone()[0]
    logger.info(f"SUCCESSfully created table 'model_features'. Rows: {rows_written:,}")
    
    conn.close()
    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    main()
