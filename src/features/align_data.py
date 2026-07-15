#!/usr/bin/env python
"""
align_data.py — Aligns all UK electricity dataset tables to a unified half-hourly grid.

Performs:
1. Generation of Europe/London local timezone-aware half-hourly grid (2020-01-01 to 2025-06-23).
2. Calculation of settlementDate and settlementPeriod mapping, handling DST transitions.
3. Pivoting of generation_mix by fuelType (fill NaNs with 0).
4. Alignment and merging of system prices, demand, solar, weather, and gas prices.
5. Linear interpolation of missing values and forward-filling of daily gas prices.
6. Validation checks (missingness, row count).
7. Outputs unified table to DuckDB (aligned_features) and data/processed/aligned_features.csv.
"""

import os
import logging
import pandas as pd
# pyrefly: ignore [missing-import]
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/db/electricity.duckdb"
OUTPUT_DIR = "data/processed"
CSV_PATH = os.path.join(OUTPUT_DIR, "aligned_features.csv")

def main():
    logger.info("Initializing alignment process...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

    # 1. Connect to DuckDB
    if not os.path.exists(DB_PATH):
        logger.error(f"DuckDB database not found at {DB_PATH}!")
        return
        
    conn = duckdb.connect(DB_PATH)
    try:
        # 2. Generate Master Grid (timezone-aware)
        logger.info("Generating half-hourly Europe/London timezone-aware master grid...")
        dr = pd.date_range(start='2020-01-01 00:00:00', end='2025-06-23 23:30:00', freq='30min', tz='Europe/London')
        df_grid = pd.DataFrame(index=dr)
        df_grid['settlementDate'] = df_grid.index.strftime('%Y-%m-%d')
        # Compute settlement period as chronological index per day (handles 46, 48, 50 periods on DST change days)
        df_grid['settlementPeriod'] = df_grid.groupby('settlementDate').cumcount() + 1
        df_grid = df_grid.reset_index().rename(columns={'index': 'datetime_local'})
        
        logger.info(f"Master grid generated. Shape: {df_grid.shape}")

        # 3. Join System Prices (Target)
        logger.info("Merging system prices...")
        df_sp = conn.execute("SELECT settlementDate, settlementPeriod, systemSellPrice, systemBuyPrice FROM system_prices").fetch_df()
        # Drop duplicates if any
        df_sp = df_sp.drop_duplicates(subset=['settlementDate', 'settlementPeriod'])
        df_merged = pd.merge(df_grid, df_sp, on=['settlementDate', 'settlementPeriod'], how='left')
        
        # 4. Join Generation Mix (Pivoted)
        logger.info("Processing and merging generation mix...")
        df_gen = conn.execute("SELECT settlementDate, settlementPeriod, fuelType, generation FROM generation_mix").fetch_df()
        # Drop duplicates before pivoting to avoid Reshape error
        df_gen = df_gen.drop_duplicates(subset=['settlementDate', 'settlementPeriod', 'fuelType'], keep='first')
        # Pivot fuel types
        df_gen_pivot = df_gen.pivot(index=['settlementDate', 'settlementPeriod'], columns='fuelType', values='generation').reset_index()
        # Clean fuel type columns
        fuel_cols = [c for c in df_gen_pivot.columns if c not in ['settlementDate', 'settlementPeriod']]
        # Fill missing fuel types with 0 before merging (e.g. if a fuel type isn't active/reported)
        df_gen_pivot[fuel_cols] = df_gen_pivot[fuel_cols].fillna(0)
        df_gen_pivot = df_gen_pivot.drop_duplicates(subset=['settlementDate', 'settlementPeriod'])
        
        df_merged = pd.merge(df_merged, df_gen_pivot, on=['settlementDate', 'settlementPeriod'], how='left')
        
        # 5. Join National Demand
        logger.info("Merging national demand...")
        df_demand = conn.execute("SELECT settlementDate, settlementPeriod, initialDemandOutturn, initialTransmissionSystemDemandOutturn FROM national_demand").fetch_df()
        df_demand = df_demand.drop_duplicates(subset=['settlementDate', 'settlementPeriod'])
        df_merged = pd.merge(df_merged, df_demand, on=['settlementDate', 'settlementPeriod'], how='left')

        # 6. Join National Solar
        logger.info("Merging solar generation...")
        df_solar = conn.execute("SELECT datetime, generation_mw FROM national_solar").fetch_df()
        df_solar['datetime_local'] = pd.to_datetime(df_solar['datetime'], utc=True).dt.tz_convert('Europe/London')
        df_solar = df_solar.dropna(subset=['datetime_local']).drop_duplicates(subset=['datetime_local'])
        df_merged = pd.merge(df_merged, df_solar[['datetime_local', 'generation_mw']], on='datetime_local', how='left')
        df_merged = df_merged.rename(columns={'generation_mw': 'solar_generation_mw'})

        # 7. Join Historical Weather
        logger.info("Merging historical weather...")
        df_weather = conn.execute("SELECT time, temperature_2m, wind_speed_10m, cloud_cover FROM historical_weather").fetch_df()
        df_weather['datetime_local'] = pd.to_datetime(df_weather['time']).dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='shift_forward')
        df_weather = df_weather.dropna(subset=['datetime_local']).drop_duplicates(subset=['datetime_local'])
        df_merged = pd.merge(df_merged, df_weather[['datetime_local', 'temperature_2m', 'wind_speed_10m', 'cloud_cover']], on='datetime_local', how='left')

        # 8. Join Gas Prices
        logger.info("Merging gas prices...")
        df_gas = conn.execute("SELECT date AS settlementDate, gas_price_close, gas_price_high, gas_price_low FROM gas_prices").fetch_df()
        df_gas = df_gas.drop_duplicates(subset=['settlementDate'])
        df_merged = pd.merge(df_merged, df_gas, on='settlementDate', how='left')

        # 9. Handle missing values and interpolation
        logger.info("Interpolating missing values and forward-filling gas prices...")
        # A. Gas prices - daily closing prices need forward-fill (ffill then bfill for start boundary)
        gas_features = ['gas_price_close', 'gas_price_high', 'gas_price_low']
        df_merged[gas_features] = df_merged[gas_features].ffill().bfill()
        
        # B. Numeric columns to linearly interpolate (weather, solar, prices, demand, fuel mix)
        # Avoid string columns like settlementDate and datetime_local (which is DatetimeTZDtype)
        numeric_cols = [
            'systemSellPrice', 'systemBuyPrice', 'initialDemandOutturn', 'initialTransmissionSystemDemandOutturn',
            'solar_generation_mw', 'temperature_2m', 'wind_speed_10m', 'cloud_cover'
        ] + fuel_cols
        
        # Interpolate numeric columns to resolve missing hours / settlement periods
        df_merged[numeric_cols] = df_merged[numeric_cols].interpolate(method='linear', limit_direction='both')

        # 10. Verification Checks
        logger.info("Verifying aligned features...")
        null_counts = df_merged.isnull().sum()
        logger.info(f"Aligned shape: {df_merged.shape}")
        logger.info("Checking for any remaining null values:")
        for col, null_c in null_counts.items():
            if null_c > 0:
                logger.warning(f"  {col}: {null_c} nulls")
        if null_counts.sum() == 0:
            logger.info("  Zero null values found! Dataset is completely clean and aligned.")

        # 11. Save output
        logger.info(f"Writing aligned dataset to CSV at: {CSV_PATH}...")
        df_merged.to_csv(CSV_PATH, index=False)
        logger.info("CSV write complete.")
        
        # Convert datetime_local to string for DuckDB compatibility
        df_merged_db = df_merged.copy()
        df_merged_db['datetime_local'] = df_merged_db['datetime_local'].astype(str)
        
        logger.info("Writing aligned dataset to DuckDB table 'aligned_features'...")
        # Drop table if exists, write new
        conn.execute("DROP TABLE IF EXISTS aligned_features")
        conn.register("df_temp", df_merged_db)
        conn.execute("CREATE TABLE aligned_features AS SELECT * FROM df_temp")
        logger.info("DuckDB table write complete.")

        # Print quick summary of row counts
        logger.info("=" * 60)
        logger.info(f"SUCCESSfully Aligned All Datasets")
        logger.info(f"  Table: aligned_features")
        logger.info(f"  Rows: {conn.execute('SELECT count(*) FROM aligned_features').fetchone()[0]:,}")
        logger.info("=" * 60)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
