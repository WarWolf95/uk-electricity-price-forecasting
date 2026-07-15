import logging
import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GasClient:
    """
    Client for fetching wholesale gas prices via yfinance.
    Uses Dutch TTF gas futures (TTF=F) — the main European gas benchmark
    that directly sets the marginal price for UK gas-fired electricity generation.
    No API key required.
    """
    
    # TTF (Dutch Title Transfer Facility) — primary European gas benchmark
    PRIMARY_TICKER = "TTF=F"
    # UK Natural Gas as fallback
    FALLBACK_TICKER = "NG=F"
    
    def __init__(self):
        pass

    def fetch_gas_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily gas prices from Yahoo Finance.
        Returns: DataFrame with columns [date, gas_price_close, gas_price_high, gas_price_low].
        """
        logger.info(f"Fetching gas prices (TTF futures) from {start_date} to {end_date}")
        
        # yfinance end_date is exclusive, so add 1 day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_adjusted = end_dt.strftime("%Y-%m-%d")
        
        try:
            df = yf.download(
                self.PRIMARY_TICKER,
                start=start_date,
                end=end_adjusted,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data from {self.PRIMARY_TICKER}, trying fallback {self.FALLBACK_TICKER}")
                df = yf.download(
                    self.FALLBACK_TICKER,
                    start=start_date,
                    end=end_adjusted,
                    progress=False,
                    auto_adjust=True
                )
            
            if df.empty:
                logger.error("No gas price data available from any source.")
                return pd.DataFrame()
            
            # yfinance returns MultiIndex columns when downloading single ticker
            # Flatten if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Close": "gas_price_close",
                "High": "gas_price_high",
                "Low": "gas_price_low",
                "Open": "gas_price_open",
                "Volume": "gas_volume"
            })
            
            cols_to_keep = ["date", "gas_price_close", "gas_price_high", "gas_price_low"]
            cols = [c for c in cols_to_keep if c in df.columns]
            df = df[cols]
            df["date"] = pd.to_datetime(df["date"])
            
            logger.info(f"Fetched {len(df)} gas price records.")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching gas prices: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = GasClient()
    gas_df = client.fetch_gas_prices("2023-01-01", "2023-01-31")
    print(f"Gas prices shape: {gas_df.shape}")
    print(gas_df.head())
