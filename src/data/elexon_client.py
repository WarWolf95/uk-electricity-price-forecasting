import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ElexonClient:
    """
    Robust client for fetching data from the Elexon BMRS API.
    Handles retries, rate limits, and date pagination.
    """
    
    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
    
    def __init__(self, retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=backoff_factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Helper method to make GET requests and extract 'data' payload."""
        url = f"{self.BASE_URL}/{endpoint}"
        if params is None:
            params = {}
        params['format'] = 'json'
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from {endpoint}: {e}")
            return None

    def _date_range_chunks(self, start_date: str, end_date: str, chunk_days: int = 7) -> List[tuple]:
        """Splits a large date range into smaller chunks to avoid overwhelming the API."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        chunks = []
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            current = chunk_end + timedelta(days=1)
            
        return chunks

    def fetch_system_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch half-hourly system prices (target variable).
        The endpoint `/balancing/settlement/system-prices` accepts settlementDate.
        We iterate through dates since it seems to only take a single date or relies on pagination if long.
        """
        logger.info(f"Fetching system prices from {start_date} to {end_date}")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        
        # Note: Depending on the API limits, we might need a delay here if requests are rapid.
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            # Endpoint accepts a specific date in the URL path
            endpoint = f"balancing/settlement/system-prices/{date_str}"
            params = {}
            
            data = self._get(endpoint, params)
            if data:
                all_data.extend(data)
                
            current += timedelta(days=1)
            
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        # Select important columns
        cols_to_keep = ["settlementDate", "settlementPeriod", "systemSellPrice", "systemBuyPrice"]
        # Ensure columns exist before filtering
        cols = [c for c in cols_to_keep if c in df.columns]
        return df[cols]

    def fetch_generation_mix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch generation mix by fuel type."""
        logger.info(f"Fetching generation mix from {start_date} to {end_date}")
        all_data = []
        
        # API often limits date ranges, chunk by 7 days
        for chunk_start, chunk_end in self._date_range_chunks(start_date, end_date, chunk_days=7):
            params = {
                "settlementDateFrom": chunk_start,
                "settlementDateTo": chunk_end
            }
            data = self._get("datasets/FUELHH", params)
            if data:
                all_data.extend(data)
                
        return pd.DataFrame(all_data)

    def fetch_demand(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical half-hourly national demand outturn.
        Uses /demand/outturn (not /datasets/INDO which is real-time only).
        Returns initialDemandOutturn and initialTransmissionSystemDemandOutturn.
        """
        logger.info(f"Fetching demand outturn from {start_date} to {end_date}")
        all_data = []
        
        # The outturn endpoint supports ~7-day windows max
        for chunk_start, chunk_end in self._date_range_chunks(start_date, end_date, chunk_days=7):
            params = {
                "settlementDateFrom": chunk_start,
                "settlementDateTo": chunk_end
            }
            data = self._get("demand/outturn", params)
            if data:
                all_data.extend(data)
                
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        cols_to_keep = ["settlementDate", "settlementPeriod", "startTime",
                        "initialDemandOutturn", "initialTransmissionSystemDemandOutturn"]
        cols = [c for c in cols_to_keep if c in df.columns]
        return df[cols]



if __name__ == "__main__":
    # Simple test to verify the client works
    client = ElexonClient()
    
    # Test for 2 days
    prices_df = client.fetch_system_prices("2024-01-15", "2024-01-16")
    print(f"Prices shape: {prices_df.shape}")
    print(prices_df.head())
    
    gen_df = client.fetch_generation_mix("2024-01-15", "2024-01-16")
    print(f"\nGeneration shape: {gen_df.shape}")
    print(gen_df.head())
