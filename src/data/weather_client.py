import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WeatherClient:
    """
    Client for fetching weather and renewable generation data.
    Focuses on Sheffield Solar (PV Live) for solar generation.
    Met Office MIDAS would typically require FTP/CEDA credentials.
    """
    
    PV_LIVE_URL = "https://api0.solar.sheffield.ac.uk/pvlive/api/v4"
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self):
        self.session = requests.Session()

    def fetch_national_solar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch half-hourly national solar generation from Sheffield PV Live.
        pes_id = 0 corresponds to National generation.
        """
        logger.info(f"Fetching national solar generation from {start_date} to {end_date}")
        
        url = f"{self.PV_LIVE_URL}/pes/0"
        
        # PVLive requires format YYYY-MM-DDTHH:MM:SSZ
        start_dt = f"{start_date}T00:00:00Z"
        # Append T23:59:59Z to the end date to get the full day
        end_dt = f"{end_date}T23:59:59Z"
        
        params = {
            "start": start_dt,
            "end": end_dt,
            "data_format": "json"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("data", [])
            if not records:
                return pd.DataFrame()
                
            num_cols = len(records[0])
            if num_cols == 3:
                cols = ["pes_id", "datetime", "generation_mw"]
            elif num_cols == 4:
                cols = ["pes_id", "datetime", "generation_mw", "capacity_mwp"]
            else:
                cols = [f"col_{i}" for i in range(num_cols)]
                
            df = pd.DataFrame(records, columns=cols)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from PV Live: {e}")
            return pd.DataFrame()

    def fetch_historical_weather(self, start_date: str, end_date: str, lat: float = 52.48, lon: float = -1.89) -> pd.DataFrame:
        """
        Fetch historical hourly weather (temperature, wind speed, cloud cover) via Open-Meteo.
        Defaults to Birmingham, UK (lat=52.48, lon=-1.89) as a proxy for national weather.
        """
        logger.info(f"Fetching historical weather for lat={lat}, lon={lon} from {start_date} to {end_date}")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_10m,cloud_cover",
            "timezone": "Europe/London"
        }
        
        try:
            response = self.session.get(self.OPEN_METEO_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            hourly_data = data.get("hourly", {})
            if not hourly_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(hourly_data)
            df["time"] = pd.to_datetime(df["time"])
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Open-Meteo: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WeatherClient()
    solar_df = client.fetch_national_solar("2024-01-15", "2024-01-16")
    print(f"Solar shape: {solar_df.shape}")
    print(solar_df.head())
