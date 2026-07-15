import os
import logging
from datetime import datetime
from elexon_client import ElexonClient
from weather_client import WeatherClient
from gas_client import GasClient
from news_client import NewsClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Orchestrator script to fetch all data for Phase 1.
    """
    # Define date range for historical fetch
    START_DATE = "2023-01-01"
    END_DATE = "2023-01-31"  # Let's keep it to 1 month initially for testing
    
    # Setup data directories
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)
    
    logger.info(f"Starting data fetch from {START_DATE} to {END_DATE}")
    
    # 1. Elexon Data
    elexon = ElexonClient()
    
    prices = elexon.fetch_system_prices(START_DATE, END_DATE)
    if not prices.empty:
        prices.to_csv(os.path.join(RAW_DIR, "system_prices.csv"), index=False)
        logger.info(f"Saved {len(prices)} system price records.")
        
    gen_mix = elexon.fetch_generation_mix(START_DATE, END_DATE)
    if not gen_mix.empty:
        gen_mix.to_csv(os.path.join(RAW_DIR, "generation_mix.csv"), index=False)
        logger.info(f"Saved {len(gen_mix)} generation mix records.")
        
    demand = elexon.fetch_demand(START_DATE, END_DATE)
    if not demand.empty:
        demand.to_csv(os.path.join(RAW_DIR, "national_demand.csv"), index=False)
        logger.info(f"Saved {len(demand)} demand records.")
        

        
    # 2. Weather Data (Solar & Open-Meteo)
    weather = WeatherClient()
    solar = weather.fetch_national_solar(START_DATE, END_DATE)
    if not solar.empty:
        solar.to_csv(os.path.join(RAW_DIR, "national_solar.csv"), index=False)
        logger.info(f"Saved {len(solar)} solar records.")
        
    hist_weather = weather.fetch_historical_weather(START_DATE, END_DATE)
    if not hist_weather.empty:
        hist_weather.to_csv(os.path.join(RAW_DIR, "historical_weather.csv"), index=False)
        logger.info(f"Saved {len(hist_weather)} historical weather records.")
        
    # 3. Gas Data
    gas = GasClient()
    gas_prices = gas.fetch_gas_prices(START_DATE, END_DATE)
    if not gas_prices.empty:
        gas_prices.to_csv(os.path.join(RAW_DIR, "gas_prices.csv"), index=False)
        logger.info(f"Saved {len(gas_prices)} gas price records.")
        
    # 4. News Data
    news = NewsClient()
    articles = news.fetch_articles(START_DATE, END_DATE, query="energy OR gas OR russia OR ukraine")
    if not articles.empty:
        articles.to_csv(os.path.join(RAW_DIR, "news_articles.csv"), index=False)
        logger.info(f"Saved {len(articles)} news articles.")
        
    logger.info("Data fetching complete. All data saved to data/raw/")

if __name__ == "__main__":
    main()
