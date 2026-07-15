import logging
import requests
import pandas as pd
import os
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class NewsClient:
    """
    Client for fetching energy and geopolitical news.
    Uses The Guardian Open Platform API.
    """
    
    BASE_URL = "https://content.guardianapis.com/search"
    
    def __init__(self):
        # Fallback to 'test' key which works for limited endpoints, but a real key is better
        self.api_key = os.getenv("GUARDIAN_API_KEY", "test")
        self.session = requests.Session()

    def fetch_articles(self, start_date: str, end_date: str, query: str = "energy OR electricity OR gas OR russia OR ukraine") -> pd.DataFrame:
        """
        Fetch news article metadata and web publication dates.
        """
        logger.info(f"Fetching news articles from {start_date} to {end_date} with query: {query}")
        
        all_articles = []
        page = 1
        total_pages = 1
        
        while page <= total_pages:
            params = {
                "q": query,
                "from-date": start_date,
                "to-date": end_date,
                "api-key": self.api_key,
                "show-fields": "headline,bodyText",
                "page": page,
                "page-size": 50
            }
            
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json().get("response", {})
                
                total_pages = data.get("pages", 1)
                results = data.get("results", [])
                
                for r in results:
                    all_articles.append({
                        "id": r.get("id"),
                        "date": r.get("webPublicationDate"),
                        "headline": r.get("fields", {}).get("headline"),
                        "body": r.get("fields", {}).get("bodyText", "")[:500] # truncate to save memory
                    })
                    
                page += 1
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching from Guardian API on page {page}: {e}")
                break
                
        if not all_articles:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_articles)
        df["date"] = pd.to_datetime(df["date"])
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = NewsClient()
    # Be careful not to overwhelm the 'test' tier
    news_df = client.fetch_articles("2024-01-15", "2024-01-16", query="energy")
    print(f"News shape: {news_df.shape}")
    print(news_df.head())
