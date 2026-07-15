#!/usr/bin/env python
"""
sentiment_analysis.py — Runs ProsusAI/finbert sentiment classification on UK news articles.
Saves article-level and daily-aggregated sentiment scores to DuckDB.
Uses connection pooling (connect/disconnect) to avoid locking the database.
Supports incremental resuming.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import logging
# pyrefly: ignore [missing-import]
import duckdb
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentiment_analysis")

DB_PATH = "data/db/electricity.duckdb"
BATCH_SIZE = 64
CHUNK_SIZE = 2000  # Number of rows to pull from DB at once

def setup_db():
    """Create tables if they do not exist."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_sentiment (
                id VARCHAR PRIMARY KEY,
                sentiment_score DOUBLE,
                prob_positive DOUBLE,
                prob_negative DOUBLE,
                prob_neutral DOUBLE,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    finally:
        conn.close()

def get_unprocessed_articles():
    """Fetch all unprocessed articles and then release the lock."""
    conn = duckdb.connect(DB_PATH)
    try:
        df = conn.execute(f"""
            SELECT n.id, n.headline, n.body 
            FROM news_articles n
            LEFT JOIN article_sentiment s ON n.id = s.id
            WHERE s.id IS NULL
        """).fetch_df()
        return df
    finally:
        conn.close()

def save_sentiment_results(results):
    """Save batch of results and release the lock."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.executemany("""
            INSERT OR IGNORE INTO article_sentiment (id, sentiment_score, prob_positive, prob_negative, prob_neutral)
            VALUES (?, ?, ?, ?, ?)
        """, results)
    finally:
        conn.close()

def process_sentiment_batch(nlp, batch_texts):
    """Run pipeline on a batch of texts and compute scores."""
    results = nlp(batch_texts)
    scores = []
    for res in results:
        prob_map = {item['label']: item['score'] for item in res}
        pos = prob_map.get('positive', 0.0)
        neg = prob_map.get('negative', 0.0)
        neu = prob_map.get('neutral', 0.0)
        # Compute continuous score: pos - neg (ranges from -1 to +1)
        score = pos - neg
        scores.append((score, pos, neg, neu))
    return scores

def main():
    logger.info("Initializing FinBERT Sentiment Analysis Pipeline...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"DuckDB database not found at {DB_PATH}!")
        return

    setup_db()
    
    # Check total articles and already processed
    conn = duckdb.connect(DB_PATH)
    total_articles = conn.execute("SELECT count(*) FROM news_articles").fetchone()[0]
    processed_articles = conn.execute("SELECT count(*) FROM article_sentiment").fetchone()[0]
    conn.close()
    
    logger.info(f"Total news articles in DB: {total_articles:,}")
    logger.info(f"Already processed articles: {processed_articles:,}")
    
    if processed_articles >= total_articles:
        logger.info("All articles already processed. Skipping classification.")
    else:
        # Load Model and Tokenizer
        logger.info("Loading ProsusAI/finbert model & tokenizer...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None, batch_size=BATCH_SIZE)
        logger.info(f"Loaded model in {time.time() - t0:.2f}s")
        
        # Load all unprocessed articles into memory and release connection
        logger.info("Reading unprocessed articles from database...")
        df_unprocessed = get_unprocessed_articles()
        remaining = len(df_unprocessed)
        logger.info(f"Loaded {remaining:,} unprocessed articles into memory. DB connection closed.")
        
        start_time = time.time()
        processed_count = 0
        chunk_results = []
        
        # Process in chunks of CHUNK_SIZE
        for chunk_idx in range(0, remaining, CHUNK_SIZE):
            chunk_df = df_unprocessed.iloc[chunk_idx:chunk_idx + CHUNK_SIZE]
            logger.info(f"Processing chunk {chunk_idx // CHUNK_SIZE + 1} ({len(chunk_df)} articles)...")
            
            # Headlines-only classification
            texts = []
            for _, row in chunk_df.iterrows():
                headline = row['headline'] or ""
                text = headline.strip()
                if not text:
                    text = "neutral"
                texts.append(text)
            
            # Run inference in batches
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_scores = process_sentiment_batch(nlp, batch_texts)
                for j, score_tuple in enumerate(batch_scores):
                    art_id = chunk_df.iloc[i + j]['id']
                    chunk_results.append((art_id, *score_tuple))
            
            # Write chunk results to DB periodically
            save_sentiment_results(chunk_results)
            processed_count += len(chunk_results)
            chunk_results = []  # Clear memory
            
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta_min = (remaining - processed_count) / rate / 60 if rate > 0 else 0
            
            logger.info(f"Saved {processed_count}/{remaining} (Rate: {rate:.1f} articles/s, ETA: {eta_min:.1f} min)")
    
    # 3. Create Daily Aggregation
    logger.info("Aggregating sentiment scores to daily metrics...")
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute("DROP TABLE IF EXISTS daily_sentiment")
        conn.execute("""
            CREATE TABLE daily_sentiment AS
            SELECT 
                strftime(strptime(substring(n.date, 1, 10), '%Y-%m-%d'), '%Y-%m-%d') AS date,
                mean(a.sentiment_score) AS sentiment_score,
                count(a.sentiment_score) AS article_count
            FROM article_sentiment a
            JOIN news_articles n ON a.id = n.id
            GROUP BY 1
            ORDER BY 1
        """)
        
        daily_count = conn.execute("SELECT count(*) FROM daily_sentiment").fetchone()[0]
        logger.info(f"Daily sentiment aggregated. Total days: {daily_count:,}")
        
        sample = conn.execute("SELECT * FROM daily_sentiment LIMIT 5").fetch_df()
        logger.info("Daily sentiment sample:\n" + str(sample))
    finally:
        conn.close()
        
    logger.info("Sentiment pipeline complete.")

if __name__ == "__main__":
    main()
