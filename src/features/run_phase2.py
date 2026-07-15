#!/usr/bin/env python
"""
run_phase2.py — Orchestrates and runs the complete Phase 2 pipeline.
1. Executes sentiment_analysis.py (FinBERT NLP pipeline) to process news and build daily_sentiment.
2. Executes build_features.py to construct lags, rolling averages, calendar, grid features, and output model_features.csv.
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase2")

def run_script(script_path):
    logger.info(f"Running: {script_path}")
    # Use the same python executable
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    if res.returncode != 0:
        logger.error(f"Script {script_path} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    logger.info(f"Script {script_path} completed successfully.")

def main():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 2 PIPELINE ORCHESTRATION")
    logger.info("=" * 60)
    
    # 1. Run sentiment analysis script
    run_script("src/features/sentiment_analysis.py")
    
    # 2. Run feature engineering script
    run_script("src/features/build_features.py")
    
    logger.info("=" * 60)
    logger.info("PHASE 2 PIPELINE COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
