#!/usr/bin/env python
"""
fetch_full.py — Parallel data fetcher with DuckDB storage and resume capability.

Fetches 2020-01-01 to 2025-06-23 across all sources:
  - System prices (Elexon)     — parallel day-by-day
  - Generation mix (Elexon)    — parallel 7-day chunks
  - National demand (Elexon)   — parallel 7-day chunks
  - National solar (PV Live)   — monthly sequential
  - Historical weather (Open-Meteo) — single call
  - Gas prices (yfinance TTF)  — single call
  - News articles (Guardian)   — monthly, rate-limited

Data is saved incrementally to DuckDB. Re-running skips completed work.

Usage:
    python src/data/fetch_full.py
"""

import os
import sys
import logging
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# pyrefly: ignore [missing-import]
import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Sibling module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_client import WeatherClient
from gas_client import GasClient
from news_client import NewsClient

# ───────────────────────── Configuration ─────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2025-06-23"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(PROJECT_ROOT, "data", "db")
DB_PATH = os.path.join(DB_DIR, "electricity.duckdb")

ELEXON_WORKERS = 10       # Concurrent requests to Elexon API
ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"
NEWS_DELAY = 1.0          # Seconds between Guardian API page fetches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_full")


# ───────────────────────── Date Helpers ──────────────────────────

def month_ranges(start: str, end: str):
    """Yield (month_start, month_end, 'YYYY-MM') for each calendar month."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s.replace(day=1)
    while cur <= e:
        m_start = max(cur, s)
        nxt = (cur.replace(month=cur.month + 1) if cur.month < 12
               else cur.replace(year=cur.year + 1, month=1))
        m_end = min(nxt - timedelta(days=1), e)
        yield m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"), cur.strftime("%Y-%m")
        cur = nxt


def day_list(start: str, end: str):
    """Return list of 'YYYY-MM-DD' strings for every day in [start, end]."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((e - s).days + 1)]


def chunk_dates(start: str, end: str, days: int = 7):
    """Split [start, end] into sub-ranges of at most *days* days."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    chunks, cur = [], s
    while cur <= e:
        ce = min(cur + timedelta(days=days - 1), e)
        chunks.append((cur.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
        cur = ce + timedelta(days=1)
    return chunks


# ───────────────────────── DuckDB Manager ────────────────────────

class DBManager:
    """Thread-safe DuckDB writer with checkpoint tracking."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.con = duckdb.connect(path)
        self._lock = threading.Lock()
        self._create_schema()

    # ── schema ──
    def _create_schema(self):
        stmts = [
            """CREATE TABLE IF NOT EXISTS system_prices (
                   settlementDate VARCHAR, settlementPeriod INTEGER,
                   systemSellPrice DOUBLE, systemBuyPrice DOUBLE)""",
            """CREATE TABLE IF NOT EXISTS generation_mix (
                   dataset VARCHAR, publishTime VARCHAR, startTime VARCHAR,
                   settlementDate VARCHAR, settlementPeriod INTEGER,
                   fuelType VARCHAR, generation BIGINT)""",
            """CREATE TABLE IF NOT EXISTS national_demand (
                   settlementDate VARCHAR, settlementPeriod INTEGER,
                   startTime VARCHAR, initialDemandOutturn INTEGER,
                   initialTransmissionSystemDemandOutturn INTEGER)""",
            """CREATE TABLE IF NOT EXISTS national_solar (
                   pes_id INTEGER, datetime VARCHAR, generation_mw DOUBLE)""",
            """CREATE TABLE IF NOT EXISTS historical_weather (
                   time VARCHAR, temperature_2m DOUBLE,
                   wind_speed_10m DOUBLE, cloud_cover DOUBLE)""",
            """CREATE TABLE IF NOT EXISTS gas_prices (
                   date VARCHAR, gas_price_close DOUBLE,
                   gas_price_high DOUBLE, gas_price_low DOUBLE)""",
            """CREATE TABLE IF NOT EXISTS news_articles (
                   id VARCHAR, date VARCHAR, headline VARCHAR, body VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS _checkpoints (
                   source VARCHAR, year_month VARCHAR,
                   rows_fetched INTEGER, completed_at TIMESTAMP)""",
        ]
        for s in stmts:
            self.con.execute(s)

    # ── checkpoint helpers ──
    def is_done(self, source: str, ym: str) -> bool:
        r = self.con.execute(
            "SELECT 1 FROM _checkpoints WHERE source=? AND year_month=?",
            [source, ym],
        ).fetchone()
        return r is not None

    def save(self, table: str, df: pd.DataFrame, source: str, ym: str):
        with self._lock:
            if not df.empty:
                self.con.execute(f"INSERT INTO {table} SELECT * FROM df")
            # upsert checkpoint
            self.con.execute(
                "DELETE FROM _checkpoints WHERE source=? AND year_month=?",
                [source, ym],
            )
            self.con.execute(
                "INSERT INTO _checkpoints VALUES (?,?,?,CURRENT_TIMESTAMP)",
                [source, ym, len(df)],
            )

    def count(self, table: str) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def close(self):
        self.con.close()


# ───────────────────────── Elexon Helpers ────────────────────────

def _session():
    """Fresh requests.Session with retry (one per thread)."""
    s = requests.Session()
    r = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"], backoff_factor=0.5)
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


def _get_prices_day(date_str: str):
    """Fetch system prices for a single settlement date."""
    try:
        resp = _session().get(
            f"{ELEXON_BASE}/balancing/settlement/system-prices/{date_str}",
            params={"format": "json"}, timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if rows:
            df = pd.DataFrame(rows)
            keep = [c for c in ["settlementDate", "settlementPeriod",
                                "systemSellPrice", "systemBuyPrice"]
                    if c in df.columns]
            return df[keep]
    except Exception:
        pass
    return None


def _get_elexon_chunk(endpoint: str, cs: str, ce: str, keep_cols=None):
    """Fetch a 7-day chunk from an Elexon dataset endpoint."""
    try:
        resp = _session().get(
            f"{ELEXON_BASE}/{endpoint}",
            params={"settlementDateFrom": cs, "settlementDateTo": ce,
                    "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if rows:
            df = pd.DataFrame(rows)
            if keep_cols:
                df = df[[c for c in keep_cols if c in df.columns]]
            return df
    except Exception:
        pass
    return None


# ───────────────────────── Source Fetchers ────────────────────────

def fetch_system_prices(db: DBManager):
    logger.info("=" * 60)
    logger.info("SYSTEM PRICES  (parallel, %d workers)", ELEXON_WORKERS)
    t0 = time.time()

    for ms, me, ym in month_ranges(START_DATE, END_DATE):
        if db.is_done("system_prices", ym):
            logger.info("  [skip] %s", ym)
            continue

        dates = day_list(ms, me)
        parts = []
        with ThreadPoolExecutor(max_workers=ELEXON_WORKERS) as pool:
            futs = {pool.submit(_get_prices_day, d): d for d in dates}
            for f in as_completed(futs):
                r = f.result()
                if r is not None:
                    parts.append(r)

        mdf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        db.save("system_prices", mdf, "system_prices", ym)
        logger.info("  [done] %s  %d rows", ym, len(mdf))

    logger.info("  TOTAL %s rows  (%.1fs)", f"{db.count('system_prices'):,}", time.time() - t0)


def fetch_generation_mix(db: DBManager):
    logger.info("=" * 60)
    logger.info("GENERATION MIX  (parallel, %d workers)", ELEXON_WORKERS)
    t0 = time.time()

    for ms, me, ym in month_ranges(START_DATE, END_DATE):
        if db.is_done("generation_mix", ym):
            logger.info("  [skip] %s", ym)
            continue

        chunks = chunk_dates(ms, me, days=7)
        parts = []
        with ThreadPoolExecutor(max_workers=ELEXON_WORKERS) as pool:
            futs = {pool.submit(_get_elexon_chunk, "datasets/FUELHH", cs, ce): (cs, ce)
                    for cs, ce in chunks}
            for f in as_completed(futs):
                r = f.result()
                if r is not None:
                    parts.append(r)

        mdf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        db.save("generation_mix", mdf, "generation_mix", ym)
        logger.info("  [done] %s  %d rows", ym, len(mdf))

    logger.info("  TOTAL %s rows  (%.1fs)", f"{db.count('generation_mix'):,}", time.time() - t0)


def fetch_demand(db: DBManager):
    logger.info("=" * 60)
    logger.info("NATIONAL DEMAND  (parallel, %d workers)", ELEXON_WORKERS)
    t0 = time.time()

    keep = ["settlementDate", "settlementPeriod", "startTime",
            "initialDemandOutturn", "initialTransmissionSystemDemandOutturn"]

    for ms, me, ym in month_ranges(START_DATE, END_DATE):
        if db.is_done("national_demand", ym):
            logger.info("  [skip] %s", ym)
            continue

        chunks = chunk_dates(ms, me, days=7)
        parts = []
        with ThreadPoolExecutor(max_workers=ELEXON_WORKERS) as pool:
            futs = {pool.submit(_get_elexon_chunk, "demand/outturn", cs, ce, keep): (cs, ce)
                    for cs, ce in chunks}
            for f in as_completed(futs):
                r = f.result()
                if r is not None:
                    parts.append(r)

        mdf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        db.save("national_demand", mdf, "national_demand", ym)
        logger.info("  [done] %s  %d rows", ym, len(mdf))

    logger.info("  TOTAL %s rows  (%.1fs)", f"{db.count('national_demand'):,}", time.time() - t0)


def fetch_weather(db: DBManager):
    logger.info("=" * 60)
    logger.info("HISTORICAL WEATHER  (single call)")

    if db.is_done("historical_weather", "all"):
        logger.info("  [skip] already fetched")
        return

    df = WeatherClient().fetch_historical_weather(START_DATE, END_DATE)
    if not df.empty:
        cols = [c for c in ["time", "temperature_2m", "wind_speed_10m", "cloud_cover"]
                if c in df.columns]
        df = df[cols]
        df["cloud_cover"] = df["cloud_cover"].astype(float)

    db.save("historical_weather", df, "historical_weather", "all")
    logger.info("  [done] %d rows", len(df))


def fetch_gas(db: DBManager):
    logger.info("=" * 60)
    logger.info("GAS PRICES  (single yfinance call)")

    if db.is_done("gas_prices", "all"):
        logger.info("  [skip] already fetched")
        return

    df = GasClient().fetch_gas_prices(START_DATE, END_DATE)
    if not df.empty:
        df["date"] = df["date"].astype(str)

    db.save("gas_prices", df, "gas_prices", "all")
    logger.info("  [done] %d rows", len(df))


def fetch_solar(db: DBManager):
    logger.info("=" * 60)
    logger.info("NATIONAL SOLAR  (monthly chunks)")
    t0 = time.time()
    client = WeatherClient()

    for ms, me, ym in month_ranges(START_DATE, END_DATE):
        if db.is_done("national_solar", ym):
            logger.info("  [skip] %s", ym)
            continue

        try:
            df = client.fetch_national_solar(ms, me)
            if not df.empty:
                cols = [c for c in ["pes_id", "datetime", "generation_mw"]
                        if c in df.columns]
                df = df[cols]
        except Exception as exc:
            logger.warning("  [fail] %s: %s", ym, exc)
            df = pd.DataFrame()

        db.save("national_solar", df, "national_solar", ym)
        logger.info("  [done] %s  %d rows", ym, len(df))
        time.sleep(0.3)

    logger.info("  TOTAL %s rows  (%.1fs)", f"{db.count('national_solar'):,}", time.time() - t0)


def fetch_news(db: DBManager):
    logger.info("=" * 60)
    logger.info("NEWS ARTICLES  (monthly, rate-limited)")
    t0 = time.time()
    client = NewsClient()

    for ms, me, ym in month_ranges(START_DATE, END_DATE):
        if db.is_done("news_articles", ym):
            logger.info("  [skip] %s", ym)
            continue

        try:
            df = client.fetch_articles(ms, me)
            if not df.empty:
                df["date"] = df["date"].astype(str)
                cols = [c for c in ["id", "date", "headline", "body"] if c in df.columns]
                df = df[cols]
        except Exception as exc:
            logger.warning("  [fail] %s: %s", ym, exc)
            df = pd.DataFrame()

        db.save("news_articles", df, "news_articles", ym)
        logger.info("  [done] %s  %d articles", ym, len(df))
        time.sleep(NEWS_DELAY)

    logger.info("  TOTAL %s articles  (%.1fs)", f"{db.count('news_articles'):,}", time.time() - t0)


# ───────────────────────── Main ──────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("FULL DATA FETCH  %s -> %s", START_DATE, END_DATE)
    logger.info("Database: %s", DB_PATH)
    logger.info("=" * 60)
    wall = time.time()

    db = DBManager(DB_PATH)

    try:
        # Quick sources first
        fetch_weather(db)
        fetch_gas(db)

        # Parallel Elexon sources (bulk of the work)
        fetch_system_prices(db)
        fetch_generation_mix(db)
        fetch_demand(db)

        # Monthly sources
        fetch_solar(db)
        fetch_news(db)

        # ── Final summary ──
        elapsed = time.time() - wall
        logger.info("=" * 60)
        logger.info("COMPLETE  (%.1f min)", elapsed / 60)
        logger.info("-" * 60)
        for tbl in ["system_prices", "generation_mix", "national_demand",
                     "national_solar", "historical_weather", "gas_prices",
                     "news_articles"]:
            logger.info("  %-30s %10s rows", tbl, f"{db.count(tbl):,}")
        logger.info("=" * 60)

    finally:
        db.close()


if __name__ == "__main__":
    main()
