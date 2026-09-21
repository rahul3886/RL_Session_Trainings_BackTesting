"""
Generate synthetic XAUUSD historical data for testing the backtester.
This version generates structured data containing explicit SMC geometry
(trending/ranging days, order blocks, FVGs, equal levels) so that
SMC detectors return meaningful, non-zero vectors.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_structured_csv(
    instrument="XAUUSD", timeframe="M15", start="2022-01-01", end="2025-01-05", out_dir="data/raw"
):
    os.makedirs(out_dir, exist_ok=True)
    freq_min = int(timeframe[1:])
    freq = f"{freq_min}min"
    
    logger.info(f"Generating structured {instrument} {timeframe} from {start} to {end}...")
    dates = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    dates = dates[dates.dayofweek < 5]  # Mon-Fri only
    
    n = len(dates)
    
    # Pre-allocate arrays
    open_p = np.zeros(n)
    high_p = np.zeros(n)
    low_p = np.zeros(n)
    close_p = np.zeros(n)
    volume = np.zeros(n)
    
    current_price = 1800.0
    atr_base = 1.5 if freq_min == 15 else 0.8
    
    # State tracking
    regime = 0  # 0=ranging, 1=bull trending, -1=bear trending
    days_in_regime = 0
    current_day = dates[0].date()
    
    for i in range(n):
        dt = dates[i]
        
        # Change regime every 3-8 days
        if dt.date() != current_day:
            current_day = dt.date()
            days_in_regime += 1
            if days_in_regime > np.random.randint(3, 8):
                regime = np.random.choice([0, 1, -1])
                days_in_regime = 0
                
        # Session logic
        hour = dt.hour
        is_london = 8 <= hour < 17
        is_ny = 12 <= hour < 21
        
        # Base volatility
        vol = atr_base * (1.5 if is_london or is_ny else 0.5)
        
        # Drift based on regime and session
        drift = 0.0
        if is_london:
            drift = regime * (atr_base * 0.2)
        elif is_ny:
            # NY sometimes sweeps London
            if hour == 12 and np.random.random() < 0.3:
                drift = -regime * (atr_base * 3.0)  # Sweep opposite
            else:
                drift = regime * (atr_base * 0.3)
                
        # Occasional Order Block / FVG (large displacement candle)
        is_displacement = False
        if (is_london or is_ny) and np.random.random() < 0.05:
            drift = np.sign(drift if drift != 0 else np.random.randn()) * (atr_base * 4.0)
            is_displacement = True
            
        # Generate candle
        o = current_price
        c = o + drift + np.random.normal(0, vol * 0.5)
        
        if is_displacement:
            # Huge body, small wicks
            h = max(o, c) + np.abs(np.random.normal(0, vol * 0.2))
            l = min(o, c) - np.abs(np.random.normal(0, vol * 0.2))
        else:
            # Normal wicks
            h = max(o, c) + np.abs(np.random.normal(0, vol))
            l = min(o, c) - np.abs(np.random.normal(0, vol))
            
        # Occasional equal levels (force high/low to be same as previous)
        if i > 0 and not is_displacement and np.random.random() < 0.1:
            if np.random.random() < 0.5:
                # Equal highs
                h = high_p[i-1] + np.random.normal(0, 0.05)
                if c > h: c = h - 0.1
                if o > h: o = h - 0.1
            else:
                # Equal lows
                l = low_p[i-1] - np.random.normal(0, 0.05)
                if c < l: c = l + 0.1
                if o < l: o = l + 0.1
                
        open_p[i] = o
        high_p[i] = h
        low_p[i] = l
        close_p[i] = c
        volume[i] = int(np.abs(np.random.normal(1000 if is_london or is_ny else 300, 200)))
        
        current_price = c
        
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.round(open_p, 3),
        "high": np.round(high_p, 3),
        "low": np.round(low_p, 3),
        "close": np.round(close_p, 3),
        "volume": volume
    })
    
    out_path = os.path.join(out_dir, f"{instrument}_{timeframe}_{start}_{end}_mt5.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} structured bars to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_structured_csv("XAUUSD", "M15", "2022-01-01", "2025-01-05")
    generate_structured_csv("XAUUSD", "M5",  "2022-01-01", "2025-01-05")
