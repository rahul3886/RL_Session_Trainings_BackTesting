import pytest
import pandas as pd
import numpy as np
from src.session.geometry import build_smc_geometry, SmcGeometry

class MockSettings:
    class smc:
        ob_atr_displacement = 1.5
        ob_lookback = 20
        ob_max_age = 50
        fvg_min_size_lr = 0.05
        fvg_max_age = 30
        equal_tolerance_lr = 0.05
        equal_min_touches = 2
        sweep_min_lr = 0.12
        swing_lookback = 5
        choch_lookback = 5
        mss_lookback = 5

def test_geometry_distances_are_lr_normalised():
    """Test that SmcGeometry computes bounds based on London Range (LR) instead of ATR."""
    
    # Create simple mock 20pt London Range
    # High 1020, Low 1000
    dates = pd.date_range("2024-01-01 08:00", periods=5, freq="15min", tz="UTC")
    london_df = pd.DataFrame({
        "open": [1005, 1010, 1015, 1020, 1015],
        "high": [1010, 1015, 1020, 1020, 1018],
        "low": [1000, 1005, 1010, 1015, 1010],
        "close": [1010, 1015, 1020, 1015, 1012],
        "volume": [100]*5
    }, index=dates)
    
    # Calculate synthetic properties usually added by ingest
    london_df["candle_body_atr"] = 1.0
    london_df["upper_wick_atr"] = 0.5
    london_df["lower_wick_atr"] = 0.5

    ny_open = pd.Series({
        "open": 1012, "high": 1015, "low": 1010, "close": 1010
    })

    full_df = london_df.copy()
    atr_series = pd.Series([2.0]*5, index=dates)

    geom = build_smc_geometry(
        london_candles=london_df,
        ny_open_candle=ny_open,
        full_df=full_df,
        smc_config=MockSettings.smc,
        atr_series=atr_series
    )

    # LR = High(1020) - Low(1000) = 20.0
    assert geom.london_range == 20.0
    
    # Normalizations shouldn't explode above 5.0
    assert geom.equal_highs_dist_lr <= 5.0
    assert geom.equal_lows_dist_lr <= 5.0
    
    # Check manual normalize functions
    # 1015 should be 5 pts from mid (1010) => 5/20 = 0.25
    assert geom.normalise(1015) == 0.25
    
    # Check that a level 5 pts away returns 0.25 distance
    assert geom.dist_lr(1005, 1000) == 0.25
