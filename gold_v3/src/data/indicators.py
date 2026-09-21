"""
Technical indicator computation via pandas-ta.
ATR, RSI, Momentum, Volume SMA, candle anatomy.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame, settings=None) -> pd.DataFrame:
    """
    Add all technical indicators to OHLCV DataFrame.
    Uses pandas-ta for robust calculation.
    """
    try:
        import pandas_ta as ta
    except ImportError:
        logger.warning("pandas-ta not installed, computing indicators manually")
        return _add_indicators_manual(df, settings)

    # Default periods from settings or hardcoded
    atr_period = 14
    rsi_period = 14
    mom_period = 8
    vol_sma_period = 20

    if settings is not None:
        atr_period = settings.indicators.atr_period
        rsi_period = settings.indicators.rsi_period
        mom_period = settings.indicators.momentum_period
        vol_sma_period = settings.indicators.volume_sma_period

    df = df.copy()

    # ATR
    atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    if atr is not None:
        df["atr"] = atr
    else:
        df["atr"] = _manual_atr(df, atr_period)

    # RSI
    rsi = ta.rsi(df["close"], length=rsi_period)
    if rsi is not None:
        df["rsi"] = rsi
    else:
        df["rsi"] = 50.0

    # Momentum
    mom = ta.mom(df["close"], length=mom_period)
    if mom is not None:
        df["momentum"] = mom
    else:
        df["momentum"] = df["close"].diff(mom_period)

    # Volume SMA
    if "volume" in df.columns:
        vol_sma = df["volume"].rolling(vol_sma_period).mean()
        df["volume_sma"] = vol_sma
        df["volume_ratio"] = np.where(
            vol_sma > 0,
            df["volume"] / vol_sma,
            1.0
        )
    else:
        df["volume_sma"] = 1.0
        df["volume_ratio"] = 1.0

    # Candle anatomy
    df["candle_body"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # ATR-normalised candle features
    atr_safe = df["atr"].replace(0, np.nan).ffill().fillna(1.0)
    df["candle_body_atr"] = df["candle_body"] / atr_safe
    df["upper_wick_atr"] = df["upper_wick"] / atr_safe
    df["lower_wick_atr"] = df["lower_wick"] / atr_safe

    # --- Institutional Quant Filters ---
    # 1. VWAP & Z-Score
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    v_tp = df["volume"] * typical_price
    df["vwap"] = v_tp.cumsum() / df["volume"].cumsum()
    # Rolling Z-Score relative to VWAP bands
    df["vwap_std"] = typical_price.rolling(vol_sma_period).std()
    df["vwap_zscore"] = (typical_price - df["vwap"]) / df["vwap_std"].replace(0, np.nan).ffill()

    # 2. Momentum R2 (Pearson Correlation squared over 15 periods)
    # Vectorized Pearson Correlation
    n = 15
    y = df["close"]
    x = pd.Series(np.arange(len(df)), index=df.index)
    
    def rolling_r2(y_win):
        if len(y_win) < n: return 0.0
        x_win = np.arange(n)
        correlation_matrix = np.corrcoef(x_win, y_win)
        correlation_xy = correlation_matrix[0, 1]
        return correlation_xy**2

    # Using rolling correlation for performance
    df["momentum_r2"] = df["close"].rolling(window=n).corr(pd.Series(np.arange(len(df)), index=df.index)) ** 2
    
    # 3. Volatility Compression (ATR Ratio)
    atr_long = df["atr"].rolling(50).mean()
    df["volatility_compression"] = df["atr"] / atr_long.replace(0, np.nan).ffill()

    # 4. Displacement Score (Force)
    # Normalized displacement over last 3 candles: (price_change) / (avg_atr * window)
    df["displacement"] = (df["close"] - df["close"].shift(3)) / (atr_safe * 3.0)

    # --- Phase 9: HMM & Regime Features ---
    # 5. Elder's Force Index (EFI)
    raw_efi = (df["close"] - df["close"].shift(1)) * df["volume"].fillna(0)
    df["efi"] = raw_efi.ewm(span=13, adjust=False).mean()

    # 6. Rolling Z-Score Scaling for HMM Inputs (Prevents Lookahead Bias)
    # Scale Returns
    ret = df["close"].pct_change()
    df["hmm_ret_z"] = (ret - ret.rolling(200).mean()) / ret.rolling(200).std().replace(0, np.nan).ffill()
    # Scale ATR
    df["hmm_atr_z"] = (df["atr"] - df["atr"].rolling(200).mean()) / df["atr"].rolling(200).std().replace(0, np.nan).ffill()
    # Scale EFI
    df["hmm_efi_z"] = (df["efi"] - df["efi"].rolling(200).mean()) / df["efi"].rolling(200).std().replace(0, np.nan).ffill()

    # Forward fill NaN from indicator warmup
    df = df.ffill().bfill()

    logger.info(f"  Added indicators: ATR({atr_period}), RSI({rsi_period}), "
                f"MOM({mom_period}), VolSMA({vol_sma_period}), "
                f"VWAP-Z, R2, Compression, Displacement")

    return df


def _manual_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Manual ATR calculation as fallback."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def _add_indicators_manual(df: pd.DataFrame, settings=None) -> pd.DataFrame:
    """Fallback manual indicator calculation without pandas-ta."""
    atr_period = 14 if settings is None else settings.indicators.atr_period
    rsi_period = 14 if settings is None else settings.indicators.rsi_period
    mom_period = 8 if settings is None else settings.indicators.momentum_period
    vol_sma_period = 20 if settings is None else settings.indicators.volume_sma_period

    df = df.copy()

    # ATR
    df["atr"] = _manual_atr(df, atr_period)

    # RSI (Wilder's smoothing)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Momentum
    df["momentum"] = df["close"].diff(mom_period)

    # Volume
    if "volume" in df.columns:
        df["volume_sma"] = df["volume"].rolling(vol_sma_period).mean()
        df["volume_ratio"] = np.where(
            df["volume_sma"] > 0, df["volume"] / df["volume_sma"], 1.0
        )
    else:
        df["volume_sma"] = 1.0
        df["volume_ratio"] = 1.0

    # Candle anatomy
    df["candle_body"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    atr_safe = df["atr"].replace(0, np.nan).ffill().fillna(1.0)
    df["candle_body_atr"] = df["candle_body"] / atr_safe
    df["upper_wick_atr"] = df["upper_wick"] / atr_safe
    df["lower_wick_atr"] = df["lower_wick"] / atr_safe

    # --- Phase 8 Institutional Quant Filters ---
    # 1. VWAP & Z-Score
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    v_tp = df["volume"] * typical_price
    df["vwap"] = v_tp.cumsum() / df["volume"].cumsum()
    df["vwap_std"] = typical_price.rolling(vol_sma_period).std()
    df["vwap_zscore"] = (typical_price - df["vwap"]) / df["vwap_std"].replace(0, np.nan).ffill()

    # 2. Momentum R2
    df["momentum_r2"] = df["close"].rolling(window=15).corr(pd.Series(np.arange(len(df)), index=df.index)) ** 2
    
    # 3. Volatility Compression
    atr_long = df["atr"].rolling(50).mean()
    df["volatility_compression"] = df["atr"] / atr_long.replace(0, np.nan).ffill()

    # 4. Displacement
    df["displacement"] = (df["close"] - df["close"].shift(3)) / (atr_safe * 3.0)

    # --- Phase 9: HMM & Regime Features ---
    # 5. Elder's Force Index (EFI)
    if "volume" in df.columns:
        raw_efi = (df["close"] - df["close"].shift(1)) * df["volume"].fillna(0)
    else:
        raw_efi = (df["close"] - df["close"].shift(1))
    df["efi"] = raw_efi.ewm(span=13, adjust=False).mean()

    # 6. Rolling Z-Score Scaling for HMM Inputs
    ret = df["close"].pct_change()
    df["hmm_ret_z"] = (ret - ret.rolling(200).mean()) / ret.rolling(200).std().replace(0, np.nan).ffill()
    df["hmm_atr_z"] = (df["atr"] - df["atr"].rolling(200).mean()) / df["atr"].rolling(200).std().replace(0, np.nan).ffill()
    df["hmm_efi_z"] = (df["efi"] - df["efi"].rolling(200).mean()) / df["efi"].rolling(200).std().replace(0, np.nan).ffill()

    df = df.ffill().bfill()
    return df
