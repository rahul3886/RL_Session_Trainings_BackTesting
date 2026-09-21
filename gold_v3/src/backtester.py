"""
Two-lens backtester — replays historical data through the full
Lens 1 → Lens 2 pipeline and generates performance reports.
"""

import os
import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from pathlib import Path

from src.session.profiler import get_trading_days, get_session_pair, get_london_until_anchor, get_ny_after_anchor
from src.session.features import extract_session_features, build_lens3_state_vector
from src.data.indicators import add_indicators

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    trading_days: list,
    lens1_model,
    lens2_predictor,
    lens2_scaler,
    lens3_model,
    df_m1: Optional[pd.DataFrame] = None,
    df_m5: Optional[pd.DataFrame] = None,
    settings=None,
    news_dates: set = None,
    output_dir: str = "output/backtest",
) -> Dict:
    """
    Run full two-lens backtest on given trading days.

    For each day:
      1. Run Lens 1 on London candles → encoding
      2. Run Lens 2 Predictor → High Probability Scenario
      3. Step Lens 3 through NY candles with encoding + scenario context
      4. Record trades and outcomes
    """
    from src.lens1.environment import Lens1LondonEnv
    from src.lens3.environment import Lens3NYEnv

    os.makedirs(output_dir, exist_ok=True)
    news_dates = news_dates or set()

    # Create environments
    lens1_env = Lens1LondonEnv(df, trading_days, settings)
    lens3_env = Lens3NYEnv(
        df, trading_days, lens1_model, lens1_env, settings, news_dates,
        lens2_predictor=lens2_predictor, lens2_scaler=lens2_scaler,
        df_m1=df_m1, df_m5=df_m5
    )

    from src.lens3.executor import ScenarioExecutor
    fallback_executor = ScenarioExecutor(settings)

    all_trades = []
    daily_pnl = []

    logger.info(f"Running backtest on {len(trading_days)} trading days...")

    for i, date in enumerate(trading_days):
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)

        # Skip news days
        if date_str in news_dates:
            daily_pnl.append({"date": date_str, "pnl": 0.0, "trades": 0, "news_filtered": True})
            continue

        try:
            # Run Lens 3 episode
            lens3_env.current_day_idx = i
            obs, info = lens3_env.reset()

            done = False
            while not done:
                # Get action masks
                masks = lens3_env.action_masks()
                action, _ = lens3_model.predict(obs, action_masks=masks, deterministic=True)

                # Fallback Executor: If Lens 3 RL engine holds but Predictor says GO strongly
                if action == 0 and lens3_env.current_step == 0:
                    fallback_action = fallback_executor.get_action_for_intent(lens3_env.opening_intent.to_dict())
                    if fallback_action > 0:
                        logger.info(f"[{date_str}] Executor Override: Forcing {fallback_action} entry on {lens3_env.opening_intent.scenario_label}")
                        action = fallback_action

                obs, reward, terminated, truncated, info = lens3_env.step(action)
                done = terminated or truncated

            # Collect results
            for trade in lens3_env.trade_log:
                trade["date"] = date_str
                trade["lens1_encoding"] = lens3_env.lens1_encoding.tolist()
                all_trades.append(trade)

            daily_pnl.append({
                "date": date_str,
                "pnl": lens3_env.episode_pnl,
                "trades": lens3_env.trades_this_episode,
                "wins": lens3_env.wins,
                "losses": lens3_env.losses,
                "news_filtered": False,
            })

        except Exception as e:
            logger.warning(f"Backtest failed for {date_str}: {e}")
            daily_pnl.append({"date": date_str, "pnl": 0.0, "trades": 0, "error": str(e)})

    # Compute summary metrics
    metrics = _compute_metrics(all_trades, daily_pnl)

    # Save outputs
    _save_backtest_results(all_trades, daily_pnl, metrics, output_dir)

    return metrics


def _compute_metrics(trades: List[Dict], daily_pnl: List[Dict]) -> Dict:
    """Compute backtest summary metrics."""
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0}

    trade_df = pd.DataFrame(trades)
    pnl_df = pd.DataFrame(daily_pnl)

    total_trades = len(trades)
    wins = len(trade_df[trade_df["pnl"] > 0])
    losses = len(trade_df[trade_df["pnl"] <= 0])
    win_rate = wins / max(total_trades, 1) * 100

    gross_profit = trade_df[trade_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trade_df[trade_df["pnl"] <= 0]["pnl"].sum())
    profit_factor = gross_profit / max(gross_loss, 0.01)

    total_pnl = trade_df["pnl"].sum()
    avg_r = trade_df["r_multiple"].mean() if "r_multiple" in trade_df.columns else 0

    # Drawdown
    cumulative = trade_df["pnl"].cumsum()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak)
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Sharpe (daily)
    daily_returns = pnl_df[pnl_df["pnl"] != 0]["pnl"]
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    avg_duration = float(trade_df["holding_minutes"].mean()) if "holding_minutes" in trade_df.columns else 0.0

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "avg_r_multiple": avg_r,
        "avg_trade_duration_min": avg_duration,
        "news_filtered_days": sum(1 for d in daily_pnl if d.get("news_filtered", False)),
    }


def _save_backtest_results(
    trades: List[Dict],
    daily_pnl: List[Dict],
    metrics: Dict,
    output_dir: str,
):
    """Save all backtest outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Trade log
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df.to_csv(os.path.join(output_dir, "trade_log.csv"), index=False)

    # Daily PnL
    pnl_df = pd.DataFrame(daily_pnl)
    pnl_df.to_csv(os.path.join(output_dir, "daily_pnl.csv"), index=False)

    # Results markdown
    results_md = f"""# XAUUSD v3 Three-Lens Backtest Results

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics['total_trades']} |
| Win Rate | {metrics['win_rate']:.1f}% |
| Profit Factor | {metrics['profit_factor']:.2f} |
| Total PnL (pts) | {metrics['total_pnl']:.1f} |
| Max Drawdown | {metrics['max_drawdown']:.1f} |
| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |
| Avg R-Multiple | {metrics['avg_r_multiple']:.2f} |
| News Filtered Days | {metrics['news_filtered_days']} |

## Trade Log
See: trade_log.csv

## Daily PnL
See: daily_pnl.csv
"""

    with open(os.path.join(output_dir, "results.md"), "w") as f:
        f.write(results_md)

    # Generate quantstats tearsheet
    try:
        _generate_tearsheet(trades, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate quantstats tearsheet: {e}")

    logger.info(f"Backtest results saved to {output_dir}")
    logger.info(f"  Total trades: {metrics['total_trades']}")
    logger.info(f"  Win rate: {metrics['win_rate']:.1f}%")
    logger.info(f"  Profit factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Total PnL: {metrics['total_pnl']:.1f} pts")


def _generate_tearsheet(trades: List[Dict], output_dir: str):
    """Generate quantstats HTML tearsheet."""
    try:
        import quantstats as qs
    except ImportError:
        logger.warning("quantstats not installed, skipping tearsheet")
        return

    if not trades:
        return

    trade_df = pd.DataFrame(trades)
    if "date" not in trade_df.columns or "pnl" not in trade_df.columns:
        return

    # Create daily returns series
    daily = trade_df.groupby("date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)

    # Convert PnL to returns (assume $100k account for percentage)
    account = 100000
    returns = daily / account

    # Fill missing days with 0
    full_range = pd.date_range(returns.index.min(), returns.index.max(), freq="B")
    returns = returns.reindex(full_range, fill_value=0.0)

    tearsheet_path = os.path.join(output_dir, "tearsheet.html")
    qs.reports.html(returns, output=tearsheet_path, title="XAUUSD v3 Three-Lens Backtest")
    logger.info(f"Quantstats tearsheet saved to {tearsheet_path}")
