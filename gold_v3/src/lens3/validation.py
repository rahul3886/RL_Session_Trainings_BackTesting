"""
Lens 2 validation helpers.

Runs deterministic validation episodes on the held-out split and computes
trading metrics that are more meaningful than PPO reward alone.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_lens3_structural(
    model,
    env,
    settings=None,
    output_dir: Optional[str] = None,
    label: str = "validation",
) -> Dict:
    """
    Evaluate a Lens 2 model on a deterministic pass over the env's trading days.
    """
    daily_records: List[Dict] = []
    all_trades: List[Dict] = []
    news_dates = getattr(env, "news_dates", set()) or set()
    original_env_idx = env.current_day_idx
    original_env_order = list(env._shuffled_days)

    lens1_env = getattr(env, "lens1_env", None)
    original_lens1_idx = None
    original_lens1_order = None
    if lens1_env is not None:
        original_lens1_idx = lens1_env.current_day_idx
        original_lens1_order = list(lens1_env._shuffled_days)
        lens1_env._shuffled_days = list(lens1_env.trading_days)

    try:
        env._shuffled_days = list(env.trading_days)

        for day_idx, date in enumerate(env.trading_days):
            date_str = str(date.date()) if hasattr(date, "date") else str(date)
            if date_str in news_dates:
                daily_records.append({
                    "date": date_str,
                    "news_filtered": True,
                    "pnl": 0.0,
                    "trades": 0,
                    "sweep_day": 0,
                    "sweep_high_day": 0,
                    "sweep_low_day": 0,
                    "sweep_direction": "none",
                    "ny_open_trade_taken": 0,
                    "first_trade_pnl": np.nan,
                    "opening_scenario_label": "NewsFiltered",
                    "opening_scenario_confidence": 0.0,
                    "opening_setup_family": "none",
                    "opening_trade_expected": 0,
                    "opening_side_bias": 0,
                    "first_trade_entry_step": np.nan,
                    "first_trade_entry_time": None,
                    "first_trade_entry_delay_from_anchor_min": np.nan,
                    "first_trade_entry_delay_from_ny_open_min": np.nan,
                    "first_trade_duration_min": np.nan,
                    "first_trade_is_ny_open": 0,
                    "first_trade_matches_intent": 0,
                })
                continue

            env.current_day_idx = day_idx
            obs, info = env.reset()
            done = False
            last_info = info

            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, _, terminated, truncated, last_info = env.step(action)
                done = terminated or truncated

            for trade in env.trade_log:
                all_trades.append(dict(trade))

            daily_records.append({
                "date": date_str,
                "news_filtered": False,
                "pnl": float(last_info.get("pnl", 0.0)),
                "trades": int(last_info.get("trades", 0)),
                "sweep_day": int(last_info.get("sweep_day", 0)),
                "sweep_high_day": int(last_info.get("sweep_high_day", 0)),
                "sweep_low_day": int(last_info.get("sweep_low_day", 0)),
                "sweep_direction": last_info.get("sweep_direction", "none"),
                "ny_open_trade_taken": int(last_info.get("ny_open_trade_taken", 0)),
                "first_trade_pnl": float(last_info.get("first_trade_pnl", np.nan)),
                "london_bias": int(last_info.get("london_bias", 0)),
                "opening_scenario_label": last_info.get("opening_scenario_label", "Mixed/Ranging"),
                "opening_scenario_confidence": float(last_info.get("opening_scenario_confidence", 0.0)),
                "opening_setup_family": last_info.get("opening_setup_family", "mixed"),
                "opening_trade_expected": int(last_info.get("opening_trade_expected", 0)),
                "opening_side_bias": int(last_info.get("opening_side_bias", 0)),
                "first_trade_entry_step": float(last_info.get("first_trade_entry_step", np.nan)),
                "first_trade_entry_time": last_info.get("first_trade_entry_time"),
                "first_trade_entry_delay_from_anchor_min": float(last_info.get("first_trade_entry_delay_from_anchor_min", np.nan)),
                "first_trade_entry_delay_from_ny_open_min": float(last_info.get("first_trade_entry_delay_from_ny_open_min", np.nan)),
                "first_trade_duration_min": float(last_info.get("first_trade_duration_min", np.nan)),
                "first_trade_is_ny_open": int(last_info.get("first_trade_is_ny_open", 0)),
                "first_trade_matches_intent": int(last_info.get("first_trade_matches_intent", 0)),
            })
    finally:
        env.current_day_idx = original_env_idx
        env._shuffled_days = original_env_order
        if lens1_env is not None:
            lens1_env.current_day_idx = original_lens1_idx
            lens1_env._shuffled_days = original_lens1_order

    metrics = _summarise_validation(all_trades, daily_records, settings)

    if output_dir is not None:
        _save_validation_outputs(output_dir, label, metrics, all_trades, daily_records)

    return metrics


def _summarise_validation(trades: List[Dict], daily_records: List[Dict], settings=None) -> Dict:
    lens3_cfg = None if settings is None else settings.lens3
    min_win_rate = 55.0 if lens3_cfg is None else lens3_cfg.structural_min_win_rate
    min_profit_factor = 1.8 if lens3_cfg is None else lens3_cfg.structural_min_profit_factor
    min_trades_per_episode = 1.0 if lens3_cfg is None else lens3_cfg.structural_min_trades_per_episode
    min_pnl_sweep_low_days = 0.0 if lens3_cfg is None else lens3_cfg.structural_min_pnl_sweep_low_days
    min_ny_open_trade_rate = 0.0 if lens3_cfg is None else lens3_cfg.structural_min_ny_open_trade_rate
    min_first_trade_intent_match_rate = 0.0 if lens3_cfg is None else lens3_cfg.structural_min_first_trade_intent_match_rate
    max_first_trade_delay = 1000000.0 if lens3_cfg is None else lens3_cfg.structural_max_first_trade_entry_delay_from_anchor_min

    daily_df = pd.DataFrame(daily_records)
    if len(daily_df) == 0:
        metrics = {
            "episodes": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades_per_episode": 0.0,
            "acceptance_passed": False,
        }
        metrics["acceptance"] = _acceptance_dict(
            metrics,
            min_win_rate,
            min_profit_factor,
            min_trades_per_episode,
            min_pnl_sweep_low_days,
            min_ny_open_trade_rate,
            min_first_trade_intent_match_rate,
            max_first_trade_delay,
        )
        return metrics

    eval_days = daily_df[~daily_df["news_filtered"]].copy()
    trade_df = pd.DataFrame(trades)

    total_trades = len(trade_df)
    wins = int((trade_df["pnl"] > 0).sum()) if total_trades > 0 else 0
    losses = int((trade_df["pnl"] <= 0).sum()) if total_trades > 0 else 0
    win_rate = (wins / max(total_trades, 1)) * 100.0
    gross_profit = float(trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum()) if total_trades > 0 else 0.0
    gross_loss = abs(float(trade_df.loc[trade_df["pnl"] <= 0, "pnl"].sum())) if total_trades > 0 else 0.0
    profit_factor = gross_profit / max(gross_loss, 0.01) if total_trades > 0 else 0.0
    trades_per_episode = float(eval_days["trades"].mean()) if len(eval_days) > 0 else 0.0

    sweep_high_days = eval_days[eval_days["sweep_high_day"] > 0]
    sweep_low_days = eval_days[eval_days["sweep_low_day"] > 0]
    sweep_days = eval_days[eval_days["sweep_day"] > 0]
    continuation_days = eval_days[eval_days["sweep_day"] == 0]
    first_trade_days = eval_days[eval_days["first_trade_pnl"].notna()]
    if total_trades > 0 and "is_first_trade" in trade_df.columns:
        first_trade_df = trade_df[trade_df["is_first_trade"] > 0].copy()
    else:
        first_trade_df = pd.DataFrame()

    if len(first_trade_df) > 0 and "opened_in_ny_open" in first_trade_df.columns:
        ny_open_first_trade_df = first_trade_df[first_trade_df["opened_in_ny_open"] > 0].copy()
    else:
        ny_open_first_trade_df = pd.DataFrame()

    scenario_summary = {}
    for scenario_label, scenario_df in eval_days.groupby("opening_scenario_label"):
        if len(first_trade_df) > 0 and "opening_scenario_label" in first_trade_df.columns:
            first_scenario = first_trade_df[first_trade_df["opening_scenario_label"] == scenario_label]
        else:
            first_scenario = pd.DataFrame()
        scenario_summary[str(scenario_label)] = {
            "days": int(len(scenario_df)),
            "ny_open_trade_rate": float(scenario_df["ny_open_trade_taken"].mean()) if len(scenario_df) > 0 else 0.0,
            "first_trade_rate": float(scenario_df["first_trade_pnl"].notna().mean()) if len(scenario_df) > 0 else 0.0,
            "first_trade_is_ny_open_rate": float(scenario_df["first_trade_is_ny_open"].mean()) if len(scenario_df) > 0 else 0.0,
            "first_trade_matches_intent_rate": float(scenario_df["first_trade_matches_intent"].mean()) if len(scenario_df) > 0 else 0.0,
            "first_trade_win_rate": float((first_scenario["pnl"] > 0).mean() * 100.0) if len(first_scenario) > 0 else 0.0,
        }

    metrics = {
        "episodes": int(len(eval_days)),
        "news_filtered_days": int(daily_df["news_filtered"].sum()),
        "total_trades": int(total_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "trades_per_episode": float(trades_per_episode),
        "avg_episode_pnl": _safe_mean(eval_days["pnl"]) if len(eval_days) > 0 else 0.0,
        "total_episode_pnl": float(eval_days["pnl"].sum()) if len(eval_days) > 0 else 0.0,
        "pnl_sweep_high_days": _safe_mean(sweep_high_days["pnl"]),
        "pnl_sweep_low_days": _safe_mean(sweep_low_days["pnl"]),
        "pnl_sweep_days": _safe_mean(sweep_days["pnl"]),
        "pnl_continuation_days": _safe_mean(continuation_days["pnl"]),
        "win_rate_sweep_days": _episode_win_rate(sweep_days),
        "win_rate_continuation": _episode_win_rate(continuation_days),
        "ny_open_trade_rate": float(eval_days["ny_open_trade_taken"].mean()) if len(eval_days) > 0 else 0.0,
        "first_trade_avg_pnl": _safe_mean(first_trade_days["first_trade_pnl"]),
        "first_trade_win_rate": float((first_trade_days["first_trade_pnl"] > 0).mean() * 100.0) if len(first_trade_days) > 0 else 0.0,
        "first_trade_is_ny_open_rate": float(first_trade_days["first_trade_is_ny_open"].mean()) if len(first_trade_days) > 0 else 0.0,
        "first_trade_matches_intent_rate": float(first_trade_days["first_trade_matches_intent"].mean()) if len(first_trade_days) > 0 else 0.0,
        "avg_first_trade_entry_step": _safe_mean(first_trade_days["first_trade_entry_step"]),
        "avg_first_trade_entry_delay_from_anchor_min": _safe_mean(first_trade_days["first_trade_entry_delay_from_anchor_min"]),
        "avg_first_trade_entry_delay_from_ny_open_min": _safe_mean(first_trade_days["first_trade_entry_delay_from_ny_open_min"]),
        "avg_first_trade_duration_min": _safe_mean(first_trade_days["first_trade_duration_min"]),
        "ny_open_first_trade_win_rate": float((ny_open_first_trade_df["pnl"] > 0).mean() * 100.0) if len(ny_open_first_trade_df) > 0 else 0.0,
        "ny_open_first_trade_profit_factor": _profit_factor(ny_open_first_trade_df),
        "scenario_counts": {
            "sweep_high_days": int(len(sweep_high_days)),
            "sweep_low_days": int(len(sweep_low_days)),
            "sweep_days": int(len(sweep_days)),
            "continuation_days": int(len(continuation_days)),
            "both_sweep_days": int((eval_days["sweep_direction"] == "both").sum()),
        },
        "opening_scenario_counts": {
            str(label): int(count)
            for label, count in eval_days["opening_scenario_label"].value_counts().to_dict().items()
        },
        "opening_scenario_summary": scenario_summary,
    }

    acceptance = _acceptance_dict(
        metrics,
        min_win_rate,
        min_profit_factor,
        min_trades_per_episode,
        min_pnl_sweep_low_days,
        min_ny_open_trade_rate,
        min_first_trade_intent_match_rate,
        max_first_trade_delay,
    )
    metrics["acceptance"] = acceptance
    metrics["acceptance_passed"] = all(acceptance.values())
    return metrics


def _acceptance_dict(
    metrics: Dict,
    min_win_rate: float,
    min_profit_factor: float,
    min_trades_per_episode: float,
    min_pnl_sweep_low_days: float,
    min_ny_open_trade_rate: float,
    min_first_trade_intent_match_rate: float,
    max_first_trade_delay: float,
) -> Dict:
    return {
        "win_rate": metrics.get("win_rate", 0.0) >= min_win_rate,
        "profit_factor": metrics.get("profit_factor", 0.0) >= min_profit_factor,
        "trades_per_episode": metrics.get("trades_per_episode", 0.0) >= min_trades_per_episode,
        "pnl_sweep_low_days": metrics.get("pnl_sweep_low_days", 0.0) > min_pnl_sweep_low_days,
        "ny_open_trade_rate": metrics.get("ny_open_trade_rate", 0.0) >= min_ny_open_trade_rate,
        "first_trade_matches_intent_rate": metrics.get("first_trade_matches_intent_rate", 0.0) >= min_first_trade_intent_match_rate,
        "first_trade_delay": metrics.get("avg_first_trade_entry_delay_from_anchor_min", 1000000.0) <= max_first_trade_delay,
    }


def _episode_win_rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float((df["pnl"] > 0).mean() * 100.0)


def _safe_mean(series) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.nanmean(series))


def _profit_factor(trade_df: pd.DataFrame) -> float:
    if len(trade_df) == 0 or "pnl" not in trade_df.columns:
        return 0.0
    gross_profit = float(trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trade_df.loc[trade_df["pnl"] <= 0, "pnl"].sum()))
    return gross_profit / max(gross_loss, 0.01)


def _save_validation_outputs(
    output_dir: str,
    label: str,
    metrics: Dict,
    trades: List[Dict],
    daily_records: List[Dict],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{label}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(daily_records).to_csv(
        os.path.join(output_dir, f"{label}_daily.csv"),
        index=False,
    )
    pd.DataFrame(trades).to_csv(
        os.path.join(output_dir, f"{label}_trades.csv"),
        index=False,
    )

    report = [
        f"# Lens 2 Structural Validation - {label}",
        "",
        f"- Episodes: {metrics['episodes']}",
        f"- Total trades: {metrics['total_trades']}",
        f"- Win rate: {metrics['win_rate']:.1f}%",
        f"- Profit factor: {metrics['profit_factor']:.2f}",
        f"- Trades/episode: {metrics['trades_per_episode']:.2f}",
        f"- Avg episode PnL: {metrics['avg_episode_pnl']:.2f}",
        f"- Sweep-high day PnL: {metrics['pnl_sweep_high_days']:.2f}",
        f"- Sweep-low day PnL: {metrics['pnl_sweep_low_days']:.2f}",
        f"- Continuation day PnL: {metrics['pnl_continuation_days']:.2f}",
        f"- NY-open trade rate: {metrics['ny_open_trade_rate']:.1%}",
        f"- First-trade win rate: {metrics['first_trade_win_rate']:.1f}%",
        f"- First trade that is NY-open: {metrics['first_trade_is_ny_open_rate']:.1%}",
        f"- First-trade intent match rate: {metrics['first_trade_matches_intent_rate']:.1%}",
        f"- Avg first-trade entry delay from anchor: {metrics['avg_first_trade_entry_delay_from_anchor_min']:.1f} min",
        f"- Avg first-trade entry delay from NY open: {metrics['avg_first_trade_entry_delay_from_ny_open_min']:.1f} min",
        f"- Avg first-trade duration: {metrics['avg_first_trade_duration_min']:.1f} min",
        f"- NY-open first-trade win rate: {metrics['ny_open_first_trade_win_rate']:.1f}%",
        f"- Acceptance passed: {metrics['acceptance_passed']}",
    ]
    with open(os.path.join(output_dir, f"{label}_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    logger.info(
        "Lens 2 structural validation %s: win_rate=%.1f%% pf=%.2f trades/ep=%.2f accepted=%s",
        label,
        metrics["win_rate"],
        metrics["profit_factor"],
        metrics["trades_per_episode"],
        metrics["acceptance_passed"],
    )
