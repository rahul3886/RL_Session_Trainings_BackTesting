"""
Lens 1 NY-open target extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.smc.liquidity import detect_sweep


@dataclass
class OpeningOutcome:
    scenario_label: str = "Mixed/Ranging"
    opening_trade_expected: int = 0
    opening_side_bias: int = 0
    opening_style_bias: float = 0.0
    opening_urgency: float = 0.0
    trigger_step: int = -1
    confidence: float = 0.0
    opening_setup_family: str = "mixed"

    def to_targets(self) -> np.ndarray:
        return np.array(
            [
                float(self.opening_side_bias),
                float(self.opening_style_bias),
                float(np.clip(self.opening_urgency, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def to_record(self) -> Dict:
        return asdict(self)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _urgency(trigger_step: int, window_len: int, confidence: float) -> float:
    if trigger_step < 0 or window_len <= 0:
        return 0.0
    early_bonus = 1.0 - (trigger_step / max(window_len - 1, 1))
    return _clamp01((0.55 * confidence) + (0.45 * early_bonus))


def _find_continuation_trigger_step(window: pd.DataFrame, reference_level: float, side: int) -> int:
    for idx in range(len(window)):
        prefix = window.iloc[: idx + 1]
        price = float(prefix["close"].iloc[-1])
        if side > 0 and price > reference_level:
            return idx
        if side < 0 and price < reference_level:
            return idx
    return len(window) - 1


def infer_opening_outcome(
    london_profile,
    ny_candles: pd.DataFrame,
    sweep_threshold_lr: float = 0.12,
    window_candles: int = 4,
) -> OpeningOutcome:
    if london_profile is None or ny_candles is None or len(ny_candles) == 0:
        return OpeningOutcome()

    window = ny_candles.iloc[: max(1, min(window_candles, len(ny_candles)))]
    london_range = max(float(london_profile.range_points), 0.01)
    london_mid = float(london_profile.mid)
    london_high = float(london_profile.high)
    london_low = float(london_profile.low)
    london_open = float(london_profile.open_price)

    for step in range(1, len(window) + 1):
        prefix = window.iloc[:step]
        current_price = float(prefix["close"].iloc[-1])
        sweep = detect_sweep(
            current_price=current_price,
            session_high=london_high,
            session_low=london_low,
            candles_after=prefix,
            threshold_lr=sweep_threshold_lr,
            require_reversal=True,
        )

        if sweep.get("sweep_low"):
            move_strength = max(float(prefix["high"].max() - london_low) / london_range, 0.0)
            confidence = _clamp01(0.55 + (0.30 * move_strength))
            return OpeningOutcome(
                scenario_label="S8_LowLiqGrab",
                opening_trade_expected=1,
                opening_side_bias=1,
                opening_style_bias=1.0,
                opening_urgency=_urgency(step - 1, len(window), confidence),
                trigger_step=step - 1,
                confidence=confidence,
                opening_setup_family="sweep_low",
            )

        if sweep.get("sweep_high"):
            move_strength = max(float(london_high - prefix["low"].min()) / london_range, 0.0)
            confidence = _clamp01(0.55 + (0.30 * move_strength))
            return OpeningOutcome(
                scenario_label="S7_HighLiqGrab",
                opening_trade_expected=1,
                opening_side_bias=-1,
                opening_style_bias=1.0,
                opening_urgency=_urgency(step - 1, len(window), confidence),
                trigger_step=step - 1,
                confidence=confidence,
                opening_setup_family="sweep_high",
            )

    window_high = float(window["high"].max())
    window_low = float(window["low"].min())
    window_close = float(window["close"].iloc[-1])

    bull_break = max((window_high - london_high) / london_range, 0.0)
    bear_break = max((london_low - window_low) / london_range, 0.0)
    bull_close = max((window_close - london_mid) / london_range, 0.0)
    bear_close = max((london_mid - window_close) / london_range, 0.0)
    bull_score = max(bull_break, bull_close)
    bear_score = max(bear_break, bear_close)

    if bull_score >= 0.18 and bull_score >= bear_score:
        trigger_step = _find_continuation_trigger_step(window, london_mid, side=1)
        confidence = _clamp01(0.25 + (0.85 * bull_score) + (0.10 if london_profile.bias >= 0 else 0.0))
        return OpeningOutcome(
            scenario_label="S1_BullContinuation" if london_profile.bias >= 0 else "S3S4_Continuation",
            opening_trade_expected=1 if confidence >= 0.20 else 0,
            opening_side_bias=1,
            opening_style_bias=-1.0,
            opening_urgency=_urgency(trigger_step, len(window), confidence),
            trigger_step=trigger_step,
            confidence=confidence,
            opening_setup_family="continuation",
        )

    if bear_score >= 0.18 and bear_score > bull_score:
        trigger_step = _find_continuation_trigger_step(window, london_mid, side=-1)
        confidence = _clamp01(0.25 + (0.85 * bear_score) + (0.10 if london_profile.bias <= 0 else 0.0))
        return OpeningOutcome(
            scenario_label="S2_BearContinuation" if london_profile.bias <= 0 else "S3S4_Continuation",
            opening_trade_expected=1 if confidence >= 0.20 else 0,
            opening_side_bias=-1,
            opening_style_bias=-1.0,
            opening_urgency=_urgency(trigger_step, len(window), confidence),
            trigger_step=trigger_step,
            confidence=confidence,
            opening_setup_family="continuation",
        )

    directional_open_move = (window_close - london_open) / london_range
    if london_profile.bias > 0 and directional_open_move > 0.10:
        confidence = _clamp01(0.15 + directional_open_move)
        trigger_step = _find_continuation_trigger_step(window, london_open, side=1)
        return OpeningOutcome(
            scenario_label="S3S4_Continuation",
            opening_trade_expected=1 if confidence >= 0.20 else 0,
            opening_side_bias=1,
            opening_style_bias=-0.6,
            opening_urgency=_urgency(trigger_step, len(window), confidence),
            trigger_step=trigger_step,
            confidence=confidence,
            opening_setup_family="continuation",
        )

    if london_profile.bias < 0 and directional_open_move < -0.10:
        confidence = _clamp01(0.15 + abs(directional_open_move))
        trigger_step = _find_continuation_trigger_step(window, london_open, side=-1)
        return OpeningOutcome(
            scenario_label="S3S4_Continuation",
            opening_trade_expected=1 if confidence >= 0.20 else 0,
            opening_side_bias=-1,
            opening_style_bias=-0.6,
            opening_urgency=_urgency(trigger_step, len(window), confidence),
            trigger_step=trigger_step,
            confidence=confidence,
            opening_setup_family="continuation",
        )

    return OpeningOutcome()
