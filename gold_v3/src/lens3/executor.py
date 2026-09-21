"""
Lens 3 — Scenario Aggressor.
A direct mapping layer that ensures Lens 1 scenarios are traded even if Lens 2 hesitates.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ScenarioExecutor:
    """Deterministic executor for Lens 1 Scenarios."""
    
    def __init__(self, settings=None):
        self.settings = settings
        
    def get_action_for_intent(self, intent: Dict) -> Optional[int]:
        """
        Maps OpeningIntent to a direct action.
        Returns: 0 (Hold), 1 (Buy), 2 (Sell)
        """
        side_bias = intent.get("opening_side_bias", 0)
        label = intent.get("scenario_label", "")
        confidence = intent.get("scenario_confidence", 0.0)
        
        # Scenario-specific overrides (The Nuclear Option)
        # S1, S3, S8 are bullish. S2, S4, S7 are bearish.
        if confidence < 0.35:
            return 0  # Still need some confidence
            
        if "BullContinuation" in label or "LowLiqGrab" in label:
            return 1  # BUY
        elif "BearContinuation" in label or "HighLiqGrab" in label:
            return 2  # SELL
        
        # If label is mixed but bias is strong, follow bias
        if abs(side_bias) == 1:
            return 1 if side_bias == 1 else 2
            
        return 0

    def build_protected_trade_plan(self, action: int, price: float, london_profile, atr_val: float) -> Dict:
        """Builds a wide, protected trade plan if Lens 2 fails to provide one."""
        if london_profile is None:
            return None
            
        buffer = 0.05 * london_profile.range_points
        
        if action == 1: # BUY
            sl = london_profile.low - buffer
            # Ensure at least 1.5 ATR for safety
            sl = min(sl, price - (1.5 * atr_val))
            tp1 = london_profile.high
        else: # SELL
            sl = london_profile.high + buffer
            sl = max(sl, price + (1.5 * atr_val))
            tp1 = london_profile.low
            
        return {
            "sl": sl,
            "tp1": tp1,
            "tp2": tp1 + (atr_val if action == 1 else -atr_val),
            "tp3": tp1 + (2 * atr_val if action == 1 else -2 * atr_val)
        }
