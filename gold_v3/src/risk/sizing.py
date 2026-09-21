"""
Position sizing — risk-based lot calculation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    point_value: float = 1.0,
) -> float:
    """
    Compute position size based on risk parameters.

    Args:
        account_balance: current account balance
        risk_pct: fraction of account to risk (e.g., 0.01 = 1%)
        entry_price: proposed entry price
        sl_price: stop loss price
        point_value: value per point per lot (XAUUSD = varies by broker)

    Returns:
        Position size in lots
    """
    risk_amount = account_balance * risk_pct
    sl_distance = abs(entry_price - sl_price)

    if sl_distance <= 0:
        logger.warning("SL distance is zero, returning minimum size")
        return 0.01

    size = risk_amount / (sl_distance * point_value)
    size = max(size, 0.01)  # minimum lot

    return round(size, 2)
