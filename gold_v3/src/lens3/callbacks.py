"""
Lens 2 Callbacks — TensorBoard metrics logging.
Separated from agent.py for clarity.
"""

import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """
    Extended TensorBoard metrics for Lens 2.
    Logs SMC-specific entry quality metrics.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ob_entries = 0
        self.fvg_entries = 0
        self.sweep_entries = 0
        self.lens1_aligned_entries = 0
        self.total_entries = 0
        self.r_multiples = []
        self.trade_durations = []

    def _on_step(self) -> bool:
        # Log every 10000 steps
        if self.num_timesteps % 10000 == 0 and self.total_entries > 0:
            self.logger.record("smc/ob_entry_rate",
                               self.ob_entries / max(self.total_entries, 1))
            self.logger.record("smc/fvg_entry_rate",
                               self.fvg_entries / max(self.total_entries, 1))
            self.logger.record("smc/sweep_entry_rate",
                               self.sweep_entries / max(self.total_entries, 1))
            self.logger.record("lens1/alignment_rate",
                               self.lens1_aligned_entries / max(self.total_entries, 1))

            if self.r_multiples:
                self.logger.record("trade/avg_r_multiple", np.mean(self.r_multiples[-50:]))
                self.logger.record("trade/profit_factor",
                                   self._compute_profit_factor(self.r_multiples[-50:]))

        return True

    @staticmethod
    def _compute_profit_factor(r_multiples: list) -> float:
        """Compute profit factor from R-multiples."""
        gross_profit = sum(r for r in r_multiples if r > 0)
        gross_loss = abs(sum(r for r in r_multiples if r < 0))
        return gross_profit / max(gross_loss, 0.01)
