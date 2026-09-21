"""
Lens 2 opening-intent helpers.

Turns the frozen Lens 1 snapshot into a richer NY-open playbook by blending:
  - the nearest Lens 1 scenario cluster
  - the explicitly supervised Lens 1 dims 5..7
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OpeningIntent:
    scenario_label: str = "Mixed/Ranging"
    scenario_confidence: float = 0.0
    cluster_id: int = -1
    cluster_distance: float = 0.0
    scenario_source: str = "fallback"
    opening_side_bias: int = 0
    opening_style_bias: float = 0.0
    opening_trade_expected: int = 0
    opening_setup_family: str = "mixed"
    opening_aggression: float = 0.0
    opening_urgency: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class Lens1OpeningIntentMapper:
    """Maps a frozen Lens 1 encoding to an NY-open intent."""

    def __init__(self, settings=None):
        cfg = None if settings is None else settings.lens3
        project_root = Path("." if settings is None else settings.project_root)

        self.enabled = True if cfg is None else cfg.opening_intent_enabled
        self.min_confidence = 0.20 if cfg is None else cfg.opening_intent_min_confidence
        cluster_path = "output/lens1_analysis/lens1_scenario_clusters.csv"
        if cfg is not None:
            cluster_path = cfg.opening_intent_clusters_path

        cluster_file = Path(cluster_path)
        if not cluster_file.is_absolute():
            cluster_file = project_root / cluster_file

        self.cluster_file = cluster_file
        self.cluster_df = self._load_clusters(cluster_file) if self.enabled else pd.DataFrame()

    def classify(self, encoding: np.ndarray, london_bias: int = 0) -> OpeningIntent:
        if encoding is None or len(encoding) == 0:
            return OpeningIntent()

        dim_intent = self._intent_from_encoding_dims(encoding, london_bias)
        if self.enabled and not self.cluster_df.empty:
            cluster_intent = self._classify_from_clusters(encoding, london_bias)
            blended = self._blend_intents(cluster_intent, dim_intent, london_bias)
            if blended.scenario_confidence >= self.min_confidence:
                return blended

        return dim_intent

    def _load_clusters(self, cluster_file: Path) -> pd.DataFrame:
        if not cluster_file.exists():
            logger.warning(
                "Lens 1 scenario cluster file not found at %s. Falling back to heuristic opening intent.",
                cluster_file,
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(cluster_file)
        except Exception as exc:
            logger.warning("Could not read Lens 1 scenario cluster file %s: %s", cluster_file, exc)
            return pd.DataFrame()

        centroid_cols = [c for c in df.columns if c.startswith("centroid_")]
        required = {"cluster", "dominant_scenario"}
        if df.empty or not required.issubset(df.columns) or len(centroid_cols) == 0:
            logger.warning(
                "Lens 1 scenario cluster file %s is missing required columns. Falling back to heuristic intent.",
                cluster_file,
            )
            return pd.DataFrame()

        return df.copy()

    def _classify_from_clusters(self, encoding: np.ndarray, london_bias: int) -> OpeningIntent:
        centroid_cols = [c for c in self.cluster_df.columns if c.startswith("centroid_")]
        centroid_matrix = self.cluster_df[centroid_cols].to_numpy(dtype=float)
        distances = np.linalg.norm(centroid_matrix - encoding.astype(float), axis=1)
        best_idx = int(np.argmin(distances))
        best_row = self.cluster_df.iloc[best_idx]
        best_dist = float(distances[best_idx])

        if len(distances) > 1:
            sorted_dists = np.sort(distances)
            second_best = float(sorted_dists[1])
        else:
            second_best = best_dist + 1.0

        base_conf = 1.0 / (1.0 + best_dist)
        separation = max(second_best - best_dist, 0.0) / max(second_best, 1e-6)
        cluster_weight = float(best_row.get("pct_total", 0.0)) / 100.0
        confidence = float(np.clip((0.55 * base_conf) + (0.35 * separation) + (0.10 * cluster_weight), 0.0, 1.0))
        return self._intent_from_cluster_row(
            best_row,
            confidence=confidence,
            london_bias=london_bias,
            cluster_id=int(best_row.get("cluster", -1)),
            cluster_distance=best_dist,
            source="cluster_csv",
        )

    def _intent_from_encoding_dims(self, encoding: np.ndarray, london_bias: int) -> OpeningIntent:
        directional = float(encoding[5]) if len(encoding) > 5 and abs(float(encoding[5])) >= 0.10 else float(encoding[0])
        style_bias = float(encoding[6]) if len(encoding) > 6 else (-0.4 if abs(directional) >= 0.35 else 0.0)
        urgency = float(np.clip(encoding[7], 0.0, 1.0)) if len(encoding) > 7 else float(np.clip(abs(directional), 0.0, 1.0))
        confidence = float(np.clip((0.45 * abs(directional)) + (0.35 * abs(style_bias)) + (0.20 * urgency), 0.0, 1.0))
        return self._intent_from_signals(
            side_signal=directional,
            style_signal=style_bias,
            confidence=confidence,
            urgency=urgency,
            london_bias=london_bias,
            source="dims",
        )

    def _intent_from_cluster_row(
        self,
        row: pd.Series,
        confidence: float,
        london_bias: int,
        cluster_id: int,
        cluster_distance: float,
        source: str,
    ) -> OpeningIntent:
        label = str(row.get("dominant_scenario", "Mixed/Ranging"))
        side_signal = float(row.get("avg_opening_side_bias", 0.0))
        style_signal = float(row.get("avg_opening_style_bias", 0.0))
        urgency = float(np.clip(row.get("avg_opening_urgency", 0.0), 0.0, 1.0))

        if abs(side_signal) < 0.1:
            if label == "S1_BullContinuation":
                side_signal = 1.0
            elif label == "S2_BearContinuation":
                side_signal = -1.0
            elif label == "S8_LowLiqGrab":
                side_signal = 1.0
            elif label == "S7_HighLiqGrab":
                side_signal = -1.0
            elif label == "S3S4_Continuation":
                side_signal = float(london_bias)

        if abs(style_signal) < 0.1:
            if label in {"S8_LowLiqGrab", "S7_HighLiqGrab"}:
                style_signal = 1.0
            elif label in {"S1_BullContinuation", "S2_BearContinuation", "S3S4_Continuation"}:
                style_signal = -1.0

        return self._intent_from_signals(
            side_signal=side_signal,
            style_signal=style_signal,
            confidence=confidence,
            urgency=urgency,
            london_bias=london_bias,
            cluster_id=cluster_id,
            cluster_distance=cluster_distance,
            source=source,
        )

    def _blend_intents(self, cluster_intent: OpeningIntent, dim_intent: OpeningIntent, london_bias: int) -> OpeningIntent:
        if cluster_intent.scenario_confidence <= 0.0:
            return dim_intent
        if dim_intent.scenario_confidence <= 0.0:
            return cluster_intent

        side_signal = (
            0.60 * cluster_intent.opening_side_bias * max(cluster_intent.scenario_confidence, 0.1)
            + 0.40 * dim_intent.opening_side_bias * max(dim_intent.scenario_confidence, 0.1)
        )
        style_signal = (
            0.60 * cluster_intent.opening_style_bias * max(cluster_intent.scenario_confidence, 0.1)
            + 0.40 * dim_intent.opening_style_bias * max(dim_intent.scenario_confidence, 0.1)
        )
        confidence = float(np.clip(max(cluster_intent.scenario_confidence, dim_intent.scenario_confidence), 0.0, 1.0))
        urgency = float(np.clip((0.55 * cluster_intent.opening_urgency) + (0.45 * dim_intent.opening_urgency), 0.0, 1.0))
        return self._intent_from_signals(
            side_signal=side_signal,
            style_signal=style_signal,
            confidence=confidence,
            urgency=urgency,
            london_bias=london_bias,
            cluster_id=cluster_intent.cluster_id,
            cluster_distance=cluster_intent.cluster_distance,
            source="cluster+dims",
        )

    def _intent_from_signals(
        self,
        side_signal: float,
        style_signal: float,
        confidence: float,
        urgency: float,
        london_bias: int,
        cluster_id: int = -1,
        cluster_distance: float = 0.0,
        source: str = "fallback",
    ) -> OpeningIntent:
        side_signal = float(np.clip(side_signal, -1.0, 1.0))
        style_signal = float(np.clip(style_signal, -1.0, 1.0))
        urgency = float(np.clip(urgency, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if abs(side_signal) < 0.12 and abs(style_signal) < 0.12 and confidence < self.min_confidence:
            return OpeningIntent(
                cluster_id=cluster_id,
                cluster_distance=cluster_distance,
                scenario_source=source,
            )

        side_bias = 0
        if side_signal > 0.12:
            side_bias = 1
        elif side_signal < -0.12:
            side_bias = -1
        elif london_bias != 0 and confidence >= self.min_confidence:
            side_bias = int(np.sign(london_bias))

        label = "Mixed/Ranging"
        setup_family = "mixed"
        trade_expected = 0

        if side_bias != 0 and style_signal >= 0.15:
            label = "S8_LowLiqGrab" if side_bias > 0 else "S7_HighLiqGrab"
            setup_family = "sweep_low" if side_bias > 0 else "sweep_high"
            trade_expected = 1
        elif side_bias != 0 and style_signal <= -0.15:
            label = "S1_BullContinuation" if side_bias > 0 else "S2_BearContinuation"
            setup_family = "continuation"
            trade_expected = 1
        elif side_bias != 0 and confidence >= self.min_confidence:
            label = "S3S4_Continuation"
            setup_family = "continuation"
            trade_expected = 1

        aggression = float(np.clip(confidence * (0.55 + (0.45 * urgency)), 0.0, 1.0))

        return OpeningIntent(
            scenario_label=label,
            scenario_confidence=confidence,
            cluster_id=cluster_id,
            cluster_distance=cluster_distance,
            scenario_source=source,
            opening_side_bias=side_bias,
            opening_style_bias=style_signal,
            opening_trade_expected=trade_expected,
            opening_setup_family=setup_family,
            opening_aggression=aggression,
            opening_urgency=urgency,
        )
